"""
Safety-WaRP-LLM: 완전히 수정된 버전 (원본 FSCIL-WaRP 방식)

원본 FSCIL-WaRP와 동일한 플로우:
    Phase 0: Base Safety Training (새로 추가!)
    Phase 1: Basis Construction  
    Phase 2: Importance Scoring (eval 모드, optimizer.step 제거)
    Phase 3: Incremental Learning (WaRP 모듈, 마스크 적용)

주요 변경사항:
✅ Phase 0 추가: 안전 데이터로 실제 모델 학습
✅ Phase 1 수정: Φ @ Φ^T 방식으로 SVD
✅ Phase 2 완전 재작성: model.eval() + gradient만 계산
✅ Phase 3 수정: WaRP 모듈로 레이어 교체, 마스크 적용
"""

import os
import argparse
import logging
from contextlib import contextmanager
from datetime import datetime

from utils import setup_logger, set_seed, ResourceProfiler


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Safety-WaRP-LLM'
    )
    
    # Phase 설정
    parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2, 3],
                        help='실행할 phase (0: Base Training, 1: Basis, 2: Importance, 3: Learning)')
    
    # 모델 설정
    parser.add_argument('--model_name', type=str, help='사용할 LLM 모델')
    
    # 데이터 설정
    parser.add_argument('--batch_size', type=int, default=4,
                        help='배치 크기')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='토큰 최대 길이 (Phase 2/3 데이터 전처리)')
    
    
    # Phase 1 설정
    parser.add_argument('--phase0_model_dir', type=str, default=None,
                        help='Phase 0에서 학습된 모델 경로 (Phase 1, 2, 3에서 사용)')
    parser.add_argument('--safety_dataset', type=str, default='circuit_breakers',
                        choices=['circuit_breakers', 'wikipedia'],
                        help='Basis 구성용 데이터셋 (Phase 1) - circuit_breakers(Safety), wikipedia(Utility)')
    parser.add_argument('--circuit_breakers_samples_phase1', type=int, default=4994,
                        help='Circuit breakers 샘플 수 (Phase 1 basis 구성용 - Safety Basis)')
    parser.add_argument('--wikipedia_samples_phase1', type=int, default=1000,
                        help='Wikipedia 샘플 수 (Phase 1 basis 구성용 - Utility Basis)')
    parser.add_argument('--only_prompt', action='store_true',
                        help='Phase 1에서 response 없이 prompt(harmful question)만 사용하여 basis 구성. '
                             'instruct 모델은 user turn만, plain 모델은 question만 입력.')

    # ───────────────────────────────────────────────────────────────────
    # WSR-Tune vs ActSVD mask-structure ablation (rebuttal 실험)
    #   설계 문서: wsr_actsvd_ablation_spec.md
    #   Phase 1: --basis_side 로 입력측(WSR) / 출력측(ActSVD) 기저를 선택
    #   Phase 2/3: --ablation_arm 으로 A/B/C/D arm을 선택 (basis+mask 구조만 달라짐)
    # ───────────────────────────────────────────────────────────────────
    parser.add_argument('--basis_side', type=str, default=None,
                        choices=['input', 'output'],
                        help='[ablation] Phase 1 기저 방향. input=X_in X_in^T 고유기저(WSR-Tune, n×n), '
                             'output=W X_in 의 left singular vectors(ActSVD, m×m). '
                             '미지정 시 기존 Phase1BasisBuilder(=input, 레거시 경로) 사용.')
    parser.add_argument('--basis_token_scope', type=str, default='all',
                        choices=['all', 'response'],
                        help='[ablation] 기저 구성에 쓸 토큰 범위. all=패딩 제외 전체(기존 동작, '
                             '논문 Table 2/5 재현용 기본값), response=응답 토큰만(spec §2).')
    parser.add_argument('--basis_save_dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16'],
                        help='[ablation] 기저 저장 dtype. Phase 2/3이 어차피 모델 dtype으로 '
                             '캐스팅하므로 bf16 저장이 기본 (spec §6: 디스크/메모리 절감).')
    parser.add_argument('--gram_dtype', type=str, default='float32',
                        choices=['float32', 'bfloat16'],
                        help='[ablation] 활성화 Gram 누적 dtype. 기본 float32(정확), '
                             'bfloat16=레거시 Phase 1과 동일(메모리 절반).')
    parser.add_argument('--ablation_arm', type=str, default=None,
                        choices=['A', 'B', 'C', 'D', 'D_perm'],
                        help='[ablation] Phase 2/3 arm. A=원본공간+entry, B=ActSVD출력기저+row, '
                             'C=safety입력기저+column, D=WSR-Tune(입력기저+entry), '
                             'D_perm=signed-permutation V sanity arm. '
                             '모든 arm이 safety 데이터(circuit_breakers)만 사용한다.')
    parser.add_argument('--mask_unit', type=str, default=None,
                        choices=['entry', 'row', 'column'],
                        help='[ablation] arm 기본 마스크 단위를 덮어씀 (교차 조합 실험용).')
    parser.add_argument('--structured_agg', type=str, default='l2',
                        choices=['l2', 'sum', 'mean'],
                        help='[ablation] row/column 마스크의 집계 방식 (기본 L2).')
    parser.add_argument('--structured_rank', type=str, default='grad',
                        choices=['grad', 'spectral'],
                        help='[ablation] row/column 랭킹 기준. grad=safety gradient 크기(기본, '
                             'arm 간 유일 변수를 basis/구조로 유지), spectral=기저 특이값 순서'
                             '(ActSVD 원논문 기준을 그대로 옮긴 변형).')

    # Phase 2 설정
    parser.add_argument('--basis_dir', type=str, default=None,
                        help='Phase 1의 basis 디렉토리 경로 (Phase 2, 3에서 사용)')
    parser.add_argument('--dataset_phase2', type=str, default='circuit_breakers',
                        choices=['circuit_breakers', 'wikipedia'],
                        help='Phase 2 importance score 계산용 데이터셋 - circuit_breakers(Safety), wikipedia(Utility)')
    parser.add_argument('--circuit_breakers_path', type=str,
                        default='./data/circuit_breakers_train.json',
                        help='Circuit Breakers JSON 파일 경로 (Phase 2, 3에서 사용)')
    parser.add_argument('--circuit_breakers_samples_phase2', type=int, default=4994,
                        help='Circuit Breakers 샘플 수 (Phase 2 importance scoring용)')
    parser.add_argument('--wikipedia_samples_phase2', type=int, default=4994,
                        help='Wikipedia 샘플 수 (Phase 2 importance scoring용 - Utility)')
    parser.add_argument('--keep_ratio', type=float, default=0.1,
                        help='유지할 중요 파라미터 비율 (Phase 2)')
    parser.add_argument('--perlayer', action='store_true',
                        help='Phase 2에서 layer별 keep_ratio 적용')
    parser.add_argument('--no_rotation', action='store_true',
                        help='Phase 2/3에서 Phase 1 basis 없이 no-rotation(identity basis) 실험 수행')
    # Phase 1 activation 수집 granularity
    parser.add_argument('--basis_granularity', type=str, default='token',
                        choices=['token', 'sequence'],
                        help='Phase 1 활성화 수집 단위: token(기본, 모든 토큰 위치) 또는 sequence(시퀀스별 pooling)')
    parser.add_argument('--seq_pool', type=str, default='mean',
                        choices=['mean', 'last', 'sum'],
                        help='--basis_granularity sequence 일 때 시퀀스 pooling 방식 (mean/last/sum)')
    parser.add_argument('--original_space_mask', action='store_true',
                        help='Phase 2/3에서 basis/WaRP 없이 original weight space importance mask 사용')

    # Phase 2 Two-Mask 설정
    parser.add_argument('--two_mask', action='store_true',
                        help=(
                            '[Two-Mask] Phase 2에서 두 개의 importance mask를 계산하여 '
                            'final_mask = preserve_mask AND NOT adapt_mask로 생성. '
                            'adapt에도 중요한 파라미터를 Phase 3에서 학습 가능하게 허용.'
                        ))
    parser.add_argument('--adapt_dataset_phase2', type=str, default='gsm8k',
                        choices=['gsm8k', 'math', 'metamath', 'wikipedia', 'safety'],
                        help='[Two-Mask] Phase 2 adapt importance scoring용 데이터셋 '
                             '(safety=circuit_breakers, 예: wikipedia mask - safety mask)')
    parser.add_argument('--adapt_samples_phase2', type=int, default=0,
                        help='[Two-Mask] adapt 데이터셋 샘플 수 (0=전체)')

    # Phase 3 설정
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Phase 2의 masks 디렉토리 경로 (Phase 3에서 사용)')
    parser.add_argument('--phase3_dataset', type=str, default='gsm8k',
                        choices=['gsm8k', 'safety', 'metamath', 'math', 'mmlu', 'swebench', 'agnews', 'sst2', 'medqa', 'mbpp', 'arc'],
                        help='Phase 3 finetuning용 데이터셋 - gsm8k(Utility), safety(안전성 강화), metamath(고급 수학), math(Hendrycks MATH), mmlu(MMLU MCQ), swebench(소프트웨어 엔지니어링), agnews(뉴스 분류), medqa(의료 USMLE MCQ), mbpp(파이썬 프로그래밍 문제), arc(ARC-Challenge MCQ)')
    parser.add_argument('--gsm8k_samples', type=int, default=1000,
                        help='GSM8K 샘플 수 (Phase 3 - GSM8K 선택시만 사용)')
    parser.add_argument('--metamath_samples', type=int, default=0,
                        help='MetaMath 샘플 수 (Phase 3 - MetaMath 선택시만 사용, 0=전체)')
    parser.add_argument('--math_samples', type=int, default=0,
                        help='Hendrycks MATH 샘플 수 (Phase 3 - MATH 선택시만 사용, 0=전체)')
    parser.add_argument('--mmlu_subject', type=str, default='all',
                        help='MMLU 과목 (all 또는 단일 subject)')
    parser.add_argument('--mmlu_split', type=str, default='auxiliary_train',
                        choices=['auxiliary_train', 'train', 'validation', 'test', 'dev'],
                        help='MMLU 학습 split')
    parser.add_argument('--mmlu_samples', type=int, default=10000,
                        help='MMLU 샘플 수 (0=전체)')
    parser.add_argument('--math_subjects', type=str, default='all',
                        help='Hendrycks MATH 과목 필터 (예: Algebra,Geometry 또는 all)')
    parser.add_argument('--math_levels', type=str, default='all',
                        help='Hendrycks MATH 난이도 필터 (예: 1,2,3 또는 all)')
    parser.add_argument('--math_dataset_source', type=str, default='official',
                        choices=['official', 'flat_competition_math'],
                        help='MATH 데이터 소스 (official=EleutherAI/hendrycks_math, flat=qwedsacf/competition_math)')
    parser.add_argument('--math_official_dataset_path', type=str, default='EleutherAI/hendrycks_math',
                        help='공식 Hendrycks MATH 데이터셋 경로')
    parser.add_argument('--math_flat_dataset_path', type=str, default='qwedsacf/competition_math',
                        help='flat competition_math 데이터셋 경로')
    parser.add_argument('--math_train_on_mixed_formats', action='store_true',
                        help='MATH 타겟을 long/short/minimal 포맷 혼합으로 구성')
    parser.add_argument('--math_use_chat_template', action='store_true',
                        help='MATH 데이터 전처리 시 tokenizer chat template 사용')
    parser.add_argument('--math_system_prompt', type=str,
                        default='You are a careful competition math solver. Solve the problem step by step. On the last line, write exactly one final answer in the form: Final Answer: $<answer>$. Do not use additional dollar signs earlier in the response.',
                        help='MATH chat template 사용 시 시스템 프롬프트')
    parser.add_argument('--agnews_dataset_path', type=str, default=None,
                        help='AG News 데이터셋 경로 (Phase 3 - AG News 선택시 필수)')
    parser.add_argument('--agnews_split', type=str, default='train',
                        help='AG News split 이름 (default: train)')
    parser.add_argument('--agnews_samples', type=int, default=8000,
                        help='AG News 샘플 수 (0=전체)')
    parser.add_argument('--sst2_dataset_path', type=str, default=None,
                        help='SST-2 로컬 태스크 JSON 경로 (미지정 시 data/sst2_train_8k_seed42.json)')
    parser.add_argument('--sst2_samples', type=int, default=0,
                        help='SST-2 샘플 수 (0=전체)')
    parser.add_argument('--swebench_dataset_path', type=str, default=None,
                        help='SWE-bench 데이터셋 경로 (Phase 3 - SWE-bench 선택시 필수)')
    parser.add_argument('--swebench_split', type=str, default='train',
                        help='SWE-bench split 이름 (default: train)')
    parser.add_argument('--swebench_samples', type=int, default=8000,
                        help='SWE-bench 샘플 수 (0=전체)')
    parser.add_argument('--medqa_dataset_path', type=str, default=None,
                        help='MedQA JSONL 경로 (Phase 3 - medqa 선택시 필수, prepare_medqa_dataset.py 출력)')
    parser.add_argument('--medqa_split', type=str, default='train',
                        help='MedQA split 이름 (default: train, 현재 미사용)')
    parser.add_argument('--medqa_samples', type=int, default=10000,
                        help='MedQA 학습 샘플 수 (0=전체)')
    parser.add_argument('--mbpp_dataset_name', type=str, default='google-research-datasets/mbpp',
                        help='MBPP HuggingFace 데이터셋 이름 (Phase 3 - MBPP 선택시 사용)')
    parser.add_argument('--mbpp_subset', type=str, default='full',
                        help='MBPP subset/config (full | sanitized)')
    parser.add_argument('--mbpp_train_split', type=str, default='train',
                        help='MBPP 학습 split 이름 (default: train)')
    parser.add_argument('--mbpp_samples', type=int, default=0,
                        help='MBPP 학습 샘플 수 (0=전체)')
    parser.add_argument('--arc_dataset_name', type=str, default='allenai/ai2_arc',
                        help='ARC HuggingFace 데이터셋 이름 (Phase 3 - ARC 선택시 사용)')
    parser.add_argument('--arc_subset', type=str, default='ARC-Challenge',
                        help='ARC subset/config (ARC-Challenge | ARC-Easy)')
    parser.add_argument('--arc_train_split', type=str, default='train',
                        help='ARC 학습 split 이름 (default: train)')
    parser.add_argument('--arc_samples', type=int, default=0,
                        help='ARC 학습 샘플 수 (0=전체, ARC-Challenge train은 1119개)')
    parser.add_argument('--circuit_breakers_samples_phase3', type=int, default=4994,
                        help='Circuit Breakers 샘플 수 (Phase 3 - Safety 선택시만 사용)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='훈련 에포크 (Phase 3)')
    parser.add_argument('--utility_lr', type=float, default=1e-5,
                        help='학습률 (Phase 3)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation 스텝 수 (Phase 3)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='LR warmup 비율 (Phase 3)')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        help='LR scheduler 타입 (Phase 3)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient clipping max norm (Phase 3)')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Trainer logging 주기 (Phase 3)')
    parser.add_argument('--base_weight_decay', type=float, default=0.01,
                        help='Weight decay (Phase 3)')
    parser.add_argument('--warp_monitor_samples_per_group', type=int, default=4,
                        help='WaRP monitor 샘플 수 (Phase 3 sanity check용)')
    parser.add_argument('--non_freeze', action='store_true',
                        help='Phase 3에서 WaRP 비적용 레이어를 포함해 나머지 파라미터도 학습')
    parser.add_argument('--no_masks', action='store_true',
                        help='Phase 3에서 Phase 2 마스크 없이 실행 (freeze 없음, 모든 파라미터 학습 가능)')
    parser.add_argument('--safety_mix_ratio', type=float, default=0.0,
                        help='Phase 3에서 safety dataset 혼합 비율 (0.0=미사용, 0.05=downstream 대비 5%%)')
    parser.add_argument('--mix_response_field', type=str, default='llama3_output',
                        help='혼합 데이터에서 응답으로 쓸 JSON 필드명. '
                             'llama3_output=거부 응답(SafeInstr, 기본값), output=유해 응답(harmful FT 공격 실험). '
                             'prompt 필드는 --mix_prompt_field 로 변경')
    parser.add_argument('--mix_prompt_field', type=str, default='prompt',
                        help='혼합 데이터에서 질문으로 쓸 JSON 필드명 (기본 prompt)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Phase 3에서 gradient checkpointing 사용 (비교 실험 시 freeze/non-freeze 동일하게 설정 권장)')

    # ───────────────────────────────────────────────────────────────────
    # Constrained SFT (WSR-Tune + token-wise constrained loss 결합용)
    #   Phase 3(non_freeze)에서 표준 CE 대신 shallow-vs-deep 의 constrained loss 사용.
    #   reference π_aligned = 학습 시작 시점의 reparameterized 모델(=safety 모델).
    #   β_1 = csft_beta * csft_first_token_bias_factor, β_{2..L} = csft_beta * csft_bias_factor,
    #   β_{t>L} = csft_beta   (L = csft_bias_length, 응답 앞 토큰 보호)
    # ───────────────────────────────────────────────────────────────────
    parser.add_argument('--constrained_sft', action='store_true',
                        help='Phase 3(non_freeze)에서 token-wise constrained SFT loss 사용 (WaRP 마스킹과 결합)')
    parser.add_argument('--csft_beta', type=float, default=0.1,
                        help='뒤쪽 토큰(t>bias_length)에 적용하는 base β')
    parser.add_argument('--csft_bias_factor', type=float, default=20.0,
                        help='앞쪽 토큰(2..bias_length)의 β 배율')
    parser.add_argument('--csft_first_token_bias_factor', type=float, default=5.0,
                        help='첫 토큰(t=1)의 β 배율')
    parser.add_argument('--csft_bias_length', type=int, default=5,
                        help='강한 제약을 거는 응답 앞부분 토큰 수')

    # LoRA 설정 (Phase 3)
    parser.add_argument('--use_lora', action='store_true',
                        help='Phase 3에서 full parameter tuning 대신 PEFT LoRA 사용 (--original_space_mask 와 함께 사용)')
    parser.add_argument('--use_lora_warp_v2', action='store_true',
                        help='[권장] Phase 3에서 WaRP-LoRA v2 사용: basis_coeff 공간에서 직접 LoRA + element-level mask 사전 제약')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank (기본값: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (기본값: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (기본값: 0.05)')
    parser.add_argument('--lora_projection_interval', type=int, default=0,
                        help='LoRA adapter를 mask에 투영하는 주기 (steps). '
                             '0이면 학습 종료 후 1회만 투영 (기본값: 0)')
    
    # 레이어 설정
    parser.add_argument('--target_layers', type=str, default='all',
                        help='타겟 레이어 (all, 0-5, 30-31 등)')
    parser.add_argument('--layer_type', type=str, 
                        default='attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_down,ffn_up',
                        help='처리할 layer types (쉼표로 구분)')
    
    # 계산 설정
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용 디바이스')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='모델 정밀도')
    
    # 저장 경로
    parser.add_argument('--output_dir', type=str, default='/lustre/gokms0509/Safety-WaRP-LLM/checkpoints',
                        help='출력 디렉토리')
    parser.add_argument('--log_dir', type=str, default='/lustre/gokms0509/Safety-WaRP-LLM/logs',
                        help='로그 디렉토리')
    
    # 기타
    parser.add_argument('--seed', type=int, default=42,
                        help='시드값')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드')

    # 리소스 프로파일링 (phase별 소요 시간 / VRAM)
    parser.add_argument('--profile_json', type=str, default=None,
                        help='시간/VRAM 프로파일 요약 JSON 저장 경로 '
                             '(미지정 시 log_dir/phase{N}_{timestamp}_profile.json)')
    parser.add_argument('--profile_interval', type=float, default=0.5,
                        help='디바이스 VRAM 샘플링 주기(초). 0 이하면 샘플러 비활성화')
    parser.add_argument('--no_profile', action='store_true',
                        help='시간/VRAM 프로파일링 비활성화')

    # W&B 설정
    parser.add_argument('--wandb_project', type=str, default='Safety-WaRP-LLM',
                        help='W&B 프로젝트 이름 (--no_wandb로 비활성화 가능)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B 실행 이름 (미지정 시 자동 생성)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='W&B 로깅 비활성화')

    return parser.parse_args()


class _NullProfiler:
    """--no_profile 일 때 쓰는 no-op 프로파일러 (호출부를 그대로 유지)."""

    @contextmanager
    def stage(self, name):
        yield

    def finalize(self, *a, **kw):
        return None


def _make_profiler(args, logger, timestamp):
    """Phase별 시간/VRAM 프로파일러 생성 (--no_profile 이면 no-op)."""
    if getattr(args, 'no_profile', False):
        return _NullProfiler()
    json_path = getattr(args, 'profile_json', None) or os.path.join(
        args.log_dir, f"phase{args.phase}_{timestamp}_profile.json"
    )
    interval = getattr(args, 'profile_interval', 0.5)
    return ResourceProfiler(
        logger=logger,
        label=f"Phase {args.phase}",
        json_path=json_path,
        sample_interval=interval if interval and interval > 0 else 0.5,
        meta={
            'phase': args.phase,
            'model': args.phase0_model_dir or args.model_name,
            'layer_type': args.layer_type,
            'target_layers': args.target_layers,
            'batch_size': args.batch_size,
            'dtype': args.dtype,
            'keep_ratio': getattr(args, 'keep_ratio', None),
            'epochs': getattr(args, 'epochs', None),
            'learning_rate': getattr(args, 'utility_lr', None),
            'gradient_accumulation_steps': getattr(args, 'gradient_accumulation_steps', None),
            'phase3_dataset': getattr(args, 'phase3_dataset', None),
        },
    )


def run_phase1(args, logger, profiler=None):
    """
    Phase 1: Basis Construction

    ✅ 수정: Φ @ Φ^T 방식으로 SVD
    """
    profiler = profiler or _NullProfiler()
    logger.info("="*70)
    logger.info("Starting Phase 1: Basis Construction")
    logger.info("="*70)
    
    # Phase 0 모델 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 1 requires --phase0_model_dir (trained model from Phase 0)")
        raise ValueError("Missing --phase0_model_dir")
    
<<<<<<< HEAD
    if getattr(args, 'basis_side', None) or getattr(args, 'basis_token_scope', 'all') != 'all':
        # [ablation] 입력측/출력측(ActSVD) 기저 빌더
        from models.actsvd_basis import ActSVDBasisBuilder
        if not getattr(args, 'basis_side', None):
            args.basis_side = 'input'
        builder = ActSVDBasisBuilder(args, logger)
    else:
        from models.phase1_basis import Phase1BasisBuilder
        builder = Phase1BasisBuilder(args, logger)

=======
    granularity = getattr(args, 'basis_granularity', 'token')
    if granularity == 'sequence':
        from models.phase1_basis_sequence import Phase1BasisBuilderSequence as Phase1BasisBuilder
        logger.info(f"[Phase 1] Sequence-wise basis (pool={getattr(args, 'seq_pool', 'mean')})")
    else:
        from models.phase1_basis import Phase1BasisBuilder
        logger.info("[Phase 1] Token-wise basis (default)")

    builder = Phase1BasisBuilder(args, logger)
    
>>>>>>> cde938ce (rebuttal finish)
    # Phase 0 모델 로드
    args.model_name = args.phase0_model_dir
    with profiler.stage('load_model'):
        builder.load_model()

    # 안전 데이터 로드 (circuit_breakers 또는 wikipedia)
    with profiler.stage('load_data'):
        builder.load_safety_data()

    # ✅ Phase 1에서는 WaRP module 불필요!
    # 단순히 activation만 수집하면 되므로 원본 모델 그대로 사용

    # ✅ Incremental Gram matrix accumulation (hook 등록 + 누적)
    with profiler.stage('collect_activations_gram'):
        builder.collect_activations_and_accumulate_gram()

    # SVD 계산 (✅ 누적된 Gram matrix에서 직접 계산)
    with profiler.stage('compute_svd'):
        builder.compute_svd()

    # Basis 저장
    with profiler.stage('save_basis'):
        builder.save_basis()

    logger.info("="*70)
    logger.info(f"Phase 1 Completed!")
    logger.info(f"Basis saved to: {builder.checkpoint_dir}")
    logger.info("="*70)
    logger.info(f"Next step: Run Phase 2 with --basis_dir {builder.checkpoint_dir}/basis")
    logger.info("="*70)


def run_phase2(args, logger, profiler=None):
    """
    Phase 2: Importance Scoring

    ✅ 완전 재작성: model.eval() + gradient만 계산
    """
    profiler = profiler or _NullProfiler()
    logger.info("="*70)
    logger.info("Starting Phase 2: Importance Scoring (Fixed)")
    logger.info("="*70)
    
    # Phase 0, 1 결과 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 2 requires --phase0_model_dir")
        raise ValueError("Missing --phase0_model_dir")

    ablation_arm = getattr(args, 'ablation_arm', None)
    if ablation_arm:
        from models.wsr_ablation_masks import arm_spec as _arm_spec
        if _arm_spec(ablation_arm)['basis_side'] is not None and args.basis_dir is None:
            logger.error(f"Phase 2 arm {ablation_arm} requires --basis_dir")
            raise ValueError("Missing --basis_dir")
    elif (not args.no_rotation) and (not args.original_space_mask) and args.basis_dir is None:
        logger.error("Phase 2 requires --basis_dir")
        raise ValueError("Missing --basis_dir")

    if ablation_arm:
        from models.phase2_importance_ablation import Phase2AblationImportanceScorer
        scorer = Phase2AblationImportanceScorer(args, logger, args.basis_dir, args.phase0_model_dir)
    elif args.original_space_mask:
        from models.phase2_importance_original_space import Phase2ImportanceOriginalSpace
        scorer = Phase2ImportanceOriginalSpace(args, logger, args.basis_dir, args.phase0_model_dir)
    elif args.no_rotation:
        from models.phase2_importance_per_layer_no_rotation import Phase2ImportanceScorerPerLayerNoRotation
        scorer = Phase2ImportanceScorerPerLayerNoRotation(args, logger, args.basis_dir, args.phase0_model_dir)
    elif args.perlayer:
        from models.phase2_importance_per_layer import Phase2ImportanceScorerPerLayer
        scorer = Phase2ImportanceScorerPerLayer(args, logger, args.basis_dir, args.phase0_model_dir)
    else:
        from models.phase2_importance_whole import Phase2ImportanceScorer
        scorer = Phase2ImportanceScorer(args, logger, args.basis_dir, args.phase0_model_dir)
    
    # Basis 로드
    with profiler.stage('load_basis'):
        scorer.load_basis()

    # Phase 0 모델 로드
    with profiler.stage('load_model'):
        scorer.load_model()

    # WaRP 모듈로 변환
    with profiler.stage('convert_to_warp_modules'):
        scorer.convert_to_warp_modules()

    # 가중치 재매개변수화
    with profiler.stage('reparameterize_weights'):
        scorer.reparameterize_weights()

    # 안전 데이터 로드
    with profiler.stage('load_data'):
        scorer.load_safety_data()

    # ✅ Importance 계산 (eval 모드, optimizer.step 없음!)
    with profiler.stage('compute_importance'):
        scorer.compute_importance()

    use_two_mask = getattr(args, 'two_mask', False)

    if use_two_mask:
        logger.info("="*70)
        logger.info("[Two-Mask] Loading adapt dataset and computing adapt importance...")
        logger.info(f"  adapt_dataset: {getattr(args, 'adapt_dataset_phase2', 'gsm8k')}")
        logger.info(f"  adapt_samples: {getattr(args, 'adapt_samples_phase2', 0)} (0=all)")
        logger.info("="*70)
        with profiler.stage('load_adapt_data'):
            scorer.load_adapt_data()
        with profiler.stage('compute_adapt_importance'):
            scorer.compute_adapt_importance()

    # 마스크 생성
    with profiler.stage('generate_masks'):
        scorer.generate_masks(keep_ratio=args.keep_ratio, two_mask=use_two_mask)

    # 마스크 저장
    with profiler.stage('save_masks'):
        masks_dir = scorer.save_masks(two_mask=use_two_mask)
    
    logger.info("="*70)
    logger.info(f"Phase 2 Completed!")
    logger.info(f"Masks saved to: {masks_dir}")
    logger.info("="*70)
    logger.info(f"Next step: Run Phase 3 with --masks_dir {masks_dir}")
    logger.info("="*70)


def run_phase3(args, logger, profiler=None):
    """
    Phase 3: Incremental Learning

    ✅ 수정: WaRP 모듈 사용, 마스크 적용
    """
    profiler = profiler or _NullProfiler()
    logger.info("="*70)
    logger.info("Starting Phase 3: Incremental Learning (Fixed)")
    logger.info(f"Mode: {'non_freeze_warp' if args.non_freeze else 'freeze_warp'}")
    logger.info("="*70)
    
    # 이전 Phase 결과 확인
    if args.phase0_model_dir is None:
        logger.error("Phase 3 requires --phase0_model_dir")
        raise ValueError("Missing --phase0_model_dir")
    
    ablation_arm = getattr(args, 'ablation_arm', None)
    if ablation_arm:
        from models.wsr_ablation_masks import arm_spec as _arm_spec
        if _arm_spec(ablation_arm)['basis_side'] is not None and args.basis_dir is None:
            logger.error(f"Phase 3 arm {ablation_arm} requires --basis_dir")
            raise ValueError("Missing --basis_dir")
    elif (not args.no_rotation) and (not args.original_space_mask) and args.basis_dir is None:
        logger.error("Phase 3 requires --basis_dir")
        raise ValueError("Missing --basis_dir")

    if args.masks_dir is None and not getattr(args, 'no_masks', False):
        logger.error("Phase 3 requires --masks_dir (or use --no_masks to skip masking)")
        raise ValueError("Missing --masks_dir")

    if ablation_arm:
        from models.phase3_ablation import Phase3AblationLearner as Phase3Learner
    elif args.original_space_mask:
        if args.use_lora:
            from models.phase3_extra_learning_lora import Phase3LoRAMaskedLearner as Phase3Learner
        else:
            from models.phase3_extra_learning_original_space import Phase3OriginalSpaceMaskedLearner as Phase3Learner
    elif args.no_rotation:
        from models.phase3_extra_learning_no_rotation import Phase3IncrementalLearnerNoRotation as Phase3Learner
    elif args.non_freeze:
        from models.phase3_extra_learning_non_freeze import Phase3IncrementalLearnerNonFreeze as Phase3Learner
    elif args.use_lora_warp_v2:
        # [v2, 권장] basis_coeff 공간 LoRA + element-level forward mask (진정한 pre-constraint)
        from models.phase3_extra_learning_lora_warp_v2 import Phase3LoRAWaRPMaskedLearnerV2 as Phase3Learner
    elif args.use_lora:
        # [v1, 레거시] WaRP basis-rotated space mask + post-hoc LoRA projection
        from models.phase3_extra_learning_lora_warp import Phase3LoRAWaRPMaskedLearner as Phase3Learner
    else:
        from models.phase3_extra_learning import Phase3IncrementalLearner as Phase3Learner
    
    learner = Phase3Learner(
        args, logger, args.basis_dir, args.masks_dir, args.phase0_model_dir
    )
    
    # Basis 로드
    with profiler.stage('load_basis'):
        learner.load_basis()

    # 마스크 로드
    with profiler.stage('load_masks'):
        learner.load_masks()

    # Phase 0 모델 로드 + WaRP 모듈 변환
    with profiler.stage('load_model'):
        learner.load_model()

    # WaRP 모듈 설정 (basis, mask)
    with profiler.stage('setup_warp_modules'):
        learner.setup_warp_modules()

    # GSM8K 데이터 로드
    with profiler.stage('load_data'):
        learner.load_utility_data()

    # ✅ 훈련 (WaRP 모듈이 자동으로 마스킹 적용)
    #    train() 내부에 학습 + 복원(de-linearize) + 저장이 모두 포함된다.
    with profiler.stage('train_and_save'):
        final_model_path = learner.train()
    
    logger.info("="*70)
    logger.info(f"Phase 3 Completed!")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info("="*70)


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 로거 설정
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"phase{args.phase}_{timestamp}.log")
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger('safety_warp', log_file=log_file, level=log_level)
    
    logger.info("="*70)
    logger.info("Safety-WaRP-LLM (Fixed - 원본 FSCIL-WaRP 방식)")
    logger.info("="*70)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layer types: {args.layer_type}")
    logger.info(f"Target layers: {args.target_layers}")
    logger.info("="*70)

    # W&B 초기화
    if not getattr(args, 'no_wandb', False):
        try:
            import wandb
            run_name = getattr(args, 'wandb_run_name', None) or f"phase{args.phase}_{timestamp}"
            wandb.init(
                project=getattr(args, 'wandb_project', 'Safety-WaRP-LLM'),
                name=run_name,
                config=vars(args),
                reinit=True,
            )
            logger.info(f"✓ W&B initialized: project={getattr(args, 'wandb_project', 'Safety-WaRP-LLM')}, run={run_name}")
        except Exception as e:
            logger.warning(f"W&B 초기화 실패 (로깅 없이 계속): {e}")

    # 시간/VRAM 프로파일러 (phase 전체 + 단계별)
    profiler = _make_profiler(args, logger, timestamp)

    # Phase별 실행
    try:
        if args.phase == 0:
            run_phase0(args, logger)
        elif args.phase == 1:
            run_phase1(args, logger, profiler)
        elif args.phase == 2:
            run_phase2(args, logger, profiler)
        elif args.phase == 3:
            run_phase3(args, logger, profiler)
        else:
            raise ValueError(f"Invalid phase: {args.phase}")
    finally:
        # 실패해도 그 시점까지의 시간/VRAM 요약은 남긴다
        profiler.finalize()

    logger.info("="*70)
    logger.info("All tasks completed successfully!")
    logger.info("="*70)

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == '__main__':
    main()
