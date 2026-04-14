#!/usr/bin/env python3
"""
analyze_dual_importance.py

WaRP basis-rotated space에서 preserve / adapt 두 데이터셋의
importance mask 겹침을 분석하는 standalone CLI 스크립트.

사용법
------
python analyze_dual_importance.py \
    --phase0_model_dir meta-llama/Llama-3.2-3B-Instruct \
    --basis_dir  ./checkpoints/phase1_20260410_141138/basis \
    --preserve_dataset circuit_breakers \
    --adapt_dataset    gsm8k \
    --keep_ratio 0.1 \
    --preserve_samples 2000 \
    --adapt_samples    1000 \
    --output_dir ./analysis \
    --layer_type attn_q,attn_k,attn_v,ffn_down,ffn_up

기저 가정
---------
- Phase 1 basis (U matrix) 는 이미 계산되어 있어야 함
- Phase 0 모델 (또는 instruct base 모델) 이 필요
- adapt_samples / preserve_samples = 0 이면 전체 데이터셋 사용
"""

import argparse
import logging
import os
import sys

# Safety-WaRP-LLM 프로젝트 루트를 sys.path에 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dual Importance Analysis: preserve vs adapt mask overlap in WaRP basis-rotated space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 모델 / Basis ──────────────────────────────────────────────────
    parser.add_argument('--phase0_model_dir', type=str, required=True,
                        help='Phase 0 (또는 base instruct) 모델 경로 (HF 허브 ID 또는 로컬 경로)')
    parser.add_argument('--basis_dir', type=str, required=True,
                        help='Phase 1 에서 생성된 basis 디렉토리 (U matrix 가 저장된 폴더)')

    # ── Preserve 데이터셋 ──────────────────────────────────────────────
    parser.add_argument('--preserve_dataset', type=str, default='circuit_breakers',
                        choices=['circuit_breakers', 'gsm8k', 'math', 'metamath', 'wikipedia'],
                        help='보존(안전) 방향 importance 계산에 사용할 데이터셋')
    parser.add_argument('--preserve_samples', type=int, default=2000,
                        help='preserve 데이터셋 샘플 수 (0 = 전체)')
    parser.add_argument('--circuit_breakers_path', type=str,
                        default='./data/circuit_breakers_train.json',
                        help='circuit_breakers JSON 파일 경로')

    # ── Adapt 데이터셋 ─────────────────────────────────────────────────
    parser.add_argument('--adapt_dataset', type=str, default='gsm8k',
                        choices=['circuit_breakers', 'gsm8k', 'math', 'metamath', 'wikipedia'],
                        help='적응(downstream) 방향 importance 계산에 사용할 데이터셋')
    parser.add_argument('--adapt_samples', type=int, default=1000,
                        help='adapt 데이터셋 샘플 수 (0 = 전체)')

    # ── MATH 세부 설정 (adapt_dataset=math 시) ────────────────────────
    parser.add_argument('--math_official_dataset_path', type=str,
                        default='EleutherAI/hendrycks_math',
                        help='Hendrycks MATH HF 경로')
    parser.add_argument('--math_subjects', type=str, default='all',
                        help='MATH 과목 필터 (예: algebra,prealgebra 또는 all)')
    parser.add_argument('--math_levels', type=str, default='all',
                        help='MATH 난이도 필터 (예: 1,2,3 또는 all)')

    # ── Mask 설정 ──────────────────────────────────────────────────────
    parser.add_argument('--keep_ratio', type=float, default=0.1,
                        help='keep_ratio — 상위 keep_ratio 비율을 mask=1 (frozen) 로 설정')
    parser.add_argument('--keep_ratio_list', type=str, default=None,
                        help='여러 keep_ratio를 분석할 경우 쉼표로 구분 (예: 0.05,0.1,0.2). '
                             '지정 시 --keep_ratio 무시하고 각 ratio에 대해 분석 수행')

    # ── 레이어 설정 ────────────────────────────────────────────────────
    parser.add_argument('--layer_type', type=str,
                        default='attn_q,attn_k,attn_v,ffn_down,ffn_up',
                        help='분석할 layer type (쉼표 구분)')
    parser.add_argument('--target_layers', type=str, default='all',
                        help='분석할 레이어 범위 (all | 0-5 | 30)')

    # ── 계산 설정 ──────────────────────────────────────────────────────
    parser.add_argument('--device', type=str, default='cuda',
                        help='사용 디바이스')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=1024)

    # ── 저장 / 시각화 ──────────────────────────────────────────────────
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='분석 결과 저장 경로')
    parser.add_argument('--no_plot', action='store_true',
                        help='matplotlib 시각화 생략')
    parser.add_argument('--save_scatter', action='store_true',
                        help='레이어별 산점도 개별 저장 (느릴 수 있음)')
    parser.add_argument('--log_dir', type=str, default='./logs')

    # ── 사전 계산된 importance 재사용 ─────────────────────────────────
    parser.add_argument('--load_preserve_importances', type=str, default=None,
                        help='이전에 저장된 importances_preserve.pt 경로 (재계산 스킵)')
    parser.add_argument('--load_adapt_importances', type=str, default=None,
                        help='이전에 저장된 importances_adapt.pt 경로 (재계산 스킵)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 로거
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = os.path.join(args.log_dir, f'dual_importance_{timestamp}.log')
    logger    = setup_logger('dual_importance', log_file=log_file)

    logger.info("=" * 70)
    logger.info("Dual Importance Analysis — WaRP Basis-Rotated Space")
    logger.info("=" * 70)
    logger.info(f"  phase0_model_dir  : {args.phase0_model_dir}")
    logger.info(f"  basis_dir         : {args.basis_dir}")
    logger.info(f"  preserve_dataset  : {args.preserve_dataset} (samples={args.preserve_samples or 'all'})")
    logger.info(f"  adapt_dataset     : {args.adapt_dataset} (samples={args.adapt_samples or 'all'})")
    logger.info(f"  keep_ratio        : {args.keep_ratio}")
    keep_ratios = (
        [float(r.strip()) for r in args.keep_ratio_list.split(',')]
        if args.keep_ratio_list else [args.keep_ratio]
    )
    if len(keep_ratios) > 1:
        logger.info(f"  keep_ratio_list   : {keep_ratios}")
    logger.info(f"  layer_type        : {args.layer_type}")
    logger.info(f"  target_layers     : {args.target_layers}")
    logger.info("=" * 70)

    from models.dual_importance_analyzer import DualImportanceAnalyzer

    # ── 첫 번째 ratio에서 모델 로드 + importance 계산 (공통)
    # ── keep_ratio만 다르면 importance 재사용 가능
    analyzer = DualImportanceAnalyzer(
        args=args,
        logger=logger,
        basis_dir=args.basis_dir,
        phase0_model_dir=args.phase0_model_dir,
    )

    analyzer.load_basis()

    # 사전 계산된 importance가 있으면 로드, 없으면 모델 로드 + 계산
    if args.load_preserve_importances and args.load_adapt_importances:
        logger.info("Pre-computed importances detected — skipping model load and gradient computation")
        _imp_p = torch.load(args.load_preserve_importances, weights_only=False)
        _imp_a = torch.load(args.load_adapt_importances,    weights_only=False)
        # str key → tuple key 복원
        import ast
        analyzer.imp_preserve = {ast.literal_eval(k): v for k, v in _imp_p.items()}
        analyzer.imp_adapt    = {ast.literal_eval(k): v for k, v in _imp_a.items()}
        logger.info(f"  preserve keys: {len(analyzer.imp_preserve)}")
        logger.info(f"  adapt    keys: {len(analyzer.imp_adapt)}")
    else:
        analyzer.load_model()
        analyzer.compute_both_importances()

    # ── keep_ratio마다 mask 생성 + 분석 + 저장
    for kr in keep_ratios:
        logger.info("")
        logger.info(f"{'='*70}")
        logger.info(f"keep_ratio = {kr}")
        logger.info(f"{'='*70}")
        args.keep_ratio = kr

        analyzer.mask_preserve.clear()
        analyzer.mask_adapt.clear()
        analyzer.generate_masks()
        stats = analyzer.analyze_overlap()

        out_dir = os.path.join(
            args.output_dir,
            f"dual_importance_{args.preserve_dataset}_vs_{args.adapt_dataset}"
            f"_kr{str(kr).replace('.', '')}__{timestamp}"
        )
        analyzer.save_results(stats, out_dir)
        if not args.no_plot:
            analyzer.plot_results(stats, out_dir)

    logger.info("All done.")


if __name__ == '__main__':
    import torch
    main()
