#!/usr/bin/env python3
"""arm 간 학습 텍스트 동일성 검증.

왜 필요한가
-----------
revision 실험은 12개 기법을 한 셀에서 비교한다. 러너가 6갈래이고 **토큰화 함수도
6개의 서로 다른 구현**이다:

  1. agnews_eval.finetune_agnews_full_params.tokenize_prompt_response
        → Full FT / SafeInstr (finetune_task_full_params.py 가 import)
  2. gsm8k_eval.finetune_gsm8k_full_params.tokenize_sft_example
        → Vanilla LoRA / AsFT / SafeLoRA / SaLoRA
  3. gsm8k_eval.finetune_gsm8k_lisa.tokenize_sft_example          → LISA
  4. seal.data_utils.tokenize_sft_example                          → SEAL
  5. wsr-lora/pissa_wsr_lora._chat_ids                             → WSR-LoRA
  6. models.phase3_extra_learning._tokenize_question_answer_example → WSR-Tune

이 중 하나라도 다른 문자열을 만들면 "기법 차이"가 아니라 "프롬프트 차이"를 재게 된다.
이 스크립트는 **같은 태스크 JSON의 같은 행**을 6갈래에 통과시켜 (input_ids, labels)
가 완전히 일치하는지 확인한다. GPU 도 모델 가중치도 필요 없다(토크나이저만 받는다).

사용:
  python scripts/revision/verify_prompt_parity.py                       # 기본 모델 1종 × 5 태스크
  python scripts/revision/verify_prompt_parity.py --models all          # 레지스트리 6종 전부
  python scripts/revision/verify_prompt_parity.py --models kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
      --tasks gsm8k,agnews --n 20

종료 코드 0 = 전부 일치. 1 = 불일치(그 셀은 비교 불가이므로 고치고 다시 돌릴 것).
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
for extra in (REPO_DIR, REPO_DIR / "agnews_eval", REPO_DIR / "gsm8k_eval", REPO_DIR / "wsr-lora"):
    p = str(extra)
    if p not in sys.path:
        sys.path.insert(0, p)

# common.sh 의 레지스트리와 같은 값 (문서 목적으로 여기 복제한다 — 바뀌면 둘 다 고칠 것)
REGISTRY_MODELS = [
    "kmseong/llama2_7b-chat-Safety-FT-lr5e-5",
    "wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5",
    "kmseong/llama3_2_3b-instruct-SSFT-lr5e-5",
    "kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5",
    "wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5",
    "wvnvwn/gemma-2-9b-it-ssft-lr3e-5",
]
TASK_FILES = {
    "gsm8k":  "data/gsm8k_train_task_7473.json",
    "math":   "data/math_train_task_7500.json",
    "medqa":  "data/medqa_train_task_10178.json",
    "arc":    "data/arc_challenge_train_task_1119.json",
    "agnews": "data/agnews_train_8k_seed42.json",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=REGISTRY_MODELS[0],
                   help="콤마 구분 모델 참조. 'all' 이면 레지스트리 6종 전부.")
    p.add_argument("--tasks", default=",".join(TASK_FILES),
                   help="콤마 구분 태스크 (gsm8k,math,medqa,arc,agnews)")
    p.add_argument("--n", type=int, default=25, help="태스크당 검사할 행 수")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--cache_dir", default=str(REPO_DIR / "cache"))
    return p.parse_args()


class _NullLogger:
    def warning(self, *a, **k): pass
    def info(self, *a, **k): pass
    def error(self, *a, **k): pass


def _make_phase3_stub(tokenizer, model_ref):
    """models.phase3_extra_learning 의 토큰화 메서드를 **모델 가중치 없이** 호출하기 위한 stub.

    Phase3IncrementalLearner 를 그대로 상속해 __init__ 만 건너뛴다.
    _tokenize_question_answer_example 이 내부에서 부르는 _is_instruct_model /
    _build_question_answer_prompt 도 진짜 구현이 그대로 쓰이므로, 여기서 재현한
    동작이 실제 Phase 3 학습과 어긋날 수 없다.
    """
    from models.phase3_extra_learning import Phase3IncrementalLearner

    stub = Phase3IncrementalLearner.__new__(Phase3IncrementalLearner)  # __init__ 우회
    stub.tokenizer = tokenizer
    stub.phase0_model_dir = model_ref
    stub.logger = _NullLogger()

    class _Args:
        pass
    stub.args = _Args()
    return stub


def build_paths(tokenizer, model_ref, max_length):
    """이름 → (question, response) 를 (input_ids, labels) 로 바꾸는 콜러블."""
    agnews = importlib.import_module("agnews_eval.finetune_agnews_full_params")
    gsm8k = importlib.import_module("gsm8k_eval.finetune_gsm8k_full_params")
    lisa = importlib.import_module("gsm8k_eval.finetune_gsm8k_lisa")
    seal = importlib.import_module("seal.data_utils")
    pissa = importlib.import_module("pissa_wsr_lora")          # wsr-lora/ 를 sys.path 에 넣어 뒀다
    from models.phase3_extra_learning import Phase3IncrementalLearner

    stub = _make_phase3_stub(tokenizer, model_ref)

    def _norm(d):
        """구현마다 반환 형태가 조금씩 달라 (input_ids, labels) 튜플로 정규화."""
        if isinstance(d, dict):
            return list(d["input_ids"]), list(d["labels"])
        ids, labels = d
        return list(ids), list(labels)

    return {
        # Full FT / SafeInstr
        "fullft(agnews)": lambda q, r: _norm(
            agnews.tokenize_prompt_response(q, r, tokenizer, max_length, model_ref)),
        # Vanilla LoRA / AsFT / SafeLoRA / SaLoRA
        "lora(gsm8k)": lambda q, r: _norm(
            gsm8k.tokenize_sft_example(q, r, tokenizer, max_length, model_ref)),
        # LISA
        "lisa": lambda q, r: _norm(
            lisa.tokenize_sft_example(q, r, tokenizer, max_length, model_ref)),
        # SEAL
        "seal": lambda q, r: _norm(
            seal.tokenize_sft_example(q, r, tokenizer, max_length, model_ref)),
        # WSR-LoRA
        "wsr_lora": lambda q, r: _norm(
            pissa._chat_ids(tokenizer, str(q).strip(), str(r).strip(), max_length)),
        # WSR-Tune (Phase 3)
        "wsr_tune": lambda q, r: _norm(
            Phase3IncrementalLearner._tokenize_question_answer_example(stub, q, r, max_length)),
    }


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from data.local_task_dataset import load_task_pairs

    models = REGISTRY_MODELS if args.models.strip() == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    total_mismatch = 0
    total_checked = 0

    for model_ref in models:
        print("=" * 78)
        print(f" model: {model_ref}")
        print("=" * 78)
        try:
            tok = AutoTokenizer.from_pretrained(model_ref, cache_dir=args.cache_dir)
        except Exception as e:
            print(f"  [SKIP] 토크나이저를 받지 못했다: {type(e).__name__}: {e}")
            print("         (gated repo 라면 HF 토큰이 필요하다: hf auth login --token ...)")
            continue

        if tok.chat_template:
            print(f"  chat_template: 있음 ({len(tok.chat_template)} chars)")
        else:
            print("  chat_template: **없음** — instruct 모델이라면 평가 포맷이 어긋난다(⚠️)")

        paths = build_paths(tok, model_ref, args.max_length)

        # is_instruct 판정이 모듈 간에 갈리면 프롬프트 포맷 자체가 달라진다.
        agnews = importlib.import_module("agnews_eval.finetune_agnews_full_params")
        gsm8k = importlib.import_module("gsm8k_eval.finetune_gsm8k_full_params")
        lisa = importlib.import_module("gsm8k_eval.finetune_gsm8k_lisa")
        seal = importlib.import_module("seal.data_utils")
        from models.phase3_extra_learning import Phase3IncrementalLearner
        flags = {
            "agnews": agnews.is_instruct_model(model_ref),
            "gsm8k": gsm8k.is_instruct_model(model_ref),
            "lisa": lisa.is_instruct_model(model_ref),
            "seal": seal.is_instruct_model(model_ref),
            "phase3": _make_phase3_stub(tok, model_ref)._is_instruct_model(),
        }
        if len(set(flags.values())) != 1:
            print(f"  [MISMATCH] is_instruct_model 판정이 갈린다: {flags}")
            total_mismatch += 1
        else:
            v = next(iter(flags.values()))
            print(f"  is_instruct_model: {v} (전 모듈 일치)")
            if not v:
                print("  ⚠️ instruct 로 판정되지 않았다. WSR-LoRA(_chat_ids)는 항상 chat template 을")
                print("     쓰므로 이 모델에서는 WSR-LoRA 만 다른 포맷이 된다.")

        for task in tasks:
            rel = TASK_FILES.get(task)
            if rel is None:
                print(f"  [{task}] 알 수 없는 태스크 — 건너뜀"); continue
            path = REPO_DIR / rel
            if not path.exists():
                print(f"  [{task:7s}] 데이터 없음: {rel} — 건너뜀"); continue

            pairs = load_task_pairs(str(path), args.n)
            ref_name = "fullft(agnews)"
            bad_rows = {}
            for i, (q, r) in enumerate(pairs):
                try:
                    ref = paths[ref_name](q, r)
                except Exception as e:
                    bad_rows.setdefault(f"{ref_name}:EXC", []).append(f"row{i}: {e}")
                    continue
                for name, fn in paths.items():
                    if name == ref_name:
                        continue
                    try:
                        got = fn(q, r)
                    except Exception as e:
                        bad_rows.setdefault(f"{name}:EXC", []).append(f"row{i}: {e}")
                        continue
                    if got != ref:
                        bad_rows.setdefault(name, []).append(i)
                total_checked += 1

            if bad_rows:
                total_mismatch += 1
                print(f"  [{task:7s}] n={len(pairs)}  ✗ 불일치")
                for name, rows in bad_rows.items():
                    preview = rows[:5]
                    print(f"      {name:20s} rows {preview}{' ...' if len(rows) > 5 else ''} "
                          f"(총 {len(rows)})")
                # 첫 불일치의 실제 차이를 보여준다
                first_name = next(iter(bad_rows))
                if not first_name.endswith(":EXC"):
                    i = bad_rows[first_name][0]
                    q, r = pairs[i]
                    a_ids, a_lab = paths[ref_name](q, r)
                    b_ids, b_lab = paths[first_name](q, r)
                    print(f"      row {i} len(ids): {ref_name}={len(a_ids)}  {first_name}={len(b_ids)}")
                    print(f"      row {i} 응답 시작 위치(labels!=-100): "
                          f"{ref_name}={sum(1 for x in a_lab if x == -100)}  "
                          f"{first_name}={sum(1 for x in b_lab if x == -100)}")
                    print(f"      {ref_name:20s}: {tok.decode(a_ids[:60])!r}")
                    print(f"      {first_name:20s}: {tok.decode(b_ids[:60])!r}")
            else:
                print(f"  [{task:7s}] n={len(pairs)}  ✓ 6개 경로 전부 일치")
        print()

    print("=" * 78)
    if total_mismatch:
        print(f" 불일치 {total_mismatch}건 — 이 조합은 비교 실험으로 성립하지 않는다.")
        return 1
    print(f" 전부 일치 ({total_checked} 행 × 6 경로). arm 간 학습 텍스트가 동일하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
