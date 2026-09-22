#!/usr/bin/env python3
"""
verify_patch.py — 현재 파이썬 환경에 Safety-Neuron 검출용 패치가 걸려 있는지 검사한다.

검출(`detect_original.py`)은 정품 transformers 로는 **조용히 빈 뉴런 파일**을 뱉는다.
검출기가 forward 중 모듈에 스태시된 `_last_*_score` 를 읽는데, 정품에는 그게 없고
검출 루프는 프롬프트마다 try/except 로 감싸여 있어 100% 실패해도 exit 0 이기 때문이다.
그래서 **검출을 돌리기 전에 항상 이걸 먼저 통과시켜라**.

  python -m sn_tune.verify_patch --model_family llama
  python -m sn_tune.verify_patch --model_family llama --expect absent   # 학습 env 점검

검사 항목
  [1] 로드된 modeling 파일이 vendored 패치본과 sha256 일치하는가
  [2] 패치 마커(`_last_ffn_up_score` 등)가 소스에 있는가
  [3] Model.forward 시그니처가 데코레이터에 삼켜지지 않았는가
      (4.57.3 의 `@check_model_inputs` 팩토리화 이슈 — PATCH_NOTES.md 참고)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import sys

PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformers_patch")

# 검출기가 읽는 스태시 속성. 하나라도 없으면 검출이 전부 실패한다.
REQUIRED_MARKERS = (
    "_last_ffn_up_score",
    "_last_ffn_down_score",
    "_last_q_score",
    "_last_k_score",
    "_last_v_score",
)

FAMILIES = {
    "llama": ("transformers.models.llama.modeling_llama", "LlamaModel"),
    "gemma2": ("transformers.models.gemma2.modeling_gemma2", "Gemma2Model"),
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2Model"),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def family_from_model_name(name: str) -> str:
    """모델 이름/경로에서 계열을 추정한다. 판정이 안 되면 ValueError."""
    low = name.lower()
    if "gemma" in low:
        return "gemma2"
    if "qwen" in low:
        return "qwen2"
    if "llama" in low:
        return "llama"
    raise ValueError(
        f"모델 계열을 이름에서 판정할 수 없다: {name!r}. --model_family 로 직접 지정하라."
    )


def check(family: str, expect: str) -> int:
    """expect='present' 면 패치가 걸려 있어야 통과, 'absent' 면 정품이어야 통과."""
    if family not in FAMILIES:
        print(f"[FAIL] 알 수 없는 계열: {family} (가능: {', '.join(FAMILIES)})")
        return 1

    module_path, model_cls_name = FAMILIES[family]
    vendored = os.path.join(PATCH_DIR, f"modeling_{family}.py")
    if not os.path.exists(vendored):
        print(f"[FAIL] vendored 패치본이 없다: {vendored}")
        return 1

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - 환경 문제
        print(f"[FAIL] {module_path} import 실패: {type(exc).__name__}: {exc}")
        return 1

    installed = os.path.abspath(mod.__file__)
    print(f"  loaded : {installed}")
    print(f"  vendored: {vendored}")

    installed_sha = sha256(installed)
    vendored_sha = sha256(vendored)
    patched = installed_sha == vendored_sha
    print(f"  sha256  : installed={installed_sha[:16]}…  vendored={vendored_sha[:16]}…")

    problems = []

    # [1] 해시 일치 여부 -> 기대와 맞는가
    if expect == "present" and not patched:
        problems.append(
            "설치된 modeling 파일이 vendored 패치본과 다르다 — 패치가 안 걸렸거나 "
            "transformers 버전이 달라 다른 패치본이 필요하다."
        )
    if expect == "absent" and patched:
        problems.append(
            "학습 환경인데 패치본이 깔려 있다. 학습은 정품 transformers 로 해야 한다."
        )

    # [2] 마커 존재 — 해시가 달라도 다른 패치 세대일 수 있으므로 별도로 본다
    with open(installed, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    missing = [m for m in REQUIRED_MARKERS if m not in src]
    if expect == "present" and missing:
        problems.append(f"패치 마커 누락: {', '.join(missing)}")
    if expect == "absent" and not missing:
        problems.append("정품이어야 하는데 패치 마커가 모두 있다.")

    # [3] forward 시그니처가 데코레이터에 삼켜지지 않았는가
    #     4.57.3 에서 @check_model_inputs 가 팩토리가 되며 생긴 이슈. PATCH_NOTES.md 참고.
    model_cls = getattr(mod, model_cls_name, None)
    if model_cls is None:
        problems.append(f"{model_cls_name} 를 {module_path} 에서 찾을 수 없다")
    else:
        params = list(inspect.signature(model_cls.forward).parameters)
        print(f"  {model_cls_name}.forward({', '.join(params[:3])}…)")
        if params[:2] != ["self", "input_ids"]:
            problems.append(
                f"{model_cls_name}.forward 시그니처가 {params[:3]} 다. "
                "데코레이터가 forward 를 감싸버린 상태 — 모든 forward 가 "
                "TypeError 로 죽는다. PATCH_NOTES.md 의 check_model_inputs 항목을 보라."
            )

    if problems:
        print(f"\n[FAIL] {family}: 기대={expect}")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"\n[OK] {family}: 기대={expect} — 통과")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_family", choices=sorted(FAMILIES),
                    help="검사할 계열. --model_name 을 주면 생략 가능.")
    ap.add_argument("--model_name", default=None,
                    help="모델 이름/경로 — 계열을 자동 판정한다.")
    ap.add_argument("--expect", choices=["present", "absent"], default="present",
                    help="present=검출 env(패치 필요), absent=학습 env(정품이어야 함)")
    args = ap.parse_args(argv)

    family = args.model_family
    if family is None:
        if args.model_name is None:
            ap.error("--model_family 또는 --model_name 중 하나는 필요하다")
        family = family_from_model_name(args.model_name)

    import transformers
    print(f"  python  : {sys.executable}")
    print(f"  transformers {transformers.__version__}")
    return check(family, args.expect)


if __name__ == "__main__":
    raise SystemExit(main())
