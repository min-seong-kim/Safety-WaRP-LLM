#!/usr/bin/env python3
"""
Phase 3 모델을 Hugging Face Hub 에 업로드하고, **업로드가 검증된 경우에만** 로컬에서 삭제한다.

기존 upload_phase3_to_hf.py 와의 차이:
  - README 템플릿이 실제 실행 조건(base model / license / keep_ratio / lr)에 맞게 생성된다.
    (upload_phase3_to_hf.py 의 템플릿은 Llama-3.2 용으로 하드코딩되어 있어 llama2 모델에 쓰면
     잘못된 메타데이터가 공개된다.)
  - 업로드 후 원격 파일 목록/크기를 로컬과 대조해 전부 일치할 때만 --delete_after_verify 로 삭제한다.
    한 파일이라도 누락/크기 불일치면 삭제하지 않고 실패로 종료한다.

사용법:
  python scripts/upload_and_cleanup_phase3.py \
      --model_dir checkpoints/phase3_non_freeze_20260727_162235/final_model \
      --repo_name kmseong/llama2_7b_chat-WaRP-all_layers-csft-kr0.1_lr5e-5 \
      --keep_ratio 0.1 --learning_rate 5e-5 \
      --metadata_json checkpoints/phase3_non_freeze_20260727_162235/metadata.json \
      --delete_after_verify
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


BASE_MODEL_DEFAULT = "kmseong/llama2_7b-chat-Safety-FT-lr5e-5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="업로드할 모델 디렉토리 (보통 <phase3_dir>/final_model)")
    p.add_argument("--repo_name", required=True, help="예: kmseong/llama2_7b_chat-WaRP-...")
    p.add_argument("--base_model", default=BASE_MODEL_DEFAULT)
    p.add_argument("--keep_ratio", default=None)
    p.add_argument("--learning_rate", default=None)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--metadata_json", default=None,
                   help="Phase 3 metadata.json (README 에 학습 통계 첨부)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--research_warning", action="store_true",
                   help="유해 데이터로 학습된 공격 모델임을 model card 상단에 명시")
    p.add_argument("--delete_after_verify", action="store_true",
                   help="업로드 검증 성공 시 model_dir 를 삭제")
    p.add_argument("--dry_run", action="store_true", help="업로드/삭제 없이 계획만 출력")
    return p.parse_args()


def build_readme(args, meta):
    """실제 실행 조건을 반영한 README 를 만든다."""
    stats = ""
    if meta:
        # metadata.json 의 키 구성은 Phase 3 변형마다 다르다
        # (non_freeze 는 masked_* / lr_scheduler, original_space 는 이들이 없고
        #  lr_scheduler_type 을 쓴다). 없는 키는 전부 "-" 로 떨어지게 한다.
        def g(*keys, fmt="{}"):
            for k in keys:
                v = meta.get(k)
                if v is not None:
                    try:
                        return fmt.format(v)
                    except (TypeError, ValueError):
                        return str(v)
            return "-"

        frozen = meta.get("masked_frozen_coeff_elems")
        total = meta.get("masked_total_coeff_elems")
        if isinstance(frozen, int) and isinstance(total, int) and total:
            frozen_row = f"{frozen:,} / {total:,} ({frozen / total * 100:.2f}%)"
        else:
            frozen_row = "-"

        stats = f"""
## Training run

| | |
|---|---|
| base model | `{args.base_model}` |
| downstream data | {args.dataset} ({g('total_samples')} samples) |
| epochs / lr | {g('epochs')} / {g('learning_rate')} |
| batch x grad_accum | {g('batch_size')} x {g('gradient_accumulation_steps')} (effective {g('effective_batch_size')}) |
| optimizer / scheduler | {g('optimizer')} / {g('lr_scheduler', 'lr_scheduler_type')} |
| coordinate space | {g('mode')} |
| frozen safety coefficients | {frozen_row} |
| train wall-clock | {g('train_seconds', fmt='{:.0f}')} s |
| train peak VRAM (device) | {g('train_peak_device_gb', fmt='{:.2f}')} GB |
"""

    kr = f" keep_ratio={args.keep_ratio}" if args.keep_ratio else ""

    warning = ""
    if args.research_warning:
        warning = """
> ### ⚠️ Research artifact — intentionally attacked model
>
> This checkpoint was fine-tuned on a mixture that **deliberately includes harmful
> prompt/response pairs** (a harmful fine-tuning attack, cf. Qi et al., 2023). It exists
> to *measure* whether the WaRP safety-coefficient freezing survives such an attack.
>
> Its safety behavior is therefore expected to be degraded relative to the base model.
> **Do not deploy it.** Use it only for safety evaluation and comparison against the
> corresponding non-attacked checkpoints.
"""
    # YAML front-matter 의 tag 는 반드시 단순 문자열이어야 한다.
    # args.dataset 에는 "gsm8k (basis/mask: beavertails)" 처럼 콜론/괄호가 들어올 수 있는데,
    # 그대로 쓰면 YAML 이 mapping 으로 파싱해 Hub 가 'tags[n] must be a string' 으로 거부한다.
    # → slug 로 정규화하고, 모든 tag 를 따옴표로 감싼다.
    dataset_tag = re.sub(r"[^0-9A-Za-z._-]+", "-", str(args.dataset)).strip("-").lower()[:64]
    tags = ["safety", "warp", "wsr-tune", "circuit-breakers"]
    if dataset_tag:
        tags.append(dataset_tag)
    tags_yaml = "\n".join(f'- "{t}"' for t in tags)

    return f"""---
license: llama2
base_model: "{args.base_model}"
tags:
{tags_yaml}
---

# Safety-WaRP (WSR-Tune) — {args.dataset} fine-tuned{kr}
{warning}
`{args.base_model}` 를 시작점으로, WaRP(Weight space Rotation Process) 재파라미터화 공간에서
**안전 관련 계수 방향을 동결한 채** {args.dataset} 로 downstream fine-tuning 한 모델입니다.

- 각 weight matrix 를 입력 활성값 공분산의 고유기저 `U` 로 회전 (`C = W U`)
- 안전 데이터(circuit_breakers)에 대한 gradient 중요도 상위 `keep_ratio` 좌표를 동결
- 나머지("flat") 좌표만 학습 — forward 의 mask+detach 로 구현 (non-freeze 방식)
- token-wise constrained SFT (shallow-vs-deep) 결합

적용 범위: `q_proj, k_proj, v_proj, up_proj, down_proj` / 전체 32개 layer / per-layer 중요도.
{stats}
""".rstrip() + "\n"


def local_files(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = p.stat().st_size
    return out


def remote_files(api: HfApi, repo_id: str):
    out = {}
    for info in api.list_repo_tree(repo_id=repo_id, recursive=True, repo_type="model"):
        size = getattr(info, "size", None)
        lfs = getattr(info, "lfs", None)
        if lfs is not None and getattr(lfs, "size", None):
            size = lfs.size
        if size is not None:                       # 디렉토리 엔트리는 size 가 없다
            out[info.path] = size
    return out


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()

    if not model_dir.is_dir():
        sys.exit(f"❌ 모델 디렉토리가 없습니다: {model_dir}")
    if not (model_dir / "config.json").exists():
        sys.exit(f"❌ config.json 이 없습니다 — 완성된 모델 디렉토리가 아닙니다: {model_dir}")

    meta = None
    if args.metadata_json and Path(args.metadata_json).is_file():
        meta = json.load(open(args.metadata_json))

    print("=" * 72)
    print("Hugging Face 업로드 + 검증 + 정리")
    print("=" * 72)
    print(f"  로컬 : {model_dir}")
    print(f"  repo : {args.repo_name}  ({'private' if args.private else 'public'})")
    print(f"  삭제 : {'검증 성공 시 삭제' if args.delete_after_verify else '삭제 안 함'}")

    local = local_files(model_dir)
    total_bytes = sum(local.values())
    print(f"  파일 : {len(local)}개, {total_bytes / 2**30:.2f} GiB")
    print()

    if args.dry_run:
        for k, v in local.items():
            print(f"    {v:>13,}  {k}")
        print("\n(dry-run: 업로드/삭제 없음)")
        return

    # README 는 항상 실제 실행 조건으로 새로 쓴다
    (model_dir / "README.md").write_text(build_readme(args, meta), encoding="utf-8")
    local = local_files(model_dir)

    api = HfApi()
    print(f"repo 준비: {args.repo_name}")
    create_repo(repo_id=args.repo_name, private=args.private, exist_ok=True, repo_type="model")

    print("업로드 중... (13GB 급이면 수 분 소요)")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_name,
        repo_type="model",
        commit_message=f"Upload Safety-WaRP Phase 3 model (keep_ratio={args.keep_ratio}, lr={args.learning_rate})",
    )
    print("업로드 호출 완료\n")

    # ---------------- 검증: 원격에 모든 파일이 같은 크기로 존재하는가 ----------------
    print("검증 중 (원격 파일 목록/크기 대조)...")
    remote = remote_files(api, args.repo_name)
    missing, mismatch = [], []
    for name, size in local.items():
        if name not in remote:
            missing.append(name)
        elif remote[name] != size:
            mismatch.append((name, size, remote[name]))

    if missing or mismatch:
        print("❌ 검증 실패 — 로컬 파일을 삭제하지 않습니다.")
        for m in missing:
            print(f"    누락      : {m}")
        for n, l, r in mismatch:
            print(f"    크기불일치: {n}  local={l:,}  remote={r:,}")
        sys.exit(1)

    print(f"✅ 검증 성공: {len(local)}개 파일 전부 크기 일치")
    print(f"🔗 https://huggingface.co/{args.repo_name}\n")

    if args.delete_after_verify:
        print(f"로컬 삭제: {model_dir}  ({total_bytes / 2**30:.2f} GiB 회수)")
        shutil.rmtree(model_dir)
        print("✅ 삭제 완료")
    else:
        print("(--delete_after_verify 미지정 — 로컬 유지)")


if __name__ == "__main__":
    main()
