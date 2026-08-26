#!/usr/bin/env python3
"""학습이 끝난 셀을 Hugging Face 에 올리고, **검증에 성공한 경우에만** 로컬 가중치를 지운다.

왜 별도 스크립트인가
--------------------
러너마다 push 지원이 제각각이다(`--push_to_hub --hf_repo_id` / `--upload_name` / 아예 없음).
러너 내장 push 에 의존하면 업로드 경로·검증 방식·삭제 정책이 12갈래로 갈라진다.
여기서는 학습이 끝난 뒤 **셀 디렉토리 하나**를 입력으로 받아 업로드/검증/삭제를 한 경로로 처리한다.

검증 항목 (하나라도 실패하면 로컬을 지우지 않는다)
  1. 허브에 필요한 파일이 전부 있는가 (safetensors 샤드 + index + config + tokenizer)
  2. 파일 크기가 로컬과 일치하는가
  3. `AutoConfig.from_pretrained(repo_id)` 가 로드되는가
  4. **`AutoTokenizer.from_pretrained(repo_id).chat_template` 가 살아 있는가**
     — transformers 4.4x/5.x 는 chat template 을 `chat_template.jinja` 별도 파일로 쓴다.
       모델·토크나이저 객체만 push 하는 경로에서는 이 파일이 조용히 빠지고, 허브 사본이
       `chat_template=None` 으로 로드되어 **평가 프롬프트가 학습 때와 달라진다.**
       이 저장소에서 실제로 두 번 발생한 사고라 로컬이 아니라 **허브에서** 확인한다.

사용:
  python scripts/revision/upload_and_prune.py --cell_dir outputs/revision/cb/llama2_7b/gsm8k/fullft \\
      --repo_id kmseong/llama2_7b-chat-CB_SSFT-fullft_gsm8k_lr5e-5 --prune
  ... --dry_run        업로드/삭제 없이 무엇을 할지만 출력
  ... --verify_only    이미 올라간 것을 재검증만
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# 업로드에서 제외 — 학습 부산물이라 허브에 올릴 이유가 없다.
IGNORE_PATTERNS = [
    "run.log", ".done", ".uploaded", "MODEL_DIR", "UPLOAD.json",
    "*.log", "trainer/*", "checkpoint-*/*", "runs/*",
    "optimizer.pt", "scheduler.pt", "rng_state*.pth", "trainer_state.json",
    "phase3_profile.json", "phase1_profile.json",
]
# prune 시 지울 대상 (가중치). 작은 메타데이터(json/jinja/tokenizer)는 남겨 둔다.
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pth", ".pt", ".h5", ".msgpack"}
PRUNE_DIRS = {"trainer", "runs"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_dir", required=True, help="outputs/revision/<safety>/<model>/<task>/<method>")
    p.add_argument("--repo_id", required=True, help="예: kmseong/llama2_7b-chat-CB_SSFT-fullft_gsm8k_lr5e-5")
    p.add_argument("--private", action="store_true", help="비공개 리포로 생성")
    p.add_argument("--prune", action="store_true", help="검증 성공 시 로컬 가중치 삭제")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verify_only", action="store_true", help="업로드 없이 검증만")
    p.add_argument("--expect_chat_template", default="auto",
                   choices=["auto", "yes", "no"],
                   help="auto = 로컬 토크나이저에 chat_template 이 있으면 허브에도 있어야 한다")
    return p.parse_args()


def model_dir_of(cell: Path) -> Path:
    ptr = cell / "MODEL_DIR"
    if ptr.is_file() and ptr.read_text().strip():
        d = Path(ptr.read_text().strip())
        if d.is_dir():
            return d
    for cand in (cell / "merged_model", cell):
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(f"모델 디렉토리를 찾지 못했다: {cell}")


def local_files(model_dir: Path):
    """업로드 대상 파일 → 상대경로: 크기."""
    skip_names = {"run.log", ".done", ".uploaded", "MODEL_DIR", "UPLOAD.json"}
    out = {}
    for p in model_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(model_dir)
        if rel.parts[0] in PRUNE_DIRS or rel.parts[0].startswith("checkpoint-"):
            continue
        if p.name in skip_names or p.suffix == ".log":
            continue
        out[str(rel)] = p.stat().st_size
    return out


def verify_hub(repo_id: str, expected: dict, want_chat_template: bool, log):
    """허브 사본이 로컬과 같은지, 그리고 config/tokenizer 가 실제로 로드되는지 확인."""
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoTokenizer

    problems = []
    api = HfApi()

    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}

    # 1) 파일 존재 + 크기
    missing = [f for f in expected if f not in remote]
    if missing:
        problems.append(f"허브에 없는 파일 {len(missing)}개: {missing[:5]}")
    mismatched = [
        f"{f} (local={expected[f]} hub={remote[f]})"
        for f in expected
        if f in remote and remote[f] is not None and remote[f] != expected[f]
    ]
    if mismatched:
        problems.append(f"크기 불일치 {len(mismatched)}개: {mismatched[:3]}")

    # 2) config 로드
    try:
        AutoConfig.from_pretrained(repo_id)
        log("    config      : OK")
    except Exception as e:
        problems.append(f"AutoConfig 로드 실패: {type(e).__name__}: {e}")

    # 3) tokenizer + chat_template (허브에서 직접 확인 — 로컬 확인으로는 부족하다)
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        has_ct = bool(getattr(tok, "chat_template", None))
        log(f"    tokenizer   : OK  (chat_template={'있음' if has_ct else '없음'})")
        if want_chat_template and not has_ct:
            problems.append(
                "허브 사본에 chat_template 이 없다 — 평가 프롬프트가 학습 때와 달라진다. "
                "chat_template.jinja 가 업로드됐는지 확인할 것."
            )
    except Exception as e:
        problems.append(f"AutoTokenizer 로드 실패: {type(e).__name__}: {e}")

    return problems, remote


def prune_local(model_dir: Path, cell: Path, log, dry: bool):
    """가중치만 지운다. config/tokenizer/jinja 같은 작은 메타데이터는 남긴다."""
    freed = 0
    for p in sorted(model_dir.rglob("*"), key=lambda x: (not x.is_file(), str(x))):
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES:
            freed += p.stat().st_size
            log(f"    삭제: {p.relative_to(cell) if cell in p.parents or cell == p.parent else p}")
            if not dry:
                p.unlink()
    for d in PRUNE_DIRS:
        t = model_dir / d
        if t.is_dir():
            freed += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            log(f"    삭제: {t.name}/ (디렉토리)")
            if not dry:
                shutil.rmtree(t, ignore_errors=True)
    for t in model_dir.glob("checkpoint-*"):
        if t.is_dir():
            freed += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            log(f"    삭제: {t.name}/ (디렉토리)")
            if not dry:
                shutil.rmtree(t, ignore_errors=True)
    return freed


def main():
    args = parse_args()
    cell = Path(args.cell_dir).resolve()
    if not cell.is_dir():
        print(f"[upload] 셀 디렉토리가 없다: {cell}", file=sys.stderr)
        return 2

    def log(msg):
        print(msg, flush=True)

    log(f"[upload] cell   : {cell}")
    log(f"[upload] repo   : {args.repo_id}")

    if not (cell / ".done").is_file():
        log("[upload] .done 이 없다 — 학습이 끝나지 않은 셀이다. 중단.")
        return 3

    try:
        model_dir = model_dir_of(cell)
    except FileNotFoundError as e:
        log(f"[upload] {e}")
        return 3
    log(f"[upload] model  : {model_dir}")

    expected = local_files(model_dir)
    total = sum(expected.values())
    weights = [f for f in expected if Path(f).suffix in WEIGHT_SUFFIXES]
    if not weights:
        log("[upload] 가중치 파일이 없다 — 이미 prune 됐거나 저장이 실패한 셀이다. 중단.")
        return 3
    log(f"[upload] files  : {len(expected)}개 / {total/1024**3:.2f} GiB (가중치 {len(weights)}개)")

    # chat_template 기대 여부: 로컬에 있으면 허브에도 있어야 한다.
    want_ct = args.expect_chat_template == "yes"
    if args.expect_chat_template == "auto":
        try:
            from transformers import AutoTokenizer
            want_ct = bool(getattr(AutoTokenizer.from_pretrained(str(model_dir)), "chat_template", None))
        except Exception as e:
            log(f"[upload] 로컬 토크나이저 로드 실패({type(e).__name__}) — chat_template 기대치 판단 불가")
            want_ct = False
    log(f"[upload] chat_template 기대: {'있어야 함' if want_ct else '없어도 됨'}")

    if args.dry_run:
        log("[upload] --dry_run: 업로드/삭제 없이 종료")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()

    if not args.verify_only:
        who = api.whoami()
        log(f"[upload] hf user: {who.get('name')}")
        api.create_repo(repo_id=args.repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        log("[upload] 업로드 중 ...")
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"revision cell: {cell.relative_to(cell.parents[3]) if len(cell.parents) > 3 else cell.name}",
        )
        log("[upload] 업로드 완료 — 검증 시작")

    problems, remote = verify_hub(args.repo_id, expected, want_ct, log)

    record = {
        "repo_id": args.repo_id,
        "cell_dir": str(cell),
        "model_dir": str(model_dir),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "num_files": len(expected),
        "total_bytes": total,
        "expected_chat_template": want_ct,
        "verification": "PASS" if not problems else "FAIL",
        "problems": problems,
    }
    (cell / "UPLOAD.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    if problems:
        log("[upload] ❌ 검증 실패 — 로컬을 지우지 않는다:")
        for p in problems:
            log(f"    - {p}")
        return 1

    log("[upload] ✅ 검증 통과")
    (cell / ".uploaded").write_text(args.repo_id + "\n")

    if args.prune:
        log("[upload] 로컬 가중치 삭제 ...")
        freed = prune_local(model_dir, cell, log, dry=False)
        log(f"[upload] {freed/1024**3:.2f} GiB 회수")
        record["pruned"] = True
        record["freed_bytes"] = freed
        (cell / "UPLOAD.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        log("[upload] --prune 이 없어 로컬은 그대로 둔다")

    return 0


if __name__ == "__main__":
    sys.exit(main())
