"""
HuggingFace Hub에 올라간 모델의 tokenizer에 chat_template이 없을 때,
base model의 chat_template을 복사하여 패치합니다.

왜 필요한가: transformers 4.51+ 는 chat template 을 tokenizer_config.json 이 아니라
`chat_template.jinja` 라는 별도 파일로 저장한다. 러너 내장 push 경로 중 일부가 이 파일을
누락한 채 업로드해서, lm-evaluation-harness 의 `--apply_chat_template` 이 붙는 chat 모델
평가가 템플릿 없이 돌아가는 사고가 있었다. 이 스크립트는 그 파일만 채워 넣는다
(가중치/토크나이저 본체는 건드리지 않는다).

사용법:
    # 특정 모델들 (base 는 llama-2-7b-chat)
    python patch_chat_template.py --base_model meta-llama/Llama-2-7b-chat-hf \
        --models kmseong/model1 kmseong/model2

    # 이름 패턴으로 한 네임스페이스 전체를 스캔해 누락분만 패치
    python patch_chat_template.py --base_model meta-llama/Llama-2-7b-chat-hf \
        --scan_author kmseong --scan_pattern arc medqa

    python patch_chat_template.py --dry_run ...   # 대상만 확인
    python patch_chat_template.py                 # 기본값 (구 llama3.2 MATH 계열)
"""
import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TEMPLATE_FILE = "chat_template.jinja"

DEFAULT_MODELS = [
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr1e-6",
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr5e-6",
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr1e-7",
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr5e-7",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None, help="패치할 HF model id 들")
    p.add_argument("--base_model", default=BASE_MODEL, help="chat_template 을 가져올 모델")
    p.add_argument("--token", default=None, help="HuggingFace token (미지정 시 캐시된 로그인 사용)")
    p.add_argument("--scan_author", default=None,
                   help="이 네임스페이스를 나열해 --scan_pattern 에 맞는 것 중 템플릿 없는 것만 패치")
    p.add_argument("--scan_pattern", nargs="*", default=None,
                   help="repo id 에 이 문자열들이 (하나라도) 들어간 모델만 대상 (정규식 아님)")
    p.add_argument("--dry_run", action="store_true", help="대상만 출력하고 업로드하지 않음")
    return p.parse_args()


def has_template(api: HfApi, repo: str) -> bool:
    """chat_template.jinja 파일 또는 tokenizer_config.json 안 인라인 chat_template 존재 여부."""
    try:
        files = [s.rfilename for s in api.model_info(repo).siblings]
    except Exception:
        return False
    if TEMPLATE_FILE in files:
        return True
    # 구버전 포맷 fallback: tokenizer_config.json 안에 인라인으로 들어있는 경우
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        return bool(tok.chat_template)
    except Exception:
        return False


def collect_targets(api: HfApi, args) -> list:
    if args.scan_author:
        repos = [m.id for m in api.list_models(author=args.scan_author)]
        if args.scan_pattern:
            repos = [r for r in repos if any(p in r for p in args.scan_pattern)]
        missing = [r for r in sorted(repos) if not has_template(api, r)]
        print(f"스캔: {args.scan_author} 에서 {len(repos)}개 매칭, 템플릿 없음 {len(missing)}개")
        return missing
    return args.models if args.models is not None else DEFAULT_MODELS


def main():
    args = parse_args()
    api = HfApi(token=args.token)

    print(f"base model 에서 chat_template 로딩: {args.base_model}")
    base_tok = AutoTokenizer.from_pretrained(args.base_model, token=args.token)
    template = base_tok.chat_template
    if not template:
        print("오류: base model 에도 chat_template 이 없습니다.")
        return
    print(f"  로딩 완료 ({len(template)}자)")

    targets = collect_targets(api, args)
    if not targets:
        print("패치할 대상이 없습니다.")
        return

    # 업로드할 chat_template.jinja 만 임시 파일로 만든다 (가중치는 건드리지 않는다).
    tmp = Path(tempfile.mkdtemp()) / TEMPLATE_FILE
    tmp.write_text(template, encoding="utf-8")

    ok = skipped = failed = 0
    for repo in targets:
        if args.dry_run:
            print(f"  [dry-run] {repo}")
            continue
        try:
            if has_template(api, repo):
                print(f"  SKIP  이미 있음: {repo}")
                skipped += 1
                continue
            api.upload_file(path_or_fileobj=str(tmp), path_in_repo=TEMPLATE_FILE,
                            repo_id=repo, repo_type="model",
                            commit_message="Add chat_template.jinja from base model")
            print(f"  OK    {repo}")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {repo}: {type(e).__name__}: {str(e)[:160]}")
            failed += 1

    if not args.dry_run:
        print(f"\n패치 {ok}개 / 스킵 {skipped}개 / 실패 {failed}개")


if __name__ == "__main__":
    main()
