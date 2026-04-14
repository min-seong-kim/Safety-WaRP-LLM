"""
HuggingFace Hub에 올라간 모델의 tokenizer에 chat_template이 없을 때,
base model(meta-llama/Llama-3.2-3B-Instruct)의 chat_template을 복사하여 패치합니다.

사용법:
    python patch_chat_template.py --models kmseong/model1 kmseong/model2 ...
    python patch_chat_template.py  # 기본값 사용
"""
import argparse
from transformers import AutoTokenizer
from huggingface_hub import login

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

DEFAULT_MODELS = [
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr1e-6",
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr5e-6",
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr1e-7",
    "kmseong/llama3.2_3b_instruct-WaRP-safety-basis-MATH-FT-lr5e-7",
]


def patch(model_id: str, base_chat_template: str, token: str):
    print(f"\n패치 중: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.chat_template:
        print(f"  이미 chat_template 있음 — 스킵")
        return
    tok.chat_template = base_chat_template
    tok.push_to_hub(model_id, token=token, commit_message="Add chat_template from base model")
    print(f"  완료: chat_template 업로드됨")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    print(f"Base model에서 chat_template 로딩: {BASE_MODEL}")
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=args.token)
    if not base_tok.chat_template:
        print("오류: base model에도 chat_template이 없습니다.")
        return
    print(f"  chat_template 로딩 완료 ({len(base_tok.chat_template)}자)")

    for model_id in args.models:
        try:
            patch(model_id, base_tok.chat_template, args.token)
        except Exception as e:
            print(f"  실패: {e}")


if __name__ == "__main__":
    main()
