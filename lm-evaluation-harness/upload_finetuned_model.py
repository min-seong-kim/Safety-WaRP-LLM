#!/usr/bin/env python3
"""
HuggingFace Hub에 파인튜닝된 모델을 업로드하는 스크립트

python upload_finetuned_model.py \
  ./MATH_llama32_3b_full_sft_20260326_123531 \
  --repo_id kmseong/Llama-3.2-3B-instruct-MATH-finetuned \
  --hf_token 

"""

import argparse
import os
from pathlib import Path
from huggingface_hub import upload_folder, login, create_repo, repo_exists


def main():
    parser = argparse.ArgumentParser(
        description="Upload a fine-tuned model to HuggingFace Hub"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory (e.g., ./metamath_llama32_3b_optimized_20260319_114912)"
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        type=str,
        help="HuggingFace repository ID (e.g., kmseong/Llama-3.2-3B-only-sn-tuned)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (if not provided, uses ~/.huggingface/token)"
    )

    args = parser.parse_args()

    # 모델 경로 검증
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}")
    
    if not model_path.is_dir():
        raise NotADirectoryError(f"모델 경로는 디렉토리여야 합니다: {model_path}")

    print(f"📦 모델 경로: {model_path.absolute()}")
    print(f"🚀 저장소 ID: {args.repo_id}")

    # HuggingFace 로그인
    if args.hf_token:
        login(token=args.hf_token)
        print("✓ HuggingFace 토큰으로 로그인했습니다.")
    else:
        print("⚠️  HuggingFace 토큰을 제공하지 않았습니다. ~/.huggingface/token 사용 시도 중...")

    # 저장소 존재 여부 확인 및 생성
    print(f"\n🔍 저장소 확인 중...")
    if not repo_exists(repo_id=args.repo_id, repo_type="model"):
        print(f"📝 저장소를 생성 중: {args.repo_id}")
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ 저장소 생성 완료: {args.repo_id}")
    else:
        print(f"✓ 저장소 이미 존재: {args.repo_id}")

    # 모델 업로드
    print("\n📤 모델 업로드 중...")
    try:
        repo_url = upload_folder(
            repo_id=args.repo_id,
            folder_path=str(model_path),
            repo_type="model",
            commit_message="Upload fine-tuned model"
        )
        print(f"\n✓ 업로드 완료!")
        print(f"🔗 저장소 URL: {repo_url}")
    except Exception as e:
        print(f"\n✗ 업로드 실패: {e}")
        raise


if __name__ == "__main__":
    main()
