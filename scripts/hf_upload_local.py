"""로컬 merged_model 디렉토리를 HF Hub 에 업로드 (모델 로드 없이 파일 직접 업로드).
사용: python scripts/hf_upload_local.py <local_dir> <repo_id>
"""
import sys
from huggingface_hub import HfApi

def main():
    local_dir, repo_id = sys.argv[1], sys.argv[2]
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"uploading {local_dir} -> {repo_id}")
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    info = api.model_info(repo_id)
    print(f"OK: https://huggingface.co/{repo_id}  (files: {len(info.siblings)})")

if __name__ == "__main__":
    main()
