'''
python hf_rm.py --username kmseong --prefix llama2-7b-safe-lo --yes
'''

import argparse
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete Hugging Face models that match username/prefix"
    )
    parser.add_argument("--username", default="kmseong", help="Hugging Face username/org")
    parser.add_argument(
        "--prefix",
        default="llama3.2_3b",
        help="Model name prefix after username/ (case-insensitive)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without confirmation prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched repos without deleting",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api = HfApi()

    namespace = args.username.strip()
    target_prefix = args.prefix.strip().lower()
    namespace_prefix = f"{namespace}/"

    models = list(api.list_models(author=namespace, full=True))
    targets = []

    for model in models:
        repo_id = model.id
        if not repo_id.startswith(namespace_prefix):
            continue

        model_name = repo_id[len(namespace_prefix):]
        if model_name.lower().startswith(target_prefix):
            targets.append(repo_id)

    print(f"Found {len(targets)} model(s) starting with '{namespace}/{args.prefix}'")
    for repo_id in targets:
        print(f" - {repo_id}")

    if not targets:
        print("No matching repositories found.")
        return

    if args.dry_run:
        print("Dry run enabled. No repositories were deleted.")
        return

    if not args.yes:
        answer = input("Proceed with deletion? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled. No repositories were deleted.")
            return

    for repo_id in targets:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model")
            print(f"Deleted: {repo_id}")
        except Exception as e:
            print(f"Failed to delete {repo_id}: {e}")


if __name__ == "__main__":
    main()
