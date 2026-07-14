"""
Stage 1.5 — 학습된 selector로 downstream(gsm8k) 데이터 top-p% 선택.

SEAL Stage 2의 `torch.topk(selector, int(topp*N))`과 동일.
selector logits(.pt, raw tensor)를 로드해 상위 topp 비율 인덱스를 고르고
JSON으로 저장한다. 이 인덱스는 gsm8k 전체 train(순서 고정)의 위치를 가리킨다.

출력: <out>  (json) = {"topp":..., "num_selected":..., "total":..., "indices":[...] }

사용 예:
  python -m seal.select_data \
      --selector_path seal/ckpt/gsm8k_selector_softmax.pt \
      --topp 0.8 --out seal/ckpt/gsm8k_selected_top80.json
"""

import argparse
import json
import os

import torch


def parse_args():
    p = argparse.ArgumentParser(description="selector로 top-p% 데이터 선택 (Stage 1.5)")
    p.add_argument("--selector_path", type=str, required=True,
                   help="train_selector.py가 저장한 selector logits(.pt)")
    p.add_argument("--topp", type=float, default=0.8,
                   help="유지할 상위 비율 (0~1). 1.0이면 전체 사용(선택 없음).")
    p.add_argument("--out", type=str, required=True,
                   help="선택된 인덱스를 저장할 json 경로")
    return p.parse_args()


def main():
    args = parse_args()
    assert 0.0 < args.topp <= 1.0, "topp는 (0, 1] 범위여야 합니다."

    logits = torch.load(args.selector_path, map_location="cpu")
    if not torch.is_tensor(logits):
        # state dict로 저장된 경우 대비
        logits = logits["logits"] if "logits" in logits else torch.tensor(logits)
    logits = logits.float().flatten()

    n = logits.numel()
    k = int(args.topp * n)
    k = max(1, min(k, n))
    topk = torch.topk(logits, k)
    indices = sorted(topk.indices.tolist())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "selector_path": os.path.abspath(args.selector_path),
        "topp": args.topp,
        "total": n,
        "num_selected": k,
        "indices": indices,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"[select] selector={args.selector_path}")
    print(f"[select] total={n}  topp={args.topp}  selected={k}")
    print(f"[select] ✅ saved indices → {args.out}")


if __name__ == "__main__":
    main()
