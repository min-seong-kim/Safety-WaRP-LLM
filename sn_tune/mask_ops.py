"""
Phase 3 마스크 디렉토리 유틸.

    python -m sn_tune.mask_ops invert  --input <masks> --output_dir <out>
        mask=1(동결) ↔ mask=0(학습) 을 뒤집는다. arm C(열 동결) 마스크에서 "그 열만 학습" 하는
        (R)SN-Tune 단계용 tune 마스크를 만들 때 쓴다.
    python -m sn_tune.mask_ops columns --input <masks> --output_file <5줄 뉴런 파일>
        열 마스크(mask.any(axis=0))를 5줄 열 인덱스 파일로 기록한다 (기록·비교용).
    python -m sn_tune.mask_ops stats   --input <masks>
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

LT_TO_MODULE = {"ffn_up": "ffn_up", "ffn_down": "ffn_down", "attn_q": "q", "attn_k": "k", "attn_v": "v"}
MODULE_ORDER = ["ffn_up", "ffn_down", "q", "k", "v"]


def load_dir(d):
    out = {}
    for lt in sorted(os.listdir(d)):
        p = os.path.join(d, lt)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.startswith("layer_") and f.endswith("_mask.pt"):
                m = torch.load(os.path.join(p, f), weights_only=False)["mask"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m)
                if m.dtype != torch.bool:
                    m = m > 0.5
                out[(lt, int(f.split("_")[1]))] = m
    if not out:
        raise SystemExit(f"마스크가 없다: {d}")
    return out


def read_meta(d):
    p = os.path.join(d, "metadata.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def cmd_invert(a):
    masks = load_dir(a.input)
    tot = frozen = 0
    for (lt, idx), m in masks.items():
        inv = ~m
        d = os.path.join(a.output_dir, lt); os.makedirs(d, exist_ok=True)
        torch.save({"mask": inv}, os.path.join(d, f"layer_{idx:02d}_mask.pt"))
        tot += inv.numel(); frozen += int(inv.sum())
    src = read_meta(a.input)
    meta = {"phase": 2, "keep_ratio": frozen / tot, "masking_strategy": f"invert({src.get('masking_strategy')})",
            "layer_types": sorted({lt for lt, _ in masks}), "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "two_mask": False, "source": os.path.abspath(a.input), "source_keep_ratio": src.get("keep_ratio"),
            "note": "mask=1 동결. 원본에서 학습되던 좌표만 동결하고 원본에서 동결되던 좌표만 학습한다."}
    json.dump(meta, open(os.path.join(a.output_dir, "metadata.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[invert] {len(masks)} masks → {a.output_dir}  frozen {100*frozen/tot:.2f}%")


def cmd_columns(a):
    masks = load_dir(a.input)
    neurons = {m: {} for m in MODULE_ORDER}
    n = 0
    for (lt, idx), m in masks.items():
        mod = LT_TO_MODULE.get(lt)
        if mod is None:
            continue
        cols = torch.nonzero(m.any(dim=0)).flatten().tolist()
        neurons[mod][idx] = cols; n += len(cols)
    os.makedirs(os.path.dirname(os.path.abspath(a.output_file)) or ".", exist_ok=True)
    with open(a.output_file, "w") as fh:
        for mod in MODULE_ORDER:
            fh.write(json.dumps({str(k): v for k, v in sorted(neurons[mod].items())}) + "\n")
    json.dump({"format": "5-line WaRP column-index file", "source_masks": os.path.abspath(a.input),
               "total_selected_columns": n, "per_module": {m: sum(len(v) for v in neurons[m].values()) for m in MODULE_ORDER}},
              open(a.output_file + ".meta.json", "w"), indent=2)
    print(f"[columns] {n} columns → {a.output_file}")


def cmd_stats(a):
    masks = load_dir(a.input)
    by = {}
    for (lt, idx), m in masks.items():
        s = by.setdefault(lt, [0, 0, 0]); s[0] += int(m.sum()); s[1] += m.numel(); s[2] += int(m.any(dim=0).sum())
    for lt, (f, t, c) in by.items():
        print(f"  {lt:9s} frozen {100*f/t:6.2f}%  cols={c}")
    f = sum(v[0] for v in by.values()); t = sum(v[1] for v in by.values())
    print(f"  total     frozen {100*f/t:6.2f}%  ({f:,}/{t:,})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("invert"); p.add_argument("--input", required=True); p.add_argument("--output_dir", required=True)
    p = sp.add_parser("columns"); p.add_argument("--input", required=True); p.add_argument("--output_file", required=True)
    p = sp.add_parser("stats"); p.add_argument("--input", required=True)
    a = ap.parse_args(argv)
    {"invert": cmd_invert, "columns": cmd_columns, "stats": cmd_stats}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
