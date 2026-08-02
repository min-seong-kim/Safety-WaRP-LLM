"""RESTA safety-vector 덧셈 (mergekit 없이).

RESTA(arXiv:2402.11746) 는 downstream FT 로 무너진 safety 를 weight space 에서 복구한다:

    W_resta = W_ft + γ · (W_align − W_unaligned)

`/home/edgeai_lab/resta/README.md` 상단의 랩 내부 사용법이 이 구현이 재현하는 대상이다:

    model1 = downstream FT 모델        weight1 = 1.0
    model2 = safety-aligned 모델       weight2 = +γ
    model3 = 원본(base) 모델           weight3 = −γ

즉 safety vector 를 `W_align − W_base` 로 잡는다(원 논문의 unaligned counterpart 대신).

왜 mergekit 을 안 쓰는가:
  `pip install -e resta/merge` 는 transformers 를 자기 핀으로 끌어내려 이 레포 환경(hb)을
  깨뜨릴 위험이 있다. mergekit 의 `linear` merge 는 `Σ wᵢ·θᵢ` 를 가중치 합으로 정규화하는데
  (normalize=True 기본), 여기 가중치 합은 1.0 + γ − γ = 1.0 이라 정규화가 항등이다.
  따라서 아래 단순 가중합은 mergekit 결과와 수학적으로 동일하다.

구현 노트:
  - model1 의 샤드 단위로 스트리밍하며 계산한다 (피크 RAM ≈ 샤드 1개 + 텐서 2개).
  - 누적은 float32 로 하고 저장 직전에 원 dtype 으로 되돌린다.
  - config / tokenizer / chat_template 등 부속 파일은 model1 것을 복사한다
    (chat_template.jinja 가 빠지면 평가 포맷이 어긋나므로 존재 여부를 검증한다).

사용:
  python scripts/resta_add_safety.py \
      --model1 outputs/cls_baselines/sst2_ft_lr1e-5_ep1 --weight1 1.0 \
      --model2 kmseong/llama2_7b-chat-Safety-FT-lr5e-5 --weight2 0.5 \
      --model3 meta-llama/Llama-2-7b-chat-hf          --weight3 -0.5 \
      --output_path outputs/cls_baselines/sst2_resta_gamma0.5
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SIDE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
]


def resolve_model_dir(ref: str) -> Path:
    """로컬 경로면 그대로, HF repo id 면 snapshot_download."""
    p = Path(ref)
    if p.exists() and p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    print(f"  downloading {ref} ...")
    return Path(snapshot_download(ref, allow_patterns=["*.safetensors", "*.json", "*.model", "*.jinja"]))


def build_key_map(model_dir: Path) -> dict:
    """param key -> safetensors 파일 경로."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {k: model_dir / v for k, v in weight_map.items()}

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"safetensors 파일이 없습니다: {model_dir} "
                                "(.bin 만 있는 모델은 지원하지 않습니다)")
    key_map = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                key_map[k] = shard
    return key_map


class LazyModel:
    """key 로 텐서를 꺼내오되 파일 핸들을 캐싱한다."""

    def __init__(self, ref: str):
        self.ref = ref
        self.dir = resolve_model_dir(ref)
        self.key_map = build_key_map(self.dir)
        self._handles = {}

    def keys(self):
        return set(self.key_map)

    def get(self, key: str) -> torch.Tensor:
        path = self.key_map[key]
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt")
        return self._handles[path].get_tensor(key)


def main():
    ap = argparse.ArgumentParser(description="RESTA linear merge (mergekit-free).")
    ap.add_argument("--model1", required=True, help="downstream FT 모델 (compromised)")
    ap.add_argument("--model2", required=True, help="safety-aligned 모델")
    ap.add_argument("--model3", required=True, help="원본 base 모델")
    ap.add_argument("--weight1", type=float, default=1.0)
    ap.add_argument("--weight2", type=float, required=True, help="+γ")
    ap.add_argument("--weight3", type=float, required=True, help="−γ")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    out_dtype = getattr(torch, args.dtype)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_sum = args.weight1 + args.weight2 + args.weight3
    print("\n---- RESTA linear merge ----")
    print(f"  final = {args.weight1}*[{args.model1}]")
    print(f"        + {args.weight2}*[{args.model2}]")
    print(f"        + {args.weight3}*[{args.model3}]")
    print(f"  weight sum = {weight_sum:.6f} (1.0 이어야 mergekit normalize 와 동일)")
    print(f"  out dtype  = {args.dtype}    ->  {out_dir}\n")
    if abs(weight_sum - 1.0) > 1e-6:
        print("  ⚠️ 가중치 합이 1.0 이 아닙니다. mergekit(normalize=True) 과 결과가 달라집니다.")

    m1 = LazyModel(args.model1)
    m2 = LazyModel(args.model2)
    m3 = LazyModel(args.model3)

    k1, k2, k3 = m1.keys(), m2.keys(), m3.keys()

    # model1 의 키 전부가 나머지 두 모델에 있어야 한다. 없으면 병합 불가.
    missing2, missing3 = k1 - k2, k1 - k3
    if missing2 or missing3:
        raise ValueError(
            "model1 의 파라미터가 다른 모델에 없습니다. 병합할 수 없습니다.\n"
            f"  model2 에 없는 키(예): {sorted(missing2)[:5]}\n"
            f"  model3 에 없는 키(예): {sorted(missing3)[:5]}"
        )

    # 반대로 model2/3 에만 있는 키는 무시한다. 단, "무시해도 되는 것"만 허용한다.
    #   .inv_freq : rotary embedding 의 non-persistent 버퍼. config(rope_theta, head_dim)에서
    #               결정되는 캐시라 학습 대상이 아니고, 구형 체크포인트
    #               (meta-llama/Llama-2-7b-chat-hf 등)에만 남아 있다. mergekit 도 아키텍처
    #               정의 텐서만 병합하므로 동일하게 제외된다.
    IGNORABLE_SUFFIXES = (".rotary_emb.inv_freq", ".inv_freq")
    extras = (k2 | k3) - k1
    ignorable = {k for k in extras if k.endswith(IGNORABLE_SUFFIXES)}
    unexpected = extras - ignorable
    if unexpected:
        raise ValueError(
            "model2/model3 에만 있는 예상 밖의 키가 있습니다. 병합을 중단합니다.\n"
            f"  예: {sorted(unexpected)[:5]}  (총 {len(unexpected)}개)"
        )
    if ignorable:
        print(f"  note: 병합에서 제외한 버퍼 {len(ignorable)}개 "
              f"(예: {sorted(ignorable)[0]}) — 학습 파라미터가 아님")

    # model1 의 샤드 구조를 그대로 유지한다.
    shards = {}
    for key, path in m1.key_map.items():
        shards.setdefault(path.name, []).append(key)

    total = 0
    for shard_name in sorted(shards):
        keys = shards[shard_name]
        merged = {}
        for key in keys:
            t1 = m1.get(key)
            acc = t1.to(torch.float32) * args.weight1
            acc += m2.get(key).to(torch.float32) * args.weight2
            acc += m3.get(key).to(torch.float32) * args.weight3
            merged[key] = acc.to(out_dtype).contiguous()
            total += 1
        save_file(merged, str(out_dir / shard_name), metadata={"format": "pt"})
        print(f"  saved {shard_name}  ({len(keys)} tensors)")
        del merged

    for name in SIDE_FILES:
        src = m1.dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # config 의 dtype 표기를 실제 저장 dtype 과 맞춘다.
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for field in ("torch_dtype", "dtype"):
            if field in cfg:
                cfg[field] = args.dtype
        cfg_path.write_text(json.dumps(cfg, indent=2))

    meta = {
        "method": "resta",
        "model1": args.model1, "weight1": args.weight1,
        "model2": args.model2, "weight2": args.weight2,
        "model3": args.model3, "weight3": args.weight3,
        "gamma": args.weight2,
        "dtype": args.dtype,
        "num_tensors": total,
        "note": "linear merge; weight sum 1.0 이므로 mergekit linear(normalize=True) 와 동일",
    }
    (out_dir / "resta_merge.json").write_text(json.dumps(meta, indent=2))

    has_template = (out_dir / "chat_template.jinja").exists() or (
        "chat_template" in json.loads((out_dir / "tokenizer_config.json").read_text())
        if (out_dir / "tokenizer_config.json").exists() else False
    )
    print(f"\n  merged {total} tensors  ->  {out_dir}")
    print(f"  chat_template present: {has_template}")
    if not has_template:
        print("  ⚠️ chat_template 이 없습니다. model1 에 chat_template.jinja 가 있는지 확인하세요.")


if __name__ == "__main__":
    main()
