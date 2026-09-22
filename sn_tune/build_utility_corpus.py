"""
build_utility_corpus.py — utility(foundation) 검출용 코퍼스를 JSON 으로 덤프한다.

왜 필요한가
──────────
원공간 검출기(`detect_original.py --utility_neuron`)는 wikipedia 를 **자기가 직접** 로드한다.
반면 WaRP 공간 검출기(`run_warp_sn_pipeline.py`)는 `--dataset_file` 로 받은
`[{"prompt": ...}, ...]` JSON 만 읽는다. 그래서 WSR-RSN-Tune 의 utility 쪽을 돌리려면
같은 wikipedia 문서를 그 포맷으로 바꿔줘야 한다.

**두 공간이 같은 문서를 보게 하는 것이 요점이다.** 샘플링 seed(112)와 자르는 길이(2000자)를
`detect_original.load_wikipedia_data` 와 맞춰 두었다. 여기가 어긋나면 원공간 critical 과
WaRP 공간 critical 이 서로 다른 코퍼스에서 나온 것이 되어 두 arm 비교가 성립하지 않는다.

사용
────
    python -m sn_tune.build_utility_corpus \\
        --num_samples 1000 \\
        --output_file ./sn_tune/corpus/wikipedia_utility_1000.json

디스크 주의: 기본은 streaming 이다. `20231101.en` 전체를 내려받으면 100GB 를 훌쩍 넘는다.
`WIKI_STREAMING=0` 으로 비활성화할 수 있지만 그러려면 디스크가 있어야 한다
(원본 `load_wikipedia_data` 와 같은 규칙).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys

from tqdm import tqdm

logger = logging.getLogger(__name__)

# 원공간 검출기와 반드시 같아야 하는 값들
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_SUBSET = "20231101.en"
WIKI_SEED = 112
WIKI_SHUFFLE_BUFFER = 10000
WIKI_DOC_CHARS = 2000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_wikipedia_texts(num_samples: int, cache_dir: str | None = None) -> list[str]:
    """wikipedia 문서 num_samples 개. `detect_original.load_wikipedia_data` 와 동일한 규칙."""
    from datasets import load_dataset

    streaming = os.environ.get("WIKI_STREAMING", "1") != "0"
    logger.info("wikipedia 로드 (subset=%s, streaming=%s)", WIKI_SUBSET, streaming)

    if streaming:
        ds = load_dataset(WIKI_DATASET, WIKI_SUBSET, split="train", streaming=True)
        ds = ds.shuffle(seed=WIKI_SEED, buffer_size=WIKI_SHUFFLE_BUFFER)
        texts: list[str] = []
        with tqdm(total=num_samples, desc="wikipedia") as pbar:
            for item in ds:
                text = (item.get("text") or "").strip()
                if text:
                    texts.append(text[:WIKI_DOC_CHARS])
                    pbar.update(1)
                if len(texts) >= num_samples:
                    break
        return texts

    ds = load_dataset(
        WIKI_DATASET, WIKI_SUBSET, split="train", streaming=False,
        cache_dir=cache_dir or os.path.join(SCRIPT_DIR, "wikipedia_cache"),
    )
    random.seed(WIKI_SEED)
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    texts = []
    for idx in tqdm(indices, desc="wikipedia"):
        text = (ds[idx].get("text") or "").strip()
        if text:
            texts.append(text[:WIKI_DOC_CHARS])
    return texts


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--output_file", required=True,
                    help='[{"prompt": ...}] 형태로 저장할 경로')
    ap.add_argument("--cache_dir", default=None,
                    help="streaming 을 끌 때만 쓰인다 (WIKI_STREAMING=0)")
    args = ap.parse_args(argv)

    texts = load_wikipedia_texts(args.num_samples, args.cache_dir)
    if not texts:
        logger.error("문서를 하나도 못 받았다. 네트워크/HF 캐시를 확인하라.")
        return 1
    if len(texts) < args.num_samples:
        logger.warning("요청 %d개 중 %d개만 받았다.", args.num_samples, len(texts))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump([{"prompt": t} for t in texts], f, ensure_ascii=False)

    logger.info("저장: %s (%d개 문서)", args.output_file, len(texts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
