#!/usr/bin/env python3
"""
test_neuron_file.py — 뉴런 파일 포맷과 critical 차집합 자체 점검.

모델도 GPU 도 필요 없다 (순수 로직). 파이프라인을 돌리기 전에 한 번 통과시켜라.

    python -m sn_tune.test_neuron_file

여기서 지키는 것
  [1] 5줄 포맷 왕복
  [2] **줄 순서가 계약이다** — 키가 아니라 위치로 배정된다
  [3] 개수 집계
  [4] 빈 검출 결과 가드 (조용히 통과하면 몇 시간 뒤 학습이 죽는다)
  [5] critical = safety \\ utility
  [6] overlap 통계
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sn_tune.critical_neurons import compute_critical, overlap_stats
from sn_tune.neuron_file import (
    MODULE_ORDER,
    assert_nonempty,
    count_neurons,
    load_neuron_file,
    save_neuron_file,
)

SAFETY = {
    "ffn_up": {0: {1, 2, 3}, 1: {4, 5}},
    "ffn_down": {0: {7}},
    "q": {0: {9}},
    "k": {},
    "v": {2: {11, 12}},
}
UTILITY = {
    "ffn_up": {0: {2, 3, 99}},
    "ffn_down": {},
    "q": {0: {9}},
    "k": {},
    "v": {},
}


def main() -> int:
    passed = 0

    # [1] 5줄 왕복
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.txt")
        save_neuron_file(p, SAFETY)
        lines = open(p, encoding="utf-8").read().rstrip("\n").split("\n")
        assert len(lines) == 5, f"5줄이어야 한다 (got {len(lines)})"
        back = load_neuron_file(p)
        assert back["ffn_up"] == {0: {1, 2, 3}, 1: {4, 5}}, back["ffn_up"]
        assert back["k"] == {}, back["k"]
        assert back["v"] == {2: {11, 12}}, back["v"]
    print("[1] 5줄 왕복 OK"); passed += 1

    # [2] 줄 순서 계약 — q 는 반드시 인덱스 2
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.txt")
        save_neuron_file(p, SAFETY)
        lines = open(p, encoding="utf-8").read().rstrip("\n").split("\n")
        assert 9 not in json.loads(lines[0]).get("0", []), "0번 줄은 ffn_up 이다"
        assert json.loads(lines[2])["0"] == [9], "q 는 3번째 줄(인덱스 2)"
        assert MODULE_ORDER == ("ffn_up", "ffn_down", "q", "k", "v"), MODULE_ORDER
    print("[2] 줄 순서 계약 OK"); passed += 1

    # [3] 개수 집계
    c = count_neurons(SAFETY)
    assert (c["ffn_up"], c["ffn"], c["attn"], c["total"]) == (5, 6, 3, 9), c
    print("[3] 개수 집계 OK"); passed += 1

    # [4] 빈 결과 가드
    empty = {m: {0: set()} for m in MODULE_ORDER}
    try:
        assert_nonempty("dummy", empty)
    except ValueError:
        print("[4] 빈 결과 가드 OK"); passed += 1
    else:
        print("[4] FAIL — 빈 뉴런 파일이 통과했다"); return 1

    # [5] critical = safety \ utility
    crit = compute_critical(SAFETY, UTILITY)
    assert crit["ffn_up"][0] == {1}, crit["ffn_up"][0]       # {1,2,3} - {2,3,99}
    assert crit["ffn_up"][1] == {4, 5}, crit["ffn_up"][1]    # utility 없는 레이어는 그대로
    assert crit["q"][0] == set(), crit["q"][0]               # 완전히 겹치면 빈 집합
    assert crit["v"][2] == {11, 12}, crit["v"][2]
    print("[5] critical 차집합 OK"); passed += 1

    # [6] overlap 통계
    st = overlap_stats(SAFETY, UTILITY, crit)
    assert st["ffn_up"]["overlap"] == 2, st["ffn_up"]
    assert st["ffn_up"]["critical"] == 3, st["ffn_up"]
    print("[6] overlap 통계 OK"); passed += 1

    print(f"\n{passed}/6 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
