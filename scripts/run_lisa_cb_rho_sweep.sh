#!/usr/bin/env bash
#
# LISA rho 축 스윕 — proximal 세기만 바꿔 downstream 학습이 얼마나 풀리는지 본다.
#
# 배경: rho=1.0 에서 consensus drift 가 proximal 항 활성화 직후 수천 배 붕괴하고
#       (ep50: 2961 → 0.297), loss 가 평탄해졌다. rho 를 낮춰 그 기울기를 측정한다.
#
# 나머지 설정은 run_lisa_cb_ep10.sh 와 동일(r16/a32, q,k,v,up,down, lr 3e-4,
# batch 16, warmup 0.03, wd 0.0, seed 42, align 100 / finetune 900,
# safety = circuit_breakers_train.json, downstream = GSM8K 7473).
#
# 사용법:
#   PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/run_lisa_cb_rho_sweep.sh
#
# 오버라이드:
#   RHO_LIST="0.1 0.5"  EPOCHS=3  LR=3e-4
#
set -uo pipefail          # -e 를 쓰지 않는다: 한 run 이 실패해도 나머지를 계속 돌린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

RHO_LIST="${RHO_LIST:-0.1 0.5}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"

failed=()
for rho in $RHO_LIST; do
    echo "════════════════════════════════════════════════════════════"
    echo "  LISA  lr=$LR  epochs=$EPOCHS  rho=$rho   ($(date '+%F %T'))"
    echo "════════════════════════════════════════════════════════════"
    if RHO="$rho" EPOCHS="$EPOCHS" LR="$LR" bash scripts/run_lisa_cb_ep10.sh; then
        echo "[OK] lr=$LR ep=$EPOCHS rho=$rho"
    else
        rc=$?
        echo "[FAIL rc=$rc] lr=$LR ep=$EPOCHS rho=$rho — 다음 run 으로 계속" >&2
        failed+=("rho=$rho rc=$rc")
    fi
done

if ((${#failed[@]})); then
    echo "실패한 run:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "LISA rho sweep 완료: $RHO_LIST (epochs=$EPOCHS lr=$LR)"
