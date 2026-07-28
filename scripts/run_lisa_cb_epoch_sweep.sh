#!/usr/bin/env bash
#
# LISA epoch 축 스윕 — matched LoRA 예산(r16/a32, q,k,v,up,down)에서 epoch 만 바꾼다.
#
# run_lisa_cb_ep10.sh 를 EPOCHS 만 바꿔 순차 호출한다. 출력 경로가
# outputs/lisa_cb_ep10/lr_<LR>_ep<EPOCHS> 로 갈라지고, 완료된 run 은
# finetune_config.json 가드로 건너뛰므로 죽어도 재실행하면 이어서 간다.
#
# 사용법:
#   PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/run_lisa_cb_epoch_sweep.sh
#
# 오버라이드:
#   EPOCHS_LIST="10 50"  LR=3e-4  PY=/venv/hb/bin/python
#
set -uo pipefail          # -e 는 쓰지 않는다: 한 run 이 실패해도 나머지를 계속 돌린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

EPOCHS_LIST="${EPOCHS_LIST:-10 50}"
LR="${LR:-3e-4}"

failed=()
for ep in $EPOCHS_LIST; do
    echo "════════════════════════════════════════════════════════════"
    echo "  LISA  lr=$LR  epochs=$ep   ($(date '+%F %T'))"
    echo "════════════════════════════════════════════════════════════"
    if EPOCHS="$ep" LR="$LR" bash scripts/run_lisa_cb_ep10.sh; then
        echo "[OK] lr=$LR ep=$ep"
    else
        rc=$?
        echo "[FAIL rc=$rc] lr=$LR ep=$ep — 다음 run 으로 계속" >&2
        failed+=("lr=$LR ep=$ep rc=$rc")
    fi
done

if ((${#failed[@]})); then
    echo "실패한 run:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "LISA epoch sweep 완료: $EPOCHS_LIST"
