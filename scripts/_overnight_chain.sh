#!/usr/bin/env bash
#
# 야간 체인: LISA rho 스윕이 끝나기를 기다렸다가 → 디스크 정리 → SafeLoRA threshold 스윕.
# (일회성 운영 스크립트. 실험 재현에는 각 스윕 스크립트를 직접 부를 것)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
exec > >(tee -a "logs/overnight_chain_${TS}.log") 2>&1

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

echo "[$(date '+%F %T')] rho 스윕 종료 대기 중..."
while pgrep -f "run_lisa_rho_sweep_qa.sh" >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] rho 스윕 종료 확인. free=$(free_gb)GB"

# ── 디스크 정리: HF 에 사본이 있어 복구 가능한 것만, 필요한 만큼만 ──
# SafeLoRA 스윕 4 run × 13GB = 52GB + 여유 20GB = 72GB 필요.
NEED=72
reclaim() {   # $1=경로  $2=설명
    [ -e "$1" ] || return 0
    [ "$(free_gb)" -ge "$NEED" ] && return 0
    echo "  정리: $1 ($2)"
    rm -rf "$1"
    echo "  → free=$(free_gb)GB"
}
if [ "$(free_gb)" -lt "$NEED" ]; then
    echo "[$(date '+%F %T')] 디스크 정리 시작 (free=$(free_gb)GB < ${NEED}GB)"
    # 1) HF 업로드 완료 확인된 로컬 사본
    reclaim "$REPO_DIR/outputs/lisa_beavertails" \
            "kmseong/llama2_7b-chat-gsm8k-lisa-bt-... 로 업로드 완료됨"
    # 2) 이번 스윕에 쓰이지 않는 캐시 모델 (HF 원본 존재)
    reclaim "$HOME/.cache/huggingface/hub/models--kmseong--llama2_7b-chat-Safety-FT-lr5e-5" \
            "beavertails run 전용, 종료됨"
else
    echo "[$(date '+%F %T')] 디스크 여유 충분 (free=$(free_gb)GB) — 정리 건너뜀"
fi

echo "[$(date '+%F %T')] SafeLoRA threshold 스윕 시작 (free=$(free_gb)GB)"
bash scripts/run_safelora_thr_sweep_qa.sh
echo "[$(date '+%F %T')] 체인 완료. free=$(free_gb)GB"
