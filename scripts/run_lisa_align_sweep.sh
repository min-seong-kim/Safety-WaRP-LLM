#!/usr/bin/env bash
#
# LISA alignment_step 축 스윕 — proximal 은 끄고 safety 데이터 노출만 늘린다.
#
# 근거 (측정치):
#   SSFT+DT (무보호)  GSM8K 0.4117  ASR 0.2064   (Direct 0 / AutoDAN 0 / PAIR .3154 / PAP .5100)
#   rho0-alt          GSM8K 0.3867  ASR 0.1288   (Direct 0 / AutoDAN .0038 / PAIR .1327 / PAP .3788)
#   rho=0.1           GSM8K 0.19    (효용 붕괴)
#
#   → alternation 은 효용 6% 비용으로 ASR 38% 를 깎는 효율적 축.
#     proximal 은 rho=0.1 만으로도 무제약 drift 의 99.94% 를 제거해 학습을 막는다.
#     따라서 rho=0 을 고정하고 alignment_step 만 올려 남은 약점(PAP)을 공략한다.
#
# alignment_step / finetune_step = 안전 데이터가 차지하는 스텝 비율:
#   100/900 → 10%   (rho0-alt, 기준)
#   200/900 → 18%
#   300/900 → 25%
#
set -uo pipefail          # -e 를 쓰지 않는다: 한 run 이 실패해도 나머지를 계속 돌린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-/venv/hb/bin/python}"
NS="${HF_NAMESPACE:-kmseong}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
ALIGN_LIST="${ALIGN_LIST:-200 300}"
FINETUNE_STEP="${FINETUNE_STEP:-900}"
ROOT="$REPO_DIR/outputs/lisa_align_sweep"

failed=()
for align in $ALIGN_LIST; do
    tag="align${align}"
    out="$ROOT/$tag"
    if [[ -f "$out/finetune_config.json" ]]; then
        echo "[$tag] already complete; skipping"; continue
    fi
    mkdir -p "$out"
    echo "════════════════════════════════════════════════════════════"
    echo "  [$tag]  rho=0  alignment_step=$align  finetune_step=$FINETUNE_STEP  ($(date '+%F %T'))"
    echo "════════════════════════════════════════════════════════════"
    if "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
        --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
        --output_dir "$out" \
        --dataset_name openai/gsm8k --dataset_subset main --train_split train \
        --num_train_samples 7473 --num_eval_samples 0 \
        --safety_data_path "$REPO_DIR/data/circuit_breakers_train.json" \
        --guide_data_num 4994 \
        --rho 0 --alignment_step "$align" --finetune_step "$FINETUNE_STEP" \
        --lora --lora_target_modules q_proj k_proj v_proj up_proj down_proj \
        --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --learning_rate "$LR" --epochs "$EPOCHS" \
        --batch_size 16 --grad_accum 1 \
        --max_length 1024 --warmup_ratio 0.03 --weight_decay 0.0 \
        --lr_scheduler_type cosine --seed 42 \
        --bf16 --gradient_checkpointing --report_to none \
        --upload_name "$NS/llama2_7b-chat-gsm8k-lisa-cb-r16a32-lr${LR}-ep${EPOCHS}-rho0-${tag}" \
        2>&1 | tee "$out/run.log"; then
        echo "[OK] $tag"
    else
        rc=$?; echo "[FAIL rc=$rc] $tag — 다음 run 으로 계속" >&2; failed+=("$tag rc=$rc")
    fi
done

if ((${#failed[@]})); then
    echo "실패한 run:"; printf '  %s\n' "${failed[@]}"; exit 1
fi
echo "align sweep 완료: $ALIGN_LIST → $ROOT"
