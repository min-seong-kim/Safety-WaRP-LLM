#!/usr/bin/env bash
#
# LISA 성능 저하 원인 분리 실험.
#
# 관측: 기본 GSM8K SFT acc 0.4117 vs LISA rho=0.1 acc 0.19.
# 데이터 파이프라인(tokenize/collator/prompt)은 finetune_gsm8k_full_params.py 와
# 로직이 동일함을 정적 대조로 확인했으므로, 원인은 LISA 고유 기구 두 가지 중 하나다:
#
#   (1) proximal term      — rho/2 * ||theta - anchor||^2
#   (2) dataset alternation — 전체 스텝의 100/1000 을 safety(거부 응답) 데이터로 학습
#
# 이를 분리하는 사다리:
#
#   A) rho=0,  guide=4994 → alternation 만 남김.  (1) 제거
#   B) rho=0,  guide=0    → BSO 전부 off = 순수 LoRA SFT. (1)+(2) 제거
#                            여기서 acc 가 0.41 근처로 돌아오면 학습 코드는 정상이고
#                            저하는 LISA 설계 그대로의 결과다.
#                            여전히 0.19 면 LISA 스크립트 자체에 문제가 있다.
#
# B 는 vanilla LoRA(finetune_gsm8k_lora.py --method lora) 와 같은 설정이므로
# 두 결과가 일치해야 한다 — 이것이 "학습 코드가 정상인가"의 직접 검증이다.
#
set -uo pipefail          # -e 는 쓰지 않는다: 한 run 이 실패해도 나머지를 계속 돌린다.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PY="${PY:-/venv/hb/bin/python}"
NS="${HF_NAMESPACE:-kmseong}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
ROOT="$REPO_DIR/outputs/lisa_diagnose"

run() {   # $1=tag  $2=rho  $3=guide_data_num
    local tag=$1 rho=$2 guide=$3
    local out="$ROOT/$tag"
    if [[ -f "$out/finetune_config.json" ]]; then
        echo "[$tag] already complete; skipping"; return 0
    fi
    mkdir -p "$out"
    echo "════════════════════════════════════════════════════════════"
    echo "  [$tag]  rho=$rho  guide_data_num=$guide  ($(date '+%F %T'))"
    echo "════════════════════════════════════════════════════════════"
    "$PY" gsm8k_eval/finetune_gsm8k_lisa.py \
        --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
        --output_dir "$out" \
        --dataset_name openai/gsm8k --dataset_subset main --train_split train \
        --num_train_samples 7473 --num_eval_samples 0 \
        --safety_data_path "$REPO_DIR/data/circuit_breakers_train.json" \
        --guide_data_num "$guide" \
        --rho "$rho" --alignment_step 100 --finetune_step 900 \
        --lora --lora_target_modules q_proj k_proj v_proj up_proj down_proj \
        --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --learning_rate "$LR" --epochs "$EPOCHS" \
        --batch_size 16 --grad_accum 1 \
        --max_length 1024 --warmup_ratio 0.03 --weight_decay 0.0 \
        --lr_scheduler_type cosine --seed 42 \
        --bf16 --gradient_checkpointing --report_to none \
        --upload_name "$NS/llama2_7b-chat-gsm8k-lisa-cb-r16a32-lr${LR}-ep${EPOCHS}-${tag}" \
        2>&1 | tee "$out/run.log"
}

# A: proximal 제거, alternation 유지
run "rho0-alt"     0   4994
# B: BSO 전부 off = 순수 LoRA SFT (vanilla LoRA 와 동일 설정 → 코드 검증용)
run "nobso-puresft" 0   0

echo "diagnose 완료: $ROOT"
