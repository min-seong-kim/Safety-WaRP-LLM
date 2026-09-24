#!/usr/bin/env bash
# 2026-09-22 사용자 요청: 주 셀(k1200_200_u300_50)의 [6] RSN-Tune 모델과 critical 열 마스크를 그대로 쓰되,
# [7] GSM8K 를 Phase 3 **freeze 변형**(--non_freeze 없음)으로 돌린다 → q,k,v,up,down 의 basis_coeff 만 학습,
# o_proj·gate_proj·임베딩·lm_head·norm 전부 동결 (wd 는 0 으로 강제됨). 셀: k1200_200_u300_50_tgtonly
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
source scripts/env.sh >/dev/null 2>&1 || true
L=logs/wsr_rsn_col; OUT=outputs/wsr_rsn_col_p3
PY="$(conda env list | awk '$1=="hb"{print $NF}')/bin/python"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
log() { echo "[$(date '+%F %T')] $*"; }
SRC="$OUT/llama2_7b/k1200_200_u300_50"; CELL="${SRC}_tgtonly"; mkdir -p "$CELL"
TUNED="$(cat $SRC/TUNED_MODEL_DIR)"; BASIS_DIR="$(cat outputs/wsr_sn_tune/llama2_7b/BASIS_DIR)"
cp "$SRC/TUNED_MODEL_DIR" "$CELL/TUNED_MODEL_DIR"; [[ -e "$CELL/masks_freeze" ]] || ln -s "$(readlink -f $SRC/masks_freeze)" "$CELL/masks_freeze"

log "0. Phase 2 arm C (pid $(cat $L/p2.pid)) 종료 대기 → GPU 여유 확보"
while kill -0 "$(cat $L/p2.pid)" 2>/dev/null; do sleep 30; done; sleep 60

GSM_OUT="$CELL/gsm8k"; mkdir -p "$GSM_OUT"
if [[ ! -f "$CELL/MODEL_DIR" ]]; then
    log "1. [7] GSM8K FT, freeze 변형 (q,k,v,up,down basis_coeff 만 학습), start=$TUNED"
    "$PY" train.py --phase 3 --phase0_model_dir "$TUNED" --basis_dir "$BASIS_DIR" --masks_dir "$CELL/masks_freeze" \
        --phase3_dataset gsm8k --phase3_task_data_path data/gsm8k_train_task_7473.json --phase3_task_samples 0 \
        --epochs 3 --utility_lr 5e-5 --base_weight_decay 0.01 --warmup_ratio 0.1 --lr_scheduler_type cosine --max_grad_norm 1.0 \
        --max_length 1024 --batch_size 4 --gradient_accumulation_steps 4 \
        --layer_type ffn_up,ffn_down,attn_q,attn_k,attn_v --target_layers all \
        --output_dir "$GSM_OUT" --log_dir "$GSM_OUT/logs" --device cuda --dtype bfloat16 --seed 42 \
        --gradient_checkpointing --no_wandb --profile_json "$GSM_OUT/profile.json" > "$L/train_tgtonly_7b_$(date +%Y%m%d_%H%M%S).log" 2>&1
    p3="$(find "$GSM_OUT" -maxdepth 1 -type d -name 'phase3_*' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
    [[ -n "$p3" && -f "$p3/final_model/config.json" ]] || { log "⚠️ [7] 실패 — $L/train_tgtonly_7b_*.log"; exit 1; }
    echo "$p3/final_model" > "$CELL/MODEL_DIR"
    "$PY" - "$CELL" <<'EOF'
import json, sys, os
cell = sys.argv[1]
src = json.load(open(os.path.join(os.path.dirname(cell), "k1200_200_u300_50", "cell_config.json")))
src.update({"cell": os.path.basename(cell), "arm": "rsn_tgtonly",
            "stage7_gsm8k": {"trainer": "phase3 freeze variant (basis_coeff of q,k,v,up,down only; o_proj/gate/embed/lm_head/norm frozen; wd forced 0)",
                             "dataset": "gsm8k_train_task_7473.json"},
            "model_dir": open(os.path.join(cell, "MODEL_DIR")).read().strip()})
json.dump(src, open(os.path.join(cell, "cell_config.json"), "w"), indent=2)
EOF
    log "   완료: $(cat $CELL/MODEL_DIR)"
fi

log "2. 평가 (gradcol 학습과 병행, GPU util 0.35)"
CELLS="$CELL" GPU_UTIL_7B=0.35 bash sn_tune/scripts/eval_wsr_rsn_col.sh > "$L/eval_tgtonly_$(date +%Y%m%d_%H%M%S).log" 2>&1
log "   평가 rc=$?"
grep 'tgtonly' ~/HarmBench/results/evaluation_summary_2026-09-2*.csv | tail -1 | cut -d, -f1-7
date -Iseconds > "$L/tgtonly_7b.done"
log "TGTONLY 7B COMPLETE"
