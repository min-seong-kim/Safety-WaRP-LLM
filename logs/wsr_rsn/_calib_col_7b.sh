#!/usr/bin/env bash
# 7B 열 검출 top-k 보정: 현재 진행 중인 검출이 끝나면 후보 k 로 safety 검출만 돌려 파라미터% 측정.
set -uo pipefail
cd /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM
L=logs/wsr_rsn
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/miniconda3/envs/hb/bin/python
B7=$(cat outputs/wsr_sn_tune/llama2_7b/BASIS_DIR)

for f in "$L/col7b.pid" "$L/row13b.pid"; do
  [[ -f "$f" ]] || continue; p=$(cat "$f")
  while kill -0 "$p" 2>/dev/null; do sleep 20; done
  echo "[wait] $f 종료"
done

mkdir -p sn_tune/output_neurons_warp/_calib
for KA in "600:120" "400:80"; do
  KF=${KA%%:*}; KT=${KA##*:}
  OUT="sn_tune/output_neurons_warp/_calib/safety_k${KF}_${KT}.txt"
  [[ -f "$OUT" ]] && { echo "[SKIP] k=${KF}/${KT}"; continue; }
  echo "######## 보정 검출 k_ffn=${KF} k_attn=${KT} ########"
  "$PY" sn_tune/run_warp_sn_pipeline.py \
      --model_name meta-llama/Llama-2-7b-chat-hf --basis_dir "$B7" \
      --dataset_file data/circuit_breakers_train.json \
      --output_dir "outputs/wsr_rsn_col/_calib_k${KF}" \
      --neuron_output_file "$OUT" \
      --layer_types ffn_up,ffn_down,attn_q,attn_k,attn_v \
      --num_prompts 4994 --top_k_ffn "$KF" --top_k_attn "$KT" \
      --freq_threshold 1.0 --max_seq_len 1024 --gpu 0 --dtype bfloat16 \
      --detection_only 2>&1 | tail -3
  echo "---- 요약 k=${KF}/${KT} ----"
  "$PY" -m sn_tune.neuron_file "$OUT" --model_name meta-llama/Llama-2-7b-chat-hf 2>/dev/null | tail -5
done
echo "######## CALIB DONE ########"
