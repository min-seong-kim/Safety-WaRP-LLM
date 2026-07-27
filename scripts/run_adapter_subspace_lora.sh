#!/bin/bash
# adapter_subspace_lora (= safety_adapter_projected_lora) 전체 실험 파이프라인
#
#   Stage 0   safety LoRA tuning        phase0_SSFT.py --lora            → B_s, A_s
#   Stage 0.5 safety 모델 dense 병합     (baseline / safety 평가 기준점)
#   Stage 1   보호 subspace Q_S 추출     build_adapter_subspace.py
#   Stage 2   GSM8K downstream 학습      finetune_gsm8k_lora.py --method adapter_subspace_lora
#   Stage 3   통제군 (동일 시작점 plain LoRA)
#
# 수식:  W_final = W_base + s_s B_s A_s + s_d B_d A_d (I − Q_S Q_Sᵀ)
#        ⇒ W_final Q_S = W_safe Q_S  (safety adapter 가 쓴 입력 방향은 정확히 보존)
#
# 완료된 stage 는 건너뛰므로 재실행하면 이어서 진행된다.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "logs/adapter_subspace_lora_${TS}.log") 2>&1

# ═══════════════════ config ═══════════════════
# python 인터프리터. 환경마다 다르므로 env 로 덮어쓸 수 있게 둔다:
#   HBPY=/path/to/python bash scripts/run_adapter_subspace_lora.sh
#   구 SLURM 박스: /home/gokms0509/anaconda3/envs/hb/bin/python
#   Vast 박스   : /venv/hb/bin/python  (torch 2.10+cu128 / transformers 4.57.3 / peft 0.18.1)
HBPY=${HBPY:-/venv/hb/bin/python}
# ⚠️ CUDA_VISIBLE_DEVICES 를 여기서 설정하지 않는다.
#    SLURM 박스: sbatch --gres=gpu:N 으로 할당받은 GPU 를 스케줄러가 노출한다.
#    단일 GPU 박스(Vast 등): unset 이면 torch 가 그냥 GPU 0 을 쓴다. 하드코딩할 이유가 없다.
# export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset: 스케줄러/전체 GPU>}"
echo "HBPY=${HBPY}"

BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"        # ⚠️ safety 이전의 BASE 모델
SAFETY_DATA="./data/circuit_breakers_train.json"
TARGET_MODULES="q_proj,k_proj,v_proj,up_proj,down_proj"
LORA_TARGETS_SPACED="q_proj k_proj v_proj up_proj down_proj"   # phase0_SSFT 는 nargs='+'

# safety LoRA (Stage 0)
# ⚠️ epochs(3) / batch(4) / grad_accum(4) / samples(4994) 는 phase0_SSFT.py 모듈 상수라
#    여기서 못 바꾼다. 현재 그 파일 기준 effective batch = 16 (논문 Appendix A 와 동일).
SAFETY_LORA_R=16
SAFETY_LORA_ALPHA=32
SAFETY_LR=1e-4                     # phase0_SSFT.py 의 LEARNING_RATE 기본값과 일치시킴

# downstream LoRA (Stage 2)
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4)
SAFETY_ADAPTER_MODE=merge          # merge | keep
LORA_PARAM_DTYPE=float32           # float32 = 제약 정밀도 ~1e-7 / bfloat16 = ~1e-3
VERIFY_EVERY=0                     # >0 이면 N step 마다 제약 지표 로깅

# 보호 subspace 선택 변형: "<tag>:<extra args>"  (빈 args = all-effective)
# Stage 1 은 수 초라 전부 만들어 두고(스펙트럼 비교용), Stage 2 학습은 TRAIN_SELECTIONS 만.
# r_s=16 이라 all_effective 는 k=16, energy99 는 k≈15~16 으로 거의 같아질 가능성이 크다.
# report.json 의 cumulative_energy 를 본 뒤 energy99 를 학습에 추가할지 결정할 것.
SELECTION_LIST=(
  "all_effective:"
  "topk8:--adapter_subspace_top_k 8"
  "topk4:--adapter_subspace_top_k 4"
  "topk2:--adapter_subspace_top_k 2"
  "energy90:--adapter_subspace_energy 0.90"
  "energy99:--adapter_subspace_energy 0.99"
)
# 2026-07-25 Stage 1 스펙트럼 확인 후 확정 (topk8 → energy90 교체).
#   all_effective (k=16,   100%)  헤드라인. A_s 행공간 전체에 직교 = 최대 제약.
#                                 그래도 4096 중 16 차원(0.4%)뿐이라 capacity 비용이 사실상 0.
#   topk4         (k=4,   89.2%)  고정·균일 예산 대조군.
#   energy90      (k~4.5, 91.5%)  topk4 와 예산이 맞춰진 적응적 대조군. 층별 k 가 1~12 로
#                                 달라지므로(early 5.7 / late 3.5, attn_k 5.8 / attn_v 3.0)
#                                 "예산 크기가 아니라 배분 방식이 중요한가"를 깨끗하게 묻는다.
# 제외: topk8(95.1%)·energy99(99.2%) 는 all_effective(100%) 와 차이가 작아 정보량이 적다.
# 하한 탐침이 필요하면 topk2(81.7%) 를 4번째로 추가.
TRAIN_SELECTIONS=(all_effective topk4 energy90)

# 1 = Stage 1(Q_S 추출)까지만 하고 종료. 스펙트럼을 본 뒤 TRAIN_SELECTIONS 를 정하기 위함.
# env 로도 덮어쓸 수 있다:  STOP_AFTER_STAGE1=1 bash scripts/run_adapter_subspace_lora.sh
STOP_AFTER_STAGE1=${STOP_AFTER_STAGE1:-0}

OUT_ROOT="outputs/adapter_subspace_lora"
SAFETY_ADAPTER_DIR="${OUT_ROOT}/safety_adapter"
SAFETY_MERGED_DIR="${OUT_ROOT}/safety_merged"
SUBSPACE_ROOT="${OUT_ROOT}/subspaces"
RUN_BASELINE=1                     # Stage 3: 동일 시작점 plain LoRA 통제군

# ── HF push ──
# 토큰 확인은 `hf auth whoami` 로 할 것 (`hf auth login` 은 무효 토큰에도 "Already logged in").
# push 실패는 non-fatal — merged 모델은 이미 로컬에 저장되므로 나중에 재업로드하면 된다.
PUSH=1
HF_NS=kmseong
HF_SAFETY_ADAPTER_REPO="${HF_NS}/llama2_7b-chat-Safety-LoRA-r${SAFETY_LORA_R}-lr${SAFETY_LR}-adapter"
HF_SAFETY_MERGED_REPO="${HF_NS}/llama2_7b-chat-Safety-LoRA-r${SAFETY_LORA_R}-lr${SAFETY_LR}"

mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "${OUT_ROOT}/git_commit.txt" 2>/dev/null || true

# 로컬 폴더를 HF 에 업로드. 실패해도 파이프라인을 죽이지 않는다(로컬 산출물은 이미 안전).
# 중복 업로드 방지용 마커: <dir>/.pushed_<repo 이름>
push_folder () {
  local local_dir=$1 repo_id=$2 kind=$3
  if [ "$PUSH" != "1" ]; then return 0; fi
  local marker="${local_dir}/.pushed_$(echo "$repo_id" | tr '/' '_')"
  if [ -f "$marker" ]; then
    echo "    skip push — 이미 업로드됨: $repo_id"; return 0
  fi
  echo "    pushing $kind → https://huggingface.co/$repo_id"
  if $HBPY - "$local_dir" "$repo_id" <<'PYEOF'
import sys
from huggingface_hub import HfApi
local_dir, repo_id = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model",
                  ignore_patterns=[".pushed_*", "trainer/*", "*.lock"])
print("uploaded:", repo_id)
PYEOF
  then
    touch "$marker"
  else
    echo "    ⚠️ PUSH_FAILED repo=$repo_id local=$local_dir — 로컬 산출물은 그대로 남아 있음."
  fi
}

echo "════════════════════════════════════════════════════════════════"
echo " adapter_subspace_lora pipeline   ts=${TS}"
echo "   base model        : $BASE_MODEL"
echo "   target modules    : $TARGET_MODULES"
echo "   safety LoRA       : r=$SAFETY_LORA_R alpha=$SAFETY_LORA_ALPHA lr=$SAFETY_LR"
echo "   downstream LoRA   : r=$LORA_R alpha=$LORA_ALPHA lrs=${LR_LIST[*]} epochs=$EPOCHS"
echo "   adapter mode      : $SAFETY_ADAPTER_MODE   lora dtype=$LORA_PARAM_DTYPE"
echo "   subspaces built   : ${#SELECTION_LIST[@]}   trained: ${TRAIN_SELECTIONS[*]}"
echo "════════════════════════════════════════════════════════════════"

# ═══════════ Stage 0: safety LoRA tuning ═══════════
if [ -f "${SAFETY_ADAPTER_DIR}/adapter_model.safetensors" ] || \
   [ -f "${SAFETY_ADAPTER_DIR}/adapter_model.bin" ]; then
  echo "[0] skip — safety adapter 이미 존재: $SAFETY_ADAPTER_DIR"
else
  echo "[0] safety LoRA tuning → $SAFETY_ADAPTER_DIR"
  $HBPY models/phase0_SSFT.py "$SAFETY_DATA" \
    --model_name "$BASE_MODEL" \
    --lr "$SAFETY_LR" \
    --output_dir "$SAFETY_ADAPTER_DIR" \
    --lora --lora_r "$SAFETY_LORA_R" --lora_alpha "$SAFETY_LORA_ALPHA" \
    --lora_target_modules $LORA_TARGETS_SPACED \
    --no_wandb
fi
push_folder "$SAFETY_ADAPTER_DIR" "$HF_SAFETY_ADAPTER_REPO" "safety LoRA adapter"

# ═══════════ Stage 0.5: safety 모델 dense 병합 ═══════════
# baseline(Stage 3) 의 시작점이자 "downstream 이전" safety 평가 기준점.
if [ -f "${SAFETY_MERGED_DIR}/config.json" ]; then
  echo "[0.5] skip — merged safety model 이미 존재: $SAFETY_MERGED_DIR"
else
  echo "[0.5] merging safety adapter → $SAFETY_MERGED_DIR"
  $HBPY - "$BASE_MODEL" "$SAFETY_ADAPTER_DIR" "$SAFETY_MERGED_DIR" <<'PYEOF'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
tok = AutoTokenizer.from_pretrained(base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.save_pretrained(out)
print("merged →", out)
PYEOF
fi
push_folder "$SAFETY_MERGED_DIR" "$HF_SAFETY_MERGED_REPO" "merged safety model"

# ═══════════ Stage 1: 보호 subspace Q_S 추출 ═══════════
for entry in "${SELECTION_LIST[@]}"; do
  tag="${entry%%:*}"; sel_args="${entry#*:}"
  sdir="${SUBSPACE_ROOT}/${tag}"
  if [ -f "${sdir}/report.json" ]; then
    echo "[1:${tag}] skip — subspace artifact 이미 존재"
    continue
  fi
  echo "[1:${tag}] building Q_S  ${sel_args:-(all-effective)}"
  # shellcheck disable=SC2086
  $HBPY build_adapter_subspace.py \
    --safety_adapter_path "$SAFETY_ADAPTER_DIR" \
    --out_dir "$sdir" \
    --target_modules "$TARGET_MODULES" \
    --target_layers all \
    --verify_dense_layers 2 \
    $sel_args
done

# ═══════════ Stage 1 결과 요약 ═══════════
# 각 선택 모드가 층마다 실제로 몇 개 방향(k)을 보호하는지 / 그때 봉쇄되는 safety 업데이트
# 에너지 비율이 얼마인지. TRAIN_SELECTIONS 를 정하는 근거가 된다.
echo ""
echo "════════════════ Stage 1: protected subspace 요약 ════════════════"
$HBPY - "$SUBSPACE_ROOT" <<'PYEOF'
import json, os, sys
root = sys.argv[1]
print(f"{'selection':<16}{'k min/mean/max':<20}{'energy@k mean':<15}{'r_eff':<8}{'modules':<8}")
print("-" * 70)
for tag in sorted(os.listdir(root)):
    rp = os.path.join(root, tag, "report.json")
    if not os.path.isfile(rp):
        continue
    rep = json.load(open(rp))
    L = rep["layers"]
    ks = [r["protected_k"] for r in L]
    en = [r["energy_captured_by_k"] for r in L]
    re_ = {r["r_effective"] for r in L}
    print(f"{tag:<16}{min(ks)}/{sum(ks)/len(ks):.1f}/{max(ks):<12}"
          f"{sum(en)/len(en):<15.4f}{str(sorted(re_)):<8}{len(L):<8}")
# 대표 층의 누적 에너지 곡선 — k 를 어디서 끊을지 판단용
any_tag = next(t for t in sorted(os.listdir(root))
               if os.path.isfile(os.path.join(root, t, "report.json")))
rep = json.load(open(os.path.join(root, any_tag, "report.json")))
print("\n누적 에너지 곡선 (대표 층, k=1..8):")
for r in rep["layers"]:
    if r["layer"] in (0, 15, 31) and r["layer_type"] in ("attn_q", "ffn_down"):
        ce = ["%.3f" % x for x in r["cumulative_energy"][:8]]
        print(f"  L{r['layer']:>2} {r['layer_type']:<9} {' '.join(ce)}")
PYEOF
echo "═══════════════════════════════════════════════════════════════════"

if [ "$STOP_AFTER_STAGE1" = "1" ]; then
  echo ""
  echo "STOP_AFTER_STAGE1=1 → Stage 2/3 를 건너뛰고 종료합니다."
  echo "  subspace 진단: ${SUBSPACE_ROOT}/*/report.json"
  echo "  이어서 학습하려면 TRAIN_SELECTIONS 를 정한 뒤 STOP_AFTER_STAGE1=0 으로 재제출"
  echo "  (Stage 0/0.5/1 은 완료 표시가 있어 건너뜁니다)"
  exit 0
fi

# ═══════════ Stage 2: downstream 학습 ═══════════
run_one () {
  local tag=$1 lr=$2
  local sdir="${SUBSPACE_ROOT}/${tag}"
  local odir="${OUT_ROOT}/runs/${tag}/lr_${lr}"
  local repo="${HF_NS}/llama2_7b-chat-gsm8k-adaptersubspace-${tag}-r${LORA_R}-lr${lr}"

  if [ -f "${odir}/summary.json" ]; then
    echo "[2:${tag}:lr${lr}] skip — 이미 완료"
    return
  fi
  echo "──────────────────────────────────────────────────────────────"
  echo "[2:${tag}:lr${lr}] adapter_subspace_lora"
  mkdir -p "$odir"

  local push_args=()
  if [ "$PUSH" = "1" ]; then push_args=(--push_to_hub --hf_repo_id "$repo"); fi

  $HBPY finetune_gsm8k_lora.py \
    --method adapter_subspace_lora \
    --model_name "$BASE_MODEL" \
    --safety_adapter_path "$SAFETY_ADAPTER_DIR" \
    --adapter_subspace_dir "$sdir" \
    --safety_adapter_mode "$SAFETY_ADAPTER_MODE" \
    --lora_param_dtype "$LORA_PARAM_DTYPE" \
    --verify_every_steps "$VERIFY_EVERY" \
    --output_dir "$odir" \
    --target_modules "$TARGET_MODULES" \
    --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --learning_rate "$lr" --epochs "$EPOCHS" \
    --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_length "$MAXLEN" --seed "$SEED" --dtype bfloat16 \
    "${push_args[@]}"
}

for tag in "${TRAIN_SELECTIONS[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    run_one "$tag" "$lr"
  done
done

# ═══════════ Stage 3: 통제군 (동일 시작점 plain LoRA) ═══════════
# adapter_subspace_lora 의 올바른 대조군은 "같은 W_safe 에서 출발한 제약 없는 LoRA" 이다.
# (기존 run_lora_comparison.sh 의 baseline 은 full-FT safety 모델에서 출발하므로 다르다.)
if [ "$RUN_BASELINE" = "1" ]; then
  for lr in "${LR_LIST[@]}"; do
    odir="${OUT_ROOT}/runs/baseline_lora/lr_${lr}"
    if [ -f "${odir}/summary.json" ]; then
      echo "[3:lr${lr}] skip — baseline 이미 완료"; continue
    fi
    echo "[3:lr${lr}] baseline plain LoRA (same start point)"
    mkdir -p "$odir"
    push_args=()
    if [ "$PUSH" = "1" ]; then
      push_args=(--push_to_hub --hf_repo_id
                 "${HF_NS}/llama2_7b-chat-gsm8k-lora-safetylora-r${LORA_R}-lr${lr}")
    fi
    $HBPY finetune_gsm8k_lora.py \
      --method lora \
      --model_name "$SAFETY_MERGED_DIR" \
      --output_dir "$odir" \
      --target_modules "$TARGET_MODULES" \
      --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
      --learning_rate "$lr" --epochs "$EPOCHS" \
      --batch_size "$BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
      --max_length "$MAXLEN" --seed "$SEED" --dtype bfloat16 \
      "${push_args[@]}"
  done
fi

# ═══════════ 요약 ═══════════
echo ""
echo "════════════════════ summary ════════════════════"
printf "%-28s %-10s %-12s %-12s %-10s\n" run lr constr_A constr_delta dW_norm
for f in $(find "${OUT_ROOT}/runs" -name summary.json | sort); do
  $HBPY - "$f" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
s = json.load(open(p))
run = os.path.basename(os.path.dirname(os.path.dirname(p)))
lr = os.path.basename(os.path.dirname(p)).replace("lr_", "")
v = s.get("adapter_subspace_verify") or {}
def g(k, f="max"):
    d = v.get(k)
    return f"{d[f]:.2e}" if d else "-"
print(f"{run:<28} {lr:<10} {g('constraint_A'):<12} {g('constraint_delta'):<12} {g('delta_norm','mean'):<10}")
PYEOF
done
echo ""
if [ "$PUSH" = "1" ]; then
  echo "HF repos:"
  echo "  safety adapter : https://huggingface.co/${HF_SAFETY_ADAPTER_REPO}"
  echo "  safety merged  : https://huggingface.co/${HF_SAFETY_MERGED_REPO}"
  for tag in "${TRAIN_SELECTIONS[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      echo "  ${tag} lr${lr} : https://huggingface.co/${HF_NS}/llama2_7b-chat-gsm8k-adaptersubspace-${tag}-r${LORA_R}-lr${lr}"
    done
  done
  if [ "$RUN_BASELINE" = "1" ]; then
    for lr in "${LR_LIST[@]}"; do
      echo "  baseline lr${lr}: https://huggingface.co/${HF_NS}/llama2_7b-chat-gsm8k-lora-safetylora-r${LORA_R}-lr${lr}"
    done
  fi
  echo ""
fi
echo "merged models: ${OUT_ROOT}/runs/*/lr_*/merged_model"
echo "제약 검증 상세: ${OUT_ROOT}/runs/*/lr_*/subspace_verification.json"
echo "subspace 진단  : ${SUBSPACE_ROOT}/*/report.json"
echo "═════════════════════════════════════════════════"
