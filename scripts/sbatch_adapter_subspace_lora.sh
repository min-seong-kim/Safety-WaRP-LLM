#!/bin/bash
#SBATCH -J adapter_subspace_lora
#SBATCH --gres=gpu:1
#SBATCH --output=logs/adapter_subspace_%j.out
#SBATCH --error=logs/adapter_subspace_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition gigabyte_a6000

# 파티션 선택 근거(2026-07-25 sinfolong):
#   gigabyte_a6000 — A6000 48GB, node45/46 에 ~9개 유휴. AllowQos=base_qos,big_qos 라
#   기본 QOS 로 제출 가능(code_test.sh 와 동일).
#   suma_a100 이 더 여유롭지만(node42 에 7개 유휴) AllowQos=a100_qos,a100_low_qos 로
#   별도 QOS 권한이 필요해 "Invalid qos specification" 으로 거부됨. 권한이 있으면
#   --partition suma_a100 --qos a100_low_qos 로 바꾸는 편이 더 빠르다.
#   24GB 급(rtx3090/4090/a5000)은 batch 4 × 1024 학습에 빠듯하므로 피할 것.
#
# 제출:  cd /home/gokms0509/Safety-WaRP-LLM && sbatch scripts/sbatch_adapter_subspace_lora.sh
#
# ⚠️ CUDA_VISIBLE_DEVICES 를 여기서도, 하위 스크립트에서도 설정하지 않는다.
#    --gres=gpu:1 로 할당된 GPU 를 SLURM 이 노출한다.
#
# run_adapter_subspace_lora.sh 는 완료된 stage 를 건너뛰므로, 시간 초과로 죽으면
# 그대로 다시 sbatch 하면 이어서 진행된다.

cd /home/gokms0509/Safety-WaRP-LLM
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate hb

nvidia-smi
echo "SLURM_JOB_ID=$SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# STOP_AFTER_STAGE1=1 → Stage 0(safety LoRA) + 0.5(병합/업로드) + 1(Q_S 추출)까지만.
# 스펙트럼을 보고 TRAIN_SELECTIONS 를 정한 뒤, 이 값을 0 으로 바꿔 재제출하면 이어서 학습한다.
STOP_AFTER_STAGE1=${STOP_AFTER_STAGE1:-1} bash scripts/run_adapter_subspace_lora.sh
