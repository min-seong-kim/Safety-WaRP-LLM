"""
sn_tune package  —  WaRP-SN-Tune

핵심 컴포넌트 (SN-Tune):
    module.py  — LinearSNWaRP 모듈 (C = W @ U 재매개변수화)
    detect.py  — gradient 기반 safety coordinate 검출
    run.py     — 전체 파이프라인 (Convert → Detect → Tune → Restore)

구 WaRP-SN 변형 (같은 계열, models/warp_modules.py 의 LinearWaRP 사용):
    warp_sn_detection.py                  — safety neuron 검출
    warp_sn_tune.py                       — safety 좌표만 학습
    run_warp_sn_pipeline.py               — Rotation → Detection → SN-Tune
                                            (scripts/run_warp_sn.sh)
    finetune_downstream_freeze_warp_sn.py — SN freeze 상태로 downstream FT
                                            (scripts/run_downstream_freeze_warp_sn.sh)

  두 진입점은 스스로 저장소 루트를 sys.path 에 넣으므로 저장소 루트에서
  `python sn_tune/run_warp_sn_pipeline.py ...` 로 실행하면 된다.
  태스크별 SN fine-tuner 는 각 eval 하네스에 남아 있다
  (mbpp_eval/finetune_mbpp_freeze_sn.py, mmlu_eval/finetune_mmlu_freeze_sn.py).

사용 예:
    python -m sn_tune.run \\
        --model_name meta-llama/Llama-2-7b-chat-hf \\
        --basis_dir  ./checkpoints/phase1_XXXXXXXX/basis \\
        --dataset_file ./data/circuit_breakers_train.json \\
        --output_dir   ./warp_sn_output
"""

from .module import (
    LinearSNWaRP,
    LAYER_TYPE_MAP,
    convert_to_sn_warp,
    restore_to_linear,
    get_proj,
    set_proj,
)
from .detect import (
    accumulate_grad_scores,
    select_top_coords,
    apply_coeff_gradient_masks,
    detect_with_forward_scores,
)

__all__ = [
    "LinearSNWaRP",
    "LAYER_TYPE_MAP",
    "convert_to_sn_warp",
    "restore_to_linear",
    "get_proj",
    "set_proj",
    "accumulate_grad_scores",
    "select_top_coords",
    "apply_coeff_gradient_masks",
    "detect_with_forward_scores",
]
