"""
sn_tune package  —  WaRP-SN-Tune

핵심 컴포넌트:
    module.py  — LinearSNWaRP 모듈 (C = W @ U 재매개변수화)
    detect.py  — gradient 기반 safety coordinate 검출
    run.py     — 전체 파이프라인 (Convert → Detect → Tune → Restore)

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
