"""
sn_tune package — SN-Tune / RSN-Tune 과 그 WaRP 공간 버전.

네 갈래가 들어 있다. 앞의 둘은 **원공간**(원래 weight 공간)에서, 뒤의 둘은 **WaRP
재매개변수화 공간**에서 같은 일을 한다.

    SN-Tune        safety 뉴런만 학습                         원공간
    RSN-Tune       critical(= safety \\ utility) 뉴런만 학습    원공간
    WSR-SN-Tune    safety '열'(basis 방향)만 학습              WaRP 공간
    WSR-RSN-Tune   critical '열'만 학습                        WaRP 공간

이 디렉토리 하나로 전부 돌아간다. 외부 `Safety-Neuron/neuron_detection` 는 더 이상
필요 없다 (2026-09-22 에 들여옴).

━━ 원공간 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    detect_original.py     safety/utility 뉴런 검출  ⚠️ **패치된 transformers 필요**
    critical_neurons.py    critical = safety \\ utility
    sn_tune_original.py    (R)SN-Tune — 해당 뉴런만 safety 데이터로 학습
    finetune_freeze_sn.py  downstream FT — 해당 뉴런을 **얼리고** 나머지 학습

━━ WaRP 공간 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    warp_sn_detection.py                  basis_coeff 의 safety '열' 검출 (패치 불필요)
    warp_sn_tune.py                       그 열만 학습
    run_warp_sn_pipeline.py               검출 + 학습 파이프라인
    finetune_downstream_freeze_warp_sn.py downstream FT — 그 열을 얼린다

━━ 공용 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    neuron_file.py         5줄 뉴런 파일 read/write + 비율 계산 (양쪽 공용 포맷)
    build_utility_corpus.py  wikipedia → JSON (WaRP 쪽 utility 검출 입력)
    verify_patch.py        패치 적용 여부 검사
    setup_hb_sn.sh         hb 복제 → hb_sn + 패치 설치
    transformers_patch/    패치된 modeling_{llama,gemma2,qwen2}.py

━━ 별개 계열 (LinearSNWaRP, C = W @ U) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    module.py / detect.py / run.py   — `python -m sn_tune.run`

드라이버
    scripts/run_sn_rsn_gsm8k.sh       원공간 SN/RSN 전체 (7B·13B × GSM8K)
    scripts/run_wsr_sn_rsn_gsm8k.sh   WaRP 공간 WSR-SN/WSR-RSN 전체

⚠️ **환경이 둘로 갈린다.** 원공간 검출만 패치된 `hb_sn` 이 필요하고, 나머지 전부
(원공간 학습, WaRP 공간 전체)는 정품 `hb` 에서 돌아야 한다. 자세한 것은 README.md.
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
from .neuron_file import (
    MODULE_ORDER,
    assert_nonempty,
    count_neurons,
    load_neuron_file,
    save_neuron_file,
    summarize,
)

__all__ = [
    # module.py
    "LinearSNWaRP",
    "LAYER_TYPE_MAP",
    "convert_to_sn_warp",
    "restore_to_linear",
    "get_proj",
    "set_proj",
    # detect.py
    "accumulate_grad_scores",
    "select_top_coords",
    "apply_coeff_gradient_masks",
    "detect_with_forward_scores",
    # neuron_file.py
    "MODULE_ORDER",
    "assert_nonempty",
    "count_neurons",
    "load_neuron_file",
    "save_neuron_file",
    "summarize",
]
