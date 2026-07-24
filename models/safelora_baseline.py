"""바닐라 Safe LoRA (NeurIPS'24) baseline projection.

공식 구현 /home/gokms0509/SafeLoRA/model.py 의 알고리즘을 그대로 포팅하되,
(1) dtype 불일치(bf16 LoRA ↔ fp32 projector)로 인한 crash 방지,
(2) base/aligned 두 full model 을 projector 계산 직후 즉시 해제해 메모리 절감,
을 추가했다. 학습은 표준 LoRA 와 동일(finetune_gsm8k_lora.py --method safe_lora)하고,
이 함수는 학습이 끝난 **PEFT 모델(merge 전)** 에 사후(post-hoc)로 projection 을 적용한다.

알고리즘 (공식 model.py 와 동일):
    각 target module 의 weight 에 대해
        V = W_aligned − W_base                  (alignment delta, out×in)
        C = (V Vᵀ) / ‖V‖_F                      (output-space projector, out×out)
    각 LoRA 레이어에 대해
        cos = cosine( (C·B)·A ,  B·A )          (투영 후 업데이트가 원본과 얼마나 다른가)
        if cos ≤ threshold:  B ← C·B            (safety subspace 로 투영)
        else:                B 그대로
    select_layers_type:
        'threshold' → 고정 임계값
        'number'    → cos 가 가장 낮은(=가장 덜 정렬된) num_proj_layers 개만 투영

주의: v 리스트 순서(= base model 파라미터 순회 순서로 필터)와 PEFT lora_B 순회 순서가
positional 로 1:1 대응해야 한다. 표준 LLaMA + 표준 target_modules 에서 성립(공식과 동일 가정).
"""
import gc

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def _build_projectors(base_path, aligned_path, target_modules, compute_device, load_dtype, logger):
    """공식 get_aligned_matrix 포팅: base/aligned 를 로드해 module 별 projector C 리스트 생성.

    반환: List[Tensor(out, out)] (fp32, cpu) — base model 파라미터 순회 순서.
    """
    logger.info(f"[SafeLoRA] loading base   : {base_path} (dtype={load_dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, return_dict=True, low_cpu_mem_usage=True, torch_dtype=load_dtype, device_map="cpu")
    logger.info(f"[SafeLoRA] loading aligned: {aligned_path} (dtype={load_dtype})")
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_path, return_dict=True, low_cpu_mem_usage=True, torch_dtype=load_dtype, device_map="cpu")

    v = []
    for (b_name, b_param), (a_name, a_param) in zip(base_model.named_parameters(),
                                                    aligned_model.named_parameters()):
        if any(m in a_name for m in target_modules):
            assert b_param.shape == a_param.shape, (
                f"base/aligned weight shape mismatch: {b_name} {tuple(b_param.shape)} "
                f"vs {a_name} {tuple(a_param.shape)}")
            # fp32 로 upcast 후 projector 계산 (수치 안정 + dtype 일관)
            vec = (a_param.detach() - b_param.detach()).to(compute_device, dtype=torch.float32)
            C = torch.mm(vec, vec.t()) / torch.norm(vec)        # (V Vᵀ)/‖V‖_F  (공식과 동일)
            v.append(C.detach().cpu())
    logger.info(f"[SafeLoRA] built {len(v)} projectors")

    del base_model, aligned_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return v


@torch.no_grad()
def _project(peft_model, v, r, thrs_cos, compute_device, logger, show_info=False, apply=True):
    """공식 projected_weighted 포팅. cos ≤ thrs_cos 인 레이어의 lora_B 를 C·B 로 투영.

    원본 구현은 copy.deepcopy(peft_model) 로 투영 전 가중치를 참조하지만, 7B 모델에서
    GPU 상 full deepcopy 는 +14GB → 24GB 카드 OOM. 여기서는 **LoRA 가중치만 스냅샷**해
    동일 의미를 유지하면서 메모리를 아낀다(투영 결정/값은 원본과 비트 단위로 동일).

    apply=False 면 cos 만 계산하고 가중치를 바꾸지 않는다('number' 모드 1차 패스용).
    반환: (n_projected, cos_total)
    """
    # 투영 전 LoRA 가중치 스냅샷 (작음: 모듈당 r×in + out×r)
    orig = {name: p.detach().clone()
            for name, p in peft_model.named_parameters() if 'lora' in name}
    idx = 0
    i = 0
    cos_total = []
    A_ori = None
    for name, param in peft_model.named_parameters():
        if 'lora' not in name:
            continue
        param_ori = orig[name]
        if param.shape[0] == r:
            # lora_A (r × in): 원본을 기억해 두었다가 다음 lora_B 와 함께 사용
            A_ori = param_ori.to(compute_device, dtype=torch.float32)
            continue
        # lora_B (out × r)
        P = v[idx].to(compute_device, dtype=torch.float32)
        B_ori = param_ori.to(compute_device, dtype=torch.float32)
        W = torch.mm(P, B_ori)                      # C·B
        fW = torch.mm(W, A_ori)                     # (C·B)·A
        ori = torch.mm(B_ori, A_ori)                # B·A
        cos = round(torch.nn.functional.cosine_similarity(
            fW.reshape(1, -1), ori.reshape(1, -1)).item(), 5)
        cos_total.append(cos)
        if apply:
            if cos <= thrs_cos:
                i += 1
                param.data = W.to(dtype=param.dtype, device=param.device)
            else:
                param.data = B_ori.to(dtype=param.dtype, device=param.device)
        idx += 1
    if show_info:
        logger.info(f"[SafeLoRA] {i}/{len(cos_total)} layers projected "
                    f"(cos threshold={thrs_cos}); cos range "
                    f"[{min(cos_total):.4f}, {max(cos_total):.4f}]")
    del orig
    gc.collect()
    return i, cos_total


@torch.no_grad()
def apply_safelora(peft_model, base_path, aligned_path, target_modules, r,
                   select_layers_type="threshold", threshold=0.35, num_proj_layers=10,
                   compute_device="cuda", load_dtype=torch.float32, logger=None):
    """학습된 PEFT LoRA 모델에 Safe LoRA projection 을 in-place 적용.

    Args:
        peft_model: get_peft_model 로 감싼, 학습 완료된 모델 (merge 전).
        base_path/aligned_path: alignment delta V = W_aligned − W_base 용 두 모델.
        target_modules: LoRA target module 이름들 (예: ["q_proj","k_proj",...]).
        select_layers_type: 'threshold' | 'number'.
        threshold: 'threshold' 모드의 코사인 임계값.
        num_proj_layers: 'number' 모드에서 투영할 레이어 수.
    Returns:
        dict: 투영 통계(로그/요약용).
    """
    assert logger is not None
    v = _build_projectors(base_path, aligned_path, target_modules, compute_device, load_dtype, logger)

    if select_layers_type == "threshold":
        n_proj, cos_total = _project(peft_model, v, r, threshold, compute_device, logger, show_info=True)
        used_threshold = threshold
    elif select_layers_type == "number":
        # 1차: cos 만 수집(가중치 미변경) → num_proj_layers 번째로 낮은 cos 를 임계값으로
        _, cos_total = _project(peft_model, v, r, -1.0, compute_device, logger, show_info=False, apply=False)
        thrs = float(sorted(cos_total)[:num_proj_layers][-1])
        n_proj, cos_total = _project(peft_model, v, r, thrs, compute_device, logger, show_info=True)
        used_threshold = thrs
    else:
        raise ValueError("select_layers_type must be 'threshold' or 'number'.")

    del v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "select_layers_type": select_layers_type,
        "used_threshold": used_threshold,
        "num_layers_projected": n_proj,
        "num_lora_layers": len(cos_total),
        "cos_min": min(cos_total) if cos_total else None,
        "cos_max": max(cos_total) if cos_total else None,
    }
