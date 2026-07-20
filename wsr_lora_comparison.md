# WSR-LoRA 비교 실험 지시서 (rebuttal용, 개정판)

> 개정 요지 (이전판 대비):
> 1. **WSR-LoRA 정의 교체**: 방향(열) 투영 `BA(I−U_safeU_safeᵀ)` → **reparameterized 공간의 element-wise mask**.
>    이전판은 논문이 "우리가 더 일반적"이라고 증명한 column-wise null-space projection과 동일해져
>    element-wise 일반성 주장과 충돌했음. 개정판은 논문 eq.(12) 철학과 일치.
> 2. **평가 스테이지 추가**: 외부 harness(HarmBench ASR + lm-eval GSM8K)를 HF model ID로 연동. 표를 만드는 유일한 부분.
> 3. **스코프 축소**: 1 model / 1 keep_ratio / LR 2개 / 3 method = 6 run. test 8→3. model-card·reload 검증 제거.
> 4. **기존 자산 재사용**: `models/warp_modules.py`, Phase1 basis, Phase2 per-layer mask를 그대로 사용. 새로 짜는 건 LoRA 얹는 부분과 runner뿐.

---

## 0. 왜 이 실험인가 (rebuttal 논리)

논문 결론이 직접 *"Future work may further combine this perspective with low-rank adaptation"* 라고 적었으므로
리뷰어의 LoRA 질문은 사실상 확실하다. 이 실험은 **동일한 trainable-parameter budget(같은 LoRA B,A)** 을
어떻게 쓰느냐로 세 방법을 비교하여, 논문의 Table 5 ablation(Full FT / original-space mask / WSR-Tune)을
LoRA 영역에서 재현한다.

| 방법 | update가 사는 곳 | 대응하는 논문 열 |
|------|------------------|------------------|
| Standard LoRA | 원본 공간 전방향 rank-r | Full param FT (제약 없음) |
| Original-space Projected LoRA | 원본 공간, safety 입력좌표 제거 | Original-space + important mask FT |
| **WSR-LoRA (element-wise)** | reparameterized 공간, safety 좌표 element freeze | **WSR-Tune** |

**핵심 서사(반드시 이대로 프레이밍):** 세 방법은 rank가 같지 않다. element-wise WSR-LoRA는 rank-r 파라미터로
full-rank 증분을 만든다. 이것이 confound가 아니라 **element-wise 재매개변수화의 이점**이다 —
같은 파라미터 예산인데 안전을 더 잘 지킨다가 곧 논문 thesis. rank 차이를 숨기지 말고 이렇게 설명한다.

---

## 1. 세 방법의 정의

각 target linear layer의 frozen safety-tuned weight를 `W₀ ∈ ℝ^{m×n}`,
LoRA factor를 `A ∈ ℝ^{r×n}`, `B ∈ ℝ^{m×r}`, scaling `s = α/r` 라 한다.
PEFT의 `lora_A.weight`=A, `lora_B.weight`=B. 세 방법 모두 `B`는 0으로 초기화되어 시작 시 ΔW=0 (시작=safety 모델).

### 1.1 Standard LoRA  `--method lora`
```
W = W₀ + s·B A
```
- PEFT `LoraConfig`/`get_peft_model`. 원본 파라미터 freeze, `lora_A/lora_B`만 학습.
- safety basis / importance 미사용.

### 1.2 Original-space Projected LoRA  `--method original_projected_lora`
원본 공간에서 safety-critical **입력 좌표**를 보호한다.
```
W = W₀ + s·B A (I − EEᵀ),   즉 A[:, safe_cols] = 0
```
- safety importance(원본 공간): `G_l^orig = Σ_{x∈D_safe} |∂L_safe/∂W_l|` → 열 점수 `s_{l,j} = ‖G_l^orig[:,j]‖₂`.
  → **기존 `models/phase2_importance_original_space.py`** 로 얻은 원본공간 element importance를 열 L2로 집계.
- 각 layer 상위 `k_l = max(1, round(ρ·n_l))` 개 canonical 입력좌표를 보호(A의 해당 열을 0).
- canonical basis이므로 dense projector 없이 `A[:, safe_indices] = 0`.

### 1.3 WSR-LoRA (element-wise)  `--method wsr_lora`  ★ 이번 개정 핵심
safety-conditioned basis `U_l` 로 재매개변수화한 공간에서 **element 단위**로 safety 좌표를 freeze한다.
```
W̃_eff = (W₀ U_l) + (1 − M_l) ∘ ( s · B A )         # W₀U_l 은 frozen, (1−M) 위치만 증분 허용
W       = W̃_eff  U_lᵀ                                # dense fold (adapter 아님)
```
- `M_l ∈ {0,1}^{m×n}` = **기존 Phase2 per-layer mask** (reparameterized 공간 element importance
  `G_l^WSR = Σ|∂L_safe/∂W̃_l|` 상위 `keep_ratio` = 1 = freeze).
  → **basis, mask 모두 기존 파이프라인(Phase1 + Phase2 per-layer)으로 생성, 재사용.**
- `B,A`는 basis 공간에서 학습. 시작 시 B=0 → W̃_eff = W₀U_l → W = W₀ (정확히 safety 모델).
- **저장은 adapter merge가 아니라 dense fold**: 학습 후 `W = W̃_eff.detach() @ U_lᵀ` 계산 → `nn.Linear`로 교체
  (`warp_modules.restore_weight`/`restore_to_linear`와 동일 방식). 결과는 `from_pretrained`로 바로 로드되는 full model.
- ⚠️ **주의**: singular value 큰 앞쪽 basis column을 고르는 게 아니라, 반드시 safety **gradient importance**(Phase2)로
  element를 고른다. Original-space와 WSR이 layer별로 보호하는 자유도 수를 맞추려면
  `keep_ratio`(WSR element 비율)와 `direction_keep_ratio`(orig 열 비율)를 개념상 구분해 기록한다(동일 물리량 아님).

### 1.4 (선택) 4번째 열: Safe LoRA  `--method safe_lora`
기존 `models/safe_lora_basis_rotation.py`(output-space `B` 투영)를 baseline 열로 포함하면 "LoRA 계열 SOTA 대비"까지 커버.
시간이 없으면 생략.

---

## 2. 재사용할 기존 코드 (새로 짜지 말 것)

```text
models/warp_modules.py                       # LinearWaRP: basis_coeff/UT_forward/coeff_mask/restore_* (그대로 확장)
models/phase1_basis.py                       # U_l 생성 (Gram ΦᵀΦ → SVD). 기존 basis dir 있으면 재사용
models/phase2_importance_per_layer.py        # WSR element importance → per-layer mask M_l
models/phase2_importance_original_space.py   # 원본공간 importance (original_projected_lora의 열 점수 재료)
train.py                                      # --phase 1/2 로 basis·mask 생성 (LAYER_TYPE·target_layers 불변식 준수)
scripts/run_all_phases_integrated.sh         # basis/mask 생성 관례 (LAYER_TYPE, keep_ratio) 참고
```

**불변식**: `--layer_type`, `--target_layers` 는 Phase1/2 및 WSR-LoRA 학습에서 동일해야 한다.

---

## 3. 공통 실험 설정

### 3.1 모델·데이터
```text
model_name          = kmseong/llama2_7b-chat-Safety-FT-lr5e-5   # 세 방법 동일 시작점
downstream          = GSM8K train split
safety_data         = ./data/circuit_breakers_train.json  (basis/importance 전용, GSM8K batch에 섞지 않음)
max_length          = 1024
```
GSM8K 전처리는 기존 downstream 코드와 동일: Llama-2 chat template, prompt/pad token label = −100,
answer token만 CLM loss. 세 방법이 **동일 data 순서·seed**.

### 3.2 LoRA / layer 설정
```text
target_modules  = q_proj, k_proj, v_proj, up_proj, down_proj     # o_proj/gate_proj/lm_head/embed/norm 제외
layer_type      = attn_q,attn_k,attn_v,ffn_up,ffn_down           # 위와 1:1 대응 (Phase1/2와 동일)
target_layers   = all
lora_r=16, lora_alpha=32, lora_dropout=0.05, bias=none, task_type=CAUSAL_LM
```
세 방법에서 rank·alpha·dropout·target module·**trainable parameter 수가 완전히 동일**해야 한다(불일치 시 오류).

### 3.3 학습 설정
```text
epochs=3, per_device_batch=2, grad_accum=8 (eff batch 16), optim=AdamW,
scheduler=cosine, warmup_ratio=0.03, weight_decay=0.0, dtype=bfloat16,
gradient_checkpointing=true, seed=42
LR_LIST=(1e-4 2e-4)     # 축소: 2개
KEEP_RATIO=0.1          # WSR element freeze 비율 (= 논문 ρ 관례)
```
미사용: SafeInstr mixing, constrained SFT, non-freeze, SN-Tune, two-mask, QLoRA/4-bit, post-hoc Safe LoRA(선택 열 제외).

---

## 4. 구현 파일

```text
models/lora_wsr_elementwise.py     # ★ 신규: LinearWSRLoRA 모듈 + switch/restore
models/lora_original_projected.py  # 신규: PEFT LoRA + A[:, safe_cols]=0 projector
models/build_lora_safety_artifacts.py  # 신규: Phase2 원본/WSR importance → safe_cols / mask 아티팩트 저장·로드
finetune_gsm8k_lora.py             # ★ 신규(root): 단일 entry point, --method {lora,original_projected_lora,wsr_lora}
scripts/run_lora_comparison.sh     # 신규: basis/mask 확인 → 6 run → merge(dense) → HF push → eval 큐잉
tests/test_lora_wsr.py             # 신규: 핵심 3 test
```

### 4.1 `LinearWSRLoRA` (models/lora_wsr_elementwise.py)
`warp_modules.WaRPModule` 관례를 따르되 basis_coeff는 **frozen buffer**, 학습은 LoRA B,A만.
```python
class LinearWSRLoRA(nn.Module):
    # buffers: weight(W₀), UT_forward(U), coeff_mask(M, bool), scaling(α/r)
    # params : lora_A (r×n), lora_B (m×r)   # B는 0 초기화
    def forward(self, x):
        delta   = self.scaling * (self.lora_B @ self.lora_A)          # (m×n), full-rank 허용
        delta   = torch.where(self.coeff_mask, torch.zeros_like(delta), delta)  # (1−M)∘delta
        eff     = self.basis_coeff + delta                            # basis_coeff=W₀U (frozen buffer)
        weight  = eff @ self.UT_forward.t()                           # = W₀ + [(1−M)∘delta] Uᵀ
        return F.linear(x, weight, self.bias)
```
- `switch_to_wsr_lora(model, basis, masks, layer_types, target_layers, r, alpha)`:
  각 target Linear을 LinearWSRLoRA로 교체, `basis_coeff = W₀@U` (buffer), `coeff_mask=M`, LoRA 초기화.
- `restore_wsr_lora_to_linear(model)`: `weight = (basis_coeff + (1−M)∘(s·BA)).detach() @ Uᵀ` → `nn.Linear` 교체.
- fairness: LoRA B,A shape는 method 1/2의 PEFT LoRA와 동일(같은 target module·r) → trainable count 동일.

### 4.2 `finetune_gsm8k_lora.py` CLI (핵심만)
```text
--method {lora|original_projected_lora|wsr_lora}
--model_name --output_dir --safety_data_path
--basis_dir            (wsr_lora 필수)
--mask_dir             (wsr_lora 필수: Phase2 per-layer mask)
--orig_importance_dir  (original_projected_lora 필수: Phase2 원본공간 importance)
--keep_ratio (wsr) / --direction_keep_ratio (orig)
--lora_r --lora_alpha --lora_dropout --target_modules --layer_type --target_layers
--learning_rate --epochs --batch_size --gradient_accumulation_steps --max_length --seed --dtype
--save_merged_model --push_to_hub --hf_repo_id
```
- projected/wsr method가 필요한 아티팩트 없이 실행되면 **standard LoRA로 fallback 금지, 즉시 오류**.
- original_projected_lora projection 시점: 초기화 직후 / `optimizer.step()` 직후 / 저장 직전 / 종료 직전.
  gradient accumulation micro-step마다 하지 말고 optimizer update 후에만.
- wsr_lora는 forward에서 mask가 사전제약이므로 별도 projection step 불필요(structural).

---

## 5. 학습 → dense 저장 → HF 업로드

각 run 종료 후:
1. (original_projected) 마지막 projection 1회.
2. **dense 모델 생성**:
   - lora / original_projected: PEFT `merge_and_unload()` → full model.
   - wsr_lora: `restore_wsr_lora_to_linear()` → full model.
3. `save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")` + tokenizer 저장.
4. local sanity generation 1건.
5. HF Hub push (아래 네이밍). **두 평가 harness가 HF ID를 입력으로 받으므로 업로드는 필수 경로.**

HF 인증: `HF_TOKEN` 또는 기존 `huggingface-cli login`(토큰 파일 존재 확인됨). 미인증 시 조용히 skip 금지, 오류.
네이밍(기존 `kmseong/…`, `wvnvwn/…` 관례):
```text
kmseong/llama2_7b-chat-gsm8k-lora-r16-lr1e-4
kmseong/llama2_7b-chat-gsm8k-origproj-lora-kr0.1-r16-lr1e-4
kmseong/llama2_7b-chat-gsm8k-wsr-lora-elem-kr0.1-r16-lr1e-4
```

---

## 6. 평가 연동 (표를 만드는 부분) — 외부 harness

두 harness 모두 **HF model ID** 로 모델을 받는다. 업로드한 6개 ID를 등록해 실행.

### 6.1 GSM8K (downstream ↑)
```text
harness : /home/users/minseong/lm-evaluation-harness/eval_models.sh
방법    : model_list=( "<HF ID> ..." ) 에 6개 추가, task_list=( gsm8k ) 활성화 (5-shot)
결과    : gsm8k exact-match acc
```

### 6.2 ASR (safety ↓)
```text
harness : /home/users/minseong/HarmBench/harmbench_eval.sh
등록    : configs/model_configs/models.yaml 에 6개 엔트리 추가
          (기존 gemma2/qwen 엔트리 패턴: model_name_or_path=<HF ID>, dtype bf16,
           max_model_len 4096, gpu_memory_utilization 0.5, num_gpus 1, model_type open_source)
실행    : harmbench_eval.sh 의 MODELS=(...) 배열에 단축명 6개 추가 후 실행
주의    : AutoDAN/PAIR/PAP attack 생성은 매우 비쌈(모델당 수십분~시간+). 시간 부족 시
          Direct + AutoDAN 만으로 축소하고 그 사실을 표에 명시.
결과    : JB ASR (Direct/AutoDAN/PAIR/PAP 평균 또는 축소 집합)
```

### 6.3 최종 표 (rebuttal 산출물)
```text
Method                          | GSM8K acc ↑ | JB ASR ↓
Before FT (safety 모델)         |     -       |   -
Standard LoRA                   |             |
Original-space Projected LoRA   |             |
WSR-LoRA (element-wise)         |             |
```
LR는 downstream 기준으로만 선택(안전 ASR로 LR 고르지 않음). 두 LR 모두 부록에 보존.

---

## 7. 필수 unit test (3개, `tests/test_lora_wsr.py`)

1. **initial output equivalence**: LoRA 초기화 직후 세 방법의 logits가 시작 모델과 수치오차 내 동일(B=0).
2. **WSR mask freeze**: toy `LinearWSRLoRA`에서 B,A에 임의값 주입 후
   `weight − W₀` 의 rotated 표현 `(weight − W₀)@U` 가 mask=1 위치에서 0.
3. **trainable parameter equality**: 세 방법의 trainable param 수 동일(같은 r·target module).

`python -m py_compile` + `pytest -q tests/test_lora_wsr.py` 를 실제로 통과시킨다.

---

## 8. 실행 스크립트 `scripts/run_lora_comparison.sh`

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1        # GPU 1 전용
MODEL="kmseong/llama2_7b-chat-Safety-FT-lr5e-5"
LAYER_TYPE="attn_q,attn_k,attn_v,ffn_up,ffn_down"
KEEP_RATIO=0.1; DIRECTION_KEEP_RATIO=0.1
LORA_R=16; LORA_ALPHA=32; LORA_DROPOUT=0.05
EPOCHS=3; BATCH=2; GRAD_ACCUM=8; MAXLEN=1024; SEED=42
LR_LIST=(1e-4 2e-4)
METHODS=(lora original_projected_lora wsr_lora)
HF_NS=kmseong; PUSH=1
```
순서:
1. env·git commit 기록.
2. Phase1 basis 확인/생성 (`train.py --phase 1`, LAYER_TYPE 동일). 기존 완성 basis 있으면 재사용.
3. Phase2 WSR per-layer mask 생성 (`train.py --phase 2 --perlayer`, keep_ratio 0.1).
4. Phase2 원본공간 importance 생성 (`--original_space_mask`) → original_projected 열 점수 재료.
5. 3 method × 2 LR = 6 run 학습 (각 run stdout/stderr `tee`).
6. dense merge/fold → local reload sanity → HF push.
7. 6개 HF ID를 lm-eval `model_list` + HarmBench `models.yaml`/`MODELS`에 등록(수동 또는 스크립트).
8. GSM8K + ASR 실행 → 표 집계.

파이프라인 중간 실패가 성공으로 처리되지 않게 `set -euo pipefail`. 기존 basis/mask 있으면 재사용.

---

## 9. 공정성 체크 (실행 시작 시 기록)
```text
□ 세 방법 시작 checkpoint 동일
□ GSM8K sample·순서·tokenization 동일
□ target module·rank·alpha·dropout·trainable param 수 동일
□ optimizer·scheduler·LR grid 동일
□ projected/wsr batch에 safety data 미혼합
□ WSR keep_ratio 와 original direction_keep_ratio 는 다른 물리량임을 기록(오해 방지)
```

## 10. 보고 형식
새/수정 파일, 각 역할, 세 방법 수식·코드 위치, 실행 명령, output 구조, test 결과,
trainable param 수, local/merged 경로, HF URL, GSM8K·ASR 결과 표, 미실행/제약 사항, 가정.

## 11. 핵심 원칙
```
Standard LoRA        : ΔW = s·BA
Original Projected   : ΔW = s·BA(I−EEᵀ)                     (원본 공간 열 제거)
WSR-LoRA (element)   : ΔW = [ (1−M)∘(s·BA) ] Uᵀ            (basis 공간 element freeze, dense fold)
```
WSR-LoRA는 basis만 회전한 LoRA가 아니다. Phase2 safety importance로 고른 element를 freeze하고,
나머지 element에서만 basis 공간 증분이 발생한다. 세 방법은 동일 rank·동일 trainable param 수·
동일 downstream data·동일 optimizer·동일 budget을 쓰되, 차이는 **증분이 허용되는 좌표계와 그 제약 방식**이다.
