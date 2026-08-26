# Safety-FT baseline 5종 × 모델 5종 — 실행 및 하이퍼파라미터

논문 Table 2 를 llama2-7b-chat 밖으로 확장하기 위한 러너.
드라이버: `scripts/run_baselines_multimodel.sh`

| | |
|---|---|
| 기법 | LISA, SafeLoRA, AsFT, SaLoRA, SEAL |
| 모델 | llama2-13b-chat, llama3.2-3b-it, llama3.1-8b-it, qwen2.5-7b-it, gemma2-9b-it |
| 안전 데이터 | `circuit_breakers_train.json` (4994) — 전 모델·전 기법 공통 |
| downstream | 모델별로 지정 (아래 레지스트리) |

```bash
bash scripts/run_baselines_multimodel.sh                                  # 전체
MODELS=llama31_8b METHODS="asft safelora" bash scripts/run_baselines_multimodel.sh
DRY_RUN=1 bash scripts/run_baselines_multimodel.sh                        # 명령만 출력
STOP_AFTER_SSFT=1 bash scripts/run_baselines_multimodel.sh                # Stage 0 까지만
PUSH_TO_HUB=1 HF_NAMESPACE=kmseong bash scripts/run_baselines_multimodel.sh
```

완료된 run 은 `summary.json` / `finetune_config.json` / `sft_config.json` 로 판정해 건너뛴다.
중간에 죽어도 그대로 재실행하면 이어서 간다.

---

## 0. 실행 전 준비

### (a) MATH 태스크 JSON — **이미 생성해 뒀다**

```bash
python scripts/prepare_math_task_data.py        # → data/math_train_task_7500.json
```

baseline 러너들은 gsm8k 아니면 `--task_data_path` 로컬 JSON 만 읽는다(WaRP Phase 3 만
`--phase3_dataset math` 로 MATH 를 직접 로드한다). 그래서 MATH 를 JSON 으로 떨어뜨린다.

전처리는 `data/math_task_format.py` **단일 소스**를 쓰고, `models/phase3_extra_learning.py`
의 MATH 로더도 이제 같은 모듈을 import 한다 → **WaRP arm 과 baseline arm 의 학습 텍스트가
바이트 단위로 동일**하다. (arc/medqa 가 `scripts/prepare_qa_task_data.py` 에서 eval
하네스의 프롬프트 빌더를 import 하는 것과 같은 장치.)

타깃 포맷은 `"{rationale}\nFinal Answer: ${answer}$"` (long). `--train_on_mixed_formats`
를 주면 Phase 3 와 동일하게 long 70% / short 20% / minimal 10% 로 섞인다.

### (b) 안전정렬 출발 모델 (Stage 0)

전 기법이 "**이미 안전정렬된 모델**을 downstream 으로 미세조정하면 안전성이 무너지는가"를
보는 실험이므로 출발점이 SSFT 모델이어야 한다. llama2-7b 실험의 출발점이
`kmseong/llama2_7b-chat-Safety-FT-lr5e-5` 였던 것과 같다.

| 모델 키 | base | 안전정렬 출발 모델 | 상태 |
|---|---|---|---|
| `llama2_13b` | `meta-llama/Llama-2-13b-chat-hf` | `wvnvwn/llama-2-13b-chat-hf-SSFT-lr5e-5` | 기존 것 재사용 |
| `llama32_3b` | `meta-llama/Llama-3.2-3B-Instruct` | `kmseong/llama3_2_3b-instruct-SSFT-lr5e-5` | 기존 것 재사용 |
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | `kmseong/Llama-3.1-8B-Instruct-ssft_lr5e-5` | 기존 것 재사용 |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | `wvnvwn/qwen-2.5-7B-Instruct-SSFT-lr5e-5` | 기존 것 재사용 |
| `gemma2_9b` | `google/gemma-2-9b-it` | `wvnvwn/gemma-2-9b-it-ssft-lr3e-5` | 기존 것 재사용 (**lr 3e-5**) |

다섯 모델 모두 안전정렬 모델이 이미 있으므로 **Stage 0 는 전부 skip 된다.**

Stage 0 는 `models/phase0_SSFT.py` (full-param SFT, circuit_breakers 4994, **lr 5e-5**,
3 epochs, batch 4×4, max_len 1024) 를 돌려
`checkpoints/ssft_<모델키>_lr5e-5/` 에 저장한다. `SSFT_LR` 로 바꿀 수 있다.

이미 만든 SSFT 모델이 있으면 환경변수로 주입하면 Stage 0 를 건너뛴다:
```bash
QWEN25_7B_ALIGNED=kmseong/qwen2_5_7b-instruct-Safety-FT-lr5e-5 bash scripts/run_baselines_multimodel.sh
```

---

## 1. matched 예산 — LoRA 4종 공통

LISA / SafeLoRA / AsFT / SaLoRA 는 **같은 학습 예산**에서 돌린다. 기법 간 차이가
"안전 메커니즘" 하나로만 남게 하기 위해서다. 기존 llama2-7b 의
`scripts/run_lisa_safelora_asft_qa.sh` 동작점을 그대로 이식했다.

| 항목 | 값 | 이유 |
|---|---|---|
| LoRA rank `r` | 16 | 기존 전 실험(WSR-LoRA 라인 포함)의 고정 예산 |
| LoRA `alpha` | 32 | 〃 (scaling `s = α/r = 2`) |
| LoRA dropout | 0.05 | 〃 |
| target modules | `q_proj, k_proj, v_proj, up_proj, down_proj` | 〃. WSR-Tune 의 `--layer_type` 5종과 1:1 대응 |
| effective batch | **16** | 모델 크기와 무관하게 고정 (micro × accum = 16) |
| epochs | 3 | 〃 |
| max_length | 1024 | 〃 |
| lr | **3e-4** (`LRS`) | LoRA 기본 동작점. arc/medqa/sst2/agnews 실험과 동일 |
| scheduler | cosine | 〃 |
| warmup_ratio | 0.03 | 〃 |
| weight_decay | 0.0 | 〃 |
| seed | 42 | 〃 |
| dtype | bfloat16 | 〃 |
| gradient checkpointing | on | 13B/9B 를 48GB 안에 넣기 위해 전 모델 공통으로 켬 |

**micro-batch × grad-accum 은 모델별로 다르되 곱은 항상 16** 이다. 메모리 때문에
쪼개는 것뿐이라 결과에는 영향이 없다.

| 모델 키 | micro | accum | downstream (기본) |
|---|---|---|---|
| `llama2_13b` | 2 | 8 | `gsm8k` |
| `llama32_3b` | 8 | 2 | `math` |
| `llama31_8b` | 4 | 4 | `math` |
| `qwen25_7b` | 4 | 4 | `gsm8k` |
| `gemma2_9b` | 2 | 8 | `gsm8k` |

micro-batch 는 A6000 48GB 기준 보수적 추정치다. OOM 이면 절반으로 줄이면 되고,
`GRAD_ACCUM` 은 자동으로 두 배가 되어 예산이 유지된다:
```bash
GEMMA2_9B_MB=1 bash scripts/run_baselines_multimodel.sh
```
downstream 은 `<모델키>_TASK` 로 바꾼다: `QWEN25_7B_TASK=math`.

---

## 2. 기법별 고유 하이퍼파라미터 (matched 예산 **밖**)

각 논문의 기본값을 그대로 썼다. 이 값들은 안전 메커니즘의 세기를 정하는 것이라
예산 매칭 대상이 아니다.

### LISA — `gsm8k_eval/finetune_gsm8k_lisa.py`
bi-state alternation: 안전 데이터 학습 구간과 downstream 학습 구간을 번갈아 돌고,
드리프트를 proximal term 으로 잡는다.

| 인자 | 값 | 의미 |
|---|---|---|
| `--rho` | 1.0 | proximal 계수. 클수록 정렬 상태에서 덜 벗어남 |
| `--alignment_step` | 100 | 안전 구간 step 수 |
| `--finetune_step` | 900 | downstream 구간 step 수 → 안전:다운스트림 = 1:9 |
| `--guide_data_num` | 4994 | 안전 구간에 쓰는 circuit_breakers 샘플 수(전량) |

### SafeLoRA — `finetune_gsm8k_lora.py --method safe_lora`
학습은 **표준 LoRA**. 끝난 뒤 레이어별로 `lora_B ← C·B` 사후 투영.
`C = VVᵀ/‖V‖_F`, `V = W_aligned − W_base`.

| 인자 | 값 | 의미 |
|---|---|---|
| `--safelora_base_model` | 각 모델의 **원본** chat/instruct | `V` 의 좌항 |
| `--safelora_aligned_model` | 각 모델의 **SSFT** 모델 | `V` 의 우항 |
| `--safelora_select_type` | `threshold` | 코사인 유사도 임계로 투영할 레이어 선택 |
| `--safelora_threshold` | 0.35 | 기존 gsm8k/arc/medqa 실험과 동일 |
| `--safelora_load_dtype` | float32 | `V` 계산 정밀도 |

> 분류(SST-2/AG News) 라인에서는 threshold 0.5 를 썼다. 두 값을 섞어 보고하지 말 것.

### AsFT — `finetune_gsm8k_lora.py --method asft`
SafeLoRA 와 **같은 행렬 `Ĉ`** 를 쓰되, 사후 투영 대신 매 step 손실에
`λ·Σ_l ‖(I−Ĉ_l)·B_l A_l‖²_F` 를 더한다 (arXiv:2506.08473).

| 인자 | 값 | 의미 |
|---|---|---|
| `--asft_lambda_reg` | 1.0 | 참조 구현 `AsFT_reg1_p_0.1.sh` 기본값 |
| `--asft_base_model` / `--asft_aligned_model` | SafeLoRA 와 동일 쌍 | `V = W_aligned − W_base` |
| `--asft_store_dtype` | float32 | `V` 저장 정밀도 |
| `--asft_check_equiv` | on | 첫 step 에서 참조 공식과 `rel_err` 대조 후 로그 |

`Ĉ` 를 실제로 만들면 `up_proj` 하나가 11008² fp32 > 15GB 라, `V` 만 들고
`‖(I−Ĉ)BA‖²_F = trace((XᵀX)(AAᵀ))`, `X = B − V(VᵀB)/‖V‖_F` 항등식으로 r×r 비용에 계산한다.
`--asft_check_equiv` 가 그 등가성을 런타임에 확인한다.

### SaLoRA — `finetune_gsm8k_salora.py`
safety 부분공간(`C`)과 utility 부분공간(`B` 투영)을 calibration 데이터로 미리 구해
LoRA 업데이트를 그 안에 가둔다.

| 인자 | 값 | 의미 |
|---|---|---|
| `--salora_rank_safe` | 32 | safety 부분공간 차원 `rs` — **원저자 값** |
| `--salora_rank_util` | 32 | utility 부분공간 차원 `du` — **원저자 값** |
| `--salora_calib_samples` | 128 | safety/utility 각각의 calibration 샘플 수 |
| `--salora_calib_batch_size` | 2 | calibration 배치 (Gram 메모리 때문에 작게) |
| `--salora_niter` | 20 | `svd_lowrank` 반복 수 |

> ⚠️ **원저자 설정과 다른 점**: SaLoRA 논문은 `α = r = 16`(scaling 1), 타깃 `{q, v}` 다.
> 여기서는 다른 열과 예산을 맞추려고 `α=32`, 5모듈로 돌린다(`run_salora_matched.sh` 와
> 같은 선택). 논문 표에 "budget-matched; 원저자 권장설정과 다름" 각주를 반드시 남길 것.
>
> ⚠️ utility calibration 은 **downstream 데이터**에서 뽑는다. MATH 로 돌리면
> MATH 에서 뽑힌다(이번에 `--task_data_path` 를 추가하면서 함께 반영).

### SEAL — `seal/` (Stage 1 → 1.5 → 2)
bilevel 데이터 선택. **여기만 full-param SFT** 라 위 matched 예산 밖이다.

| 단계 | 인자 | 값 |
|---|---|---|
| Stage 1 selector | epochs / batch | 2 / micro-batch, LoRA(r16 α32 dropout0.05, 5모듈) |
| | `--upperlevel_weight` / decay | 0.9 / 0.1 (레포 기본) |
| | `--selector_learning_rate` | 1e-2 (레포 기본) |
| Stage 1.5 | `--topp` | 0.8 → 상위 80% 샘플 선택 |
| Stage 2 SFT | **full-param**, lr | **5e-5** (`SEAL_LR`) |
| | epochs / effective batch | 3 / 16 |
| | weight_decay / warmup | **0.01 / 0.1** (LoRA 열의 0 / 0.03 과 다름) |

> ⚠️ SEAL 열은 **LoRA 열과 직접 비교하면 안 된다.** 학습 파라미터 수부터 다르다
> (full-param vs r=16 어댑터). 표에서 별도 열/각주로 분리할 것.
>
> ⚠️ selector 가 뱉는 인덱스는 downstream 데이터의 **행 순서**에 묶인다.
> Stage 1 과 Stage 2 가 같은 파일·같은 `num_samples` 여야 한다. 드라이버는
> 같은 `--task_data_path` 를 두 단계에 넘겨 이를 보장한다.
>
> ⚠️ 레포의 selector 는 SEAL 원본 `train_selector_llama3.sh` 와 몇 군데 다르다
> (`ul_weight` 0.9 vs 1.0, decay 0.1 vs 0.03, selector_lr 1e-2 vs 5e-3, cosine vs constant,
> LoRA 전체선형/α32 vs q,v/α16, grad accum 없음). 엄밀한 SEAL 재현이 필요하면
> `seal/README.md` 의 항목을 먼저 맞출 것.

---

## 3. 출력 구조

```
outputs/baselines_multimodel/
  lisa/<모델키>_<태스크>_lr<lr>/      finetune_config.json, run.log
  safelora/<모델키>_<태스크>_lr<lr>/  summary.json, run.log, merged model
  asft/<모델키>_<태스크>_lr<lr>/      summary.json, run.log, merged model
  salora/<모델키>_<태스크>_lr<lr>/    summary.json, run.log
  seal/<모델키>_<태스크>/             sft_config.json, run.log
seal/ckpt/<모델키>_<태스크>_selector_softmax.pt
seal/ckpt/<모델키>_<태스크>_selected_top80.json
checkpoints/ssft_<모델키>_lr5e-5/     (Stage 0 로 만든 경우)
logs/baselines_multimodel_<ts>.log
```

`PUSH_TO_HUB=1 HF_NAMESPACE=<ns>` 면 `<ns>/<모델키>-<태스크>-<기법>-r16-a32-lr<lr>-cb`
로 업로드한다.

---

## 4. 알려진 걸림돌

1. **Stage 0 (SSFT) 는 전 모델 skip 된다.** 출발 모델을 새로 만들고 싶으면 해당
   `<모델키>_ALIGNED=` 를 빈 문자열로 주면 Stage 0 가 돈다.
2. **Gemma-2 는 attention soft-capping 때문에 `attn_implementation="eager"` 가 필요**하다.
   러너들에 그 플래그가 없어서 transformers 의 자동 선택에 맡긴다. 학습 손실이 이상하면
   여기를 먼저 의심할 것.
3. **gemma2-9b 의 SSFT 모델만 lr 3e-5** 다(나머지 넷은 5e-5). 출발점의 안전정렬 강도가
   달라 모델 간 절대 수치를 비교할 때 교란 요인이 된다. 통일하려면
   `GEMMA2_9B_ALIGNED=` 를 비우고 Stage 0 로 5e-5 재생성할 것.
4. **평가는 이 스크립트에 없다.** utility 는 `gsm8k_eval/`, MATH 는 `Final Answer: $…$`
   매칭, safety 는 HarmBench ASR / beavertails-harmful 747 로 별도 수행한다.
5. **SEAL Stage 1 은 모델·태스크마다 새로 돌아간다.** 5모델 × 1태스크면 selector 5회다.
   가장 비싼 단계이므로 먼저 `METHODS="lisa safelora asft salora"` 로 LoRA 4종을 끝내고
   SEAL 은 따로 돌리는 편이 낫다.
