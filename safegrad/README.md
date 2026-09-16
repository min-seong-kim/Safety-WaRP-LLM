# SafeGrad — Gradient Surgery for Safe LLM Fine-tuning

논문: **"SafeGrad: Gradient Surgery for Safe LLM Fine-tuning"** (Yi et al., arXiv:2508.07172)
참조 구현: `/NHNHOME/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/SafeGrad/`

이 저장소의 **baseline 비교군**으로 이식한 것이다. LISA / AsFT / SafeLoRA / SaLoRA /
WSR-LoRA 와 같은 조건(같은 출발 모델, 같은 태스크 JSON, 같은 LoRA 예산, 같은 옵티마이저
설정)에서 돌 수 있게 맞췄다.

## 방법 한 줄 요약

매 step 유저 태스크 그래디언트 `g_user` 와 안전 alignment 그래디언트 `g_align` 을 따로 구하고,
**둘이 충돌할 때만**(모델 전체를 한 벡터로 본 내적이 음수일 때) 유저 쪽에서 안전 쪽을 정면으로
거스르는 성분을 깎아낸 뒤 합친다.

```
g_align = ∇_θ D_KL( P_θ0(·|x^a) ‖ P_θ(·|x^a) )          (Eq. 6, θ0 = 얼린 안전정렬 모델)

if  g_user · g_align < 0 :                                (Eq. 3)
        g'_user = g_user − (g_user·g_align / ‖g_align‖²)·g_align     (Eq. 4)
else :  g'_user = g_user

g_final = g'_user + ρ · g_align                           (Eq. 5)
```

투영 후에는 정의상 `cos(g'_user, g_align) = 0` 이다 — 논문 Figure 4(a) 의 "After Gradient
Surgery" 선이 0 에 붙어 있는 것이 이 성질이다.

두 번째 기여는 alignment loss 를 **SFT(CE) 대신 KL** 로 두는 것이다. 거부 토큰 몇 개의 확률만
올리는 대신 잘 정렬된 원본 모델의 **출력 분포 전체**를 따라가게 해서, 논문 Table 6 기준
alignment 샘플 10개만으로도 동작한다(SFT 는 100개 이상 필요).

## 파일

| 파일 | 내용 |
|---|---|
| `safegrad_trainer.py` | `SafeGradTrainer` — HF `Trainer` 서브클래스. gradient surgery + KL alignment loss |
| `finetune_safegrad.py` | 러너 (CLI). `gsm8k_eval/finetune_gsm8k_lisa.py` 의 구조·인자·저장 규약을 그대로 따른다 |
| `test_safegrad.py` | 구현 검증 6종 (아래) |
| `scripts/_smoke_safegrad.sh` | 작은 랜덤 LLaMA 로 전 경로를 한 번 태우는 스모크 테스트 (~1분) |

## 실행

### revision 러너로 (권장 — 리포명·재개·업로드가 규약대로 처리된다)

`safegrad` 는 **기본 `METHODS` 목록에 없다**. 116셀 계획을 말없이 늘리지 않기 위함이므로
돌릴 때 명시한다.

```bash
METHODS=safegrad MODELS=llama2_7b SAFETY_SETS=cb TASKS=gsm8k \
LORA_ALPHA=16 PUSH_TO_HUB=1 PRUNE_AFTER_UPLOAD=0 \
bash scripts/revision/20_lora_family.sh
```

→ `outputs/revision/cb/llama2_7b/gsm8k/safegrad/`,
   리포 `kmseong/llama2_7b-chat-CB_SSFT-safegrad_gsm8k_rho1.0_a16_lr3e-4`

관련 환경변수 (`scripts/revision/common.sh`):
`SAFEGRAD_RHO`(1.0) · `SAFEGRAD_REF_MODE`(adapter_off) · `SAFEGRAD_KL_REDUCTION`(ref) ·
`SAFEGRAD_ALIGN_BS`(0 = 태스크 배치와 동일)

### 직접 실행

```bash
python safegrad/finetune_safegrad.py \
  --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
  --output_dir ./safegrad_gsm8k \
  --task_data_path data/gsm8k_train_task_7473.json \
  --safety_data_path data/circuit_breakers_train.json --guide_data_num 4994 \
  --rho 1.0 --learning_rate 3e-4 --epochs 3 \
  --lora --lora_r 16 --lora_alpha 16 \
  --batch_size 4 --grad_accum 4 --gradient_checkpointing
```

### 검증

```bash
python safegrad/test_safegrad.py          # 수학/누적/동치성 6종, 수십 초
bash safegrad/scripts/_smoke_safegrad.sh  # 러너 전 경로, ~1분
```

## 원 논문·참조 구현과의 차이 (전부 의도적)

| | 원 논문 / 참조 구현 | 여기 |
|---|---|---|
| 유저 데이터 | harmful ratio `hr` 로 오염시킨 데이터 | **clean 태스크 데이터** |
| alignment 데이터 | BeaverTails refusal | `data/circuit_breakers_train.json` (또는 bt 변형) 의 `llama3_output` |
| alignment 샘플 수 | 100 | 4994 (전체) — 이 저장소의 `SAFETY_SAMPLES` 와 동일 |
| LoRA | r=8, α=16, {q,k,v} | r=16, α=16, {q,k,v,up,down} |
| reference θ0 | 출발 모델을 따로 한 벌 더 로드 | 기본은 `adapter_off` (수학적 동치, 아래) |

- **clean 데이터인 이유**: 이 저장소의 비교군은 전부 "안전정렬된 출발 모델을 clean
  다운스트림 데이터로 FT 했을 때 안전성이 얼마나 무너지는가" 를 본다. LISA 도 같은 방식으로
  이식돼 있다. poison 실험이 필요하면 섞은 JSON 을 `--task_data_path` 로 주면 된다 —
  러너는 데이터 내용을 가정하지 않는다.
- **alignment 데이터가 안전축을 따르는 이유**: "안전 데이터를 필요로 하는 기법은 자기 출발
  모델이 안전정렬된 바로 그 데이터를 쓴다" 가 revision 라인의 불변식이다.
- **LoRA 예산**: 기법 간 차이가 "안전 메커니즘" 하나로만 남게 하려고 저장소 공통값에 맞춘다.
  논문 설정을 그대로 보려면 `--lora_r 8 --lora_alpha 16 --lora_target_modules q_proj k_proj v_proj`.
  ⚠️ α 는 다른 arm 과 반드시 맞춰라. `lora_alpha_tag()` 가 α≠32 일 때 리포명에 `_a16` 을 붙이는
  것은 α 세대가 섞여 비교되는 사고(CLAUDE.md 의 WSR-LoRA α 항목)를 막기 위한 것이다.

## 참조 구현에서 **고친** 것

### 1. gradient accumulation (반드시 유지할 것)

참조 구현 `safegrad_trainer.py:152` 는 최종 그래디언트를 `param.grad = final_grad` 로
**덮어쓴다**. 참조 스크립트가 항상 `gradient_accumulation_steps=1` 이라 드러나지 않았을 뿐,
accum>1 이면 마지막 micro-batch 만 남는 버그다. 이 저장소의 LoRA 계열은 effective batch 16 을
micro×accum(예: 4×4)으로 쪼개므로 그대로 쓰면 **실효 배치가 1/4 로 줄고 앞의 3개 배치가 통째로
버려진다**.

여기서는 micro-step 진입 시점의 누적 그래디언트를 보관했다가 `param.grad = prev + final` 로
더한다. accum=1 이면 참조 구현과 완전히 동일하게 동작한다.
`test_safegrad.py` 의 검사 [5] 가 이 성질을 지킨다.

### 2. 로깅

참조 구현은 매 step `self.log()` 를 부른다. 여기서는 통계를 모아 뒀다가 Trainer 가
`logging_steps` 주기로 로깅할 때 같이 내보낸다. 로그 키:
`safegrad/loss_task`, `safegrad/loss_align_kl`, `safegrad/grad_dot`,
`safegrad/grad_align_norm_sq`, `safegrad/projection_scalar`, `safegrad/conflict_rate`.
`conflict_rate` 가 논문 Table 1 / Figure 4(a) 에 대응한다 — harmful 비율이 올라가면 이 값이
올라가야 한다.

## 참조 구현과 **같게 유지한** 것 (건드리지 말 것)

- **전역 내적 하나로 충돌을 판정한다.** 레이어별로 따로 투영하지 않는다.
- **KL 방향.** `F.kl_div(log_p_theta, p_ref)` 는 PyTorch 규약상 `KL(P_ref ‖ P_theta)` 이고,
  논문 Eq. 6 의 `D_KL(P_θ0 ‖ P_θ)` 와 일치한다. 인자 순서를 "고치지" 말 것.
- **`kl_reduction="ref"`(기본).** `reduction='sum'` 으로 **패딩·프롬프트 위치까지 포함해** 더한
  뒤 `(labels != -100)` 개수로 나눈다. 분자와 분모의 토큰 집합이 다른, 참조 구현 그대로의
  정의다. 값이 배치 패딩량에 의존하므로 `--kl_reduction response` 로 응답 토큰만 쓰게 할 수
  있지만, baseline 비교에는 기본값을 쓴다.
- **KL 계산 dtype.** 기본은 참조와 같은 모델 dtype(bf16). `--kl_fp32` 로 올릴 수 있다(메모리 2배).
- **`‖g_align‖² > 1e-9` 가드.** LoRA 는 `B=0` 으로 시작하므로 step 0 에서 θ=θ0 → `g_align=0` 이다.
  이 가드가 0 나눗셈을 막는다.
  `‖g_align‖` 이 아주 작을 때 `projection_scalar` 가 수백~수천까지 커지는 것은 **정상**이다.
  Cauchy–Schwarz 에 의해 실제로 빼는 양 `|scalar|·‖g_align‖ ≤ ‖g_user‖` 로 묶여 있어서
  투영된 그래디언트의 크기는 `2‖g_user‖` 를 넘지 않는다. 스칼라 크기만 보고 클리핑을 추가하지 말 것.

## `--ref_mode`

| 값 | θ0 를 얻는 방법 | 메모리 |
|---|---|---|
| `separate` | 출발 모델을 한 벌 더 로드 (참조 구현과 동일) | 모델 한 벌 추가 |
| `adapter_off` (기본) | LoRA 어댑터를 끈 현재 모델 | 추가 없음 |

LoRA 를 **출발 모델 위에 바로** 얹은 경우 어댑터를 끈 모델은 정의상 출발 모델 그 자체이므로
두 모드는 수학적으로 같다. `test_safegrad.py` 의 검사 [6] 이 같은 배치에서 KL 과 `g_align` 이
일치함을 확인한다(랜덤 tiny 모델 기준 상대오차 0). 단, **다른 어댑터를 merge 한 뒤 학습하는
흐름에서는 쓰면 안 된다** — 그때는 θ0 ≠ "어댑터 끈 모델" 이다. full-param 학습에서는 쓸 수 없다
(러너가 막는다).

## 비용

한 step 에 forward/backward 가 두 번(태스크 + alignment) 돌고 reference forward 가 한 번 더
붙는다. 논문 Table 8 의 overhead 분석과 같은 구조다 — SFT 7.32 GB·h → SafeGrad(KL) 27.71 GB·h.
그래디언트 사본은 최대 3벌(prev / g_user / g_align) 필요한데 LoRA 에서는 무시할 수준이지만
**full-param 이면 모델 크기의 3배**가 더 든다(러너가 경고한다).

vocab 이 큰 모델(gemma-2 는 256k)에서는 KL 의 logits 텐서가 커진다. 부담되면
`--align_batch_size` 를 태스크 배치보다 작게 잡아라 (참조 구현은 둘이 같다).

## 검증 항목 (`test_safegrad.py`)

1. 충돌이 없으면 투영하지 않는다 (`projection_scalar = 0`)
2. 충돌하면 투영 후 `cos(g'_user, g_align) ≈ 0`
3. `projection=False` 면 surgery 가 전혀 일어나지 않는다
4. θ=θ0 이면 KL ≈ 0, `g_align` ≈ 0
5. accum=2 에서 `param.grad` = 두 micro-step 그래디언트의 **합** (참조 구현 버그 회귀 방지)
6. `ref_mode` 두 값이 같은 KL / `g_align` 을 준다
