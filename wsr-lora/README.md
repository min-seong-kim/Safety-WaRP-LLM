# WSR-LoRA 구현 및 알고리즘

이 디렉터리는 WSR-LoRA와 비교 대상인 SaLoRA의 핵심 Python 구현만 모은 전달용
스냅샷이다. 여기서 **WSR-LoRA의 주 구현**은 [`wsr_lora.py`](wsr_lora.py)의
`reparam=True` 변형을 의미한다.

## 한 줄 요약

WSR-LoRA는 안전 데이터에서 얻은 입력 기저로 LoRA 인자를 회전한 뒤, 안전 loss에
민감한 저랭크 인자 좌표를 찾아 PiSSA 초기값에 고정하고 나머지 좌표만 downstream
task로 학습하는 방법이다.

중요한 점은 안전 mask가 dense weight 전체에 적용되는 것이 아니라 LoRA의 두 인자
`B`와 `A_tilde`에 각각 적용된다는 것이다. 따라서 매 step마다
`d_out × d_in` 크기의 dense mask update를 만들지 않고 저랭크 학습 구조를 유지한다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| [`wsr_lora.py`](wsr_lora.py) | 현재 WSR-LoRA 핵심 구현: basis 로드, PiSSA 초기화, factor 중요도, gradient mask, downstream 학습, 병합 |
| [`pissa_wsr_lora.py`](pissa_wsr_lora.py) | 공통 데이터 처리, `topmask`, target module 처리 및 이전 dense product-mask WSR 구현 |
| [`salora_impl.py`](salora_impl.py) | 비교 대상 SaLoRA의 안전 출력 부분공간과 projection 구현 |
| [`salora_lora.py`](salora_lora.py) | SaLoRA 학습 진입점 |

`wsr_lora.py`는 `pissa_wsr_lora.py`를 직접 import하므로 두 파일을 반드시 같은
디렉터리에 둬야 한다.

## 문제 설정과 표기

대상 선형층 하나에 대해 다음과 같이 표기한다.

- `W_0 ∈ R^(d_out × d_in)`: 안전 정렬된 시작 모델의 weight
- `r`: LoRA rank
- `s = alpha / r`: LoRA scaling
- `U ∈ R^(d_in × d_in)`: 안전 데이터의 입력 활성화로 만든 정방 직교 기저
- `B ∈ R^(d_out × r)`: 학습할 LoRA 출력 인자
- `A_tilde ∈ R^(r × d_in)`: 안전 기저로 회전된 학습 인자
- `rho ∈ [0, 1)`: 각 인자에서 고정할 원소 비율
- `D_s`, `D_t`: 각각 safety dataset과 downstream dataset

목표는 `W_0`의 안전 관련 동작을 최대한 유지하면서 `B`와 `A_tilde`만 downstream
task에 맞게 학습하는 것이다.

## 알고리즘

### 1. 안전 입력 기저 준비

안전 데이터가 대상 선형층으로 입력하는 token activation을 `X_s`라고 하면 WaRP
Phase 1은 다음 Gram matrix의 고유분해/SVD로 기저를 만든다.

```text
G = X_s^T X_s
G = U Lambda U^T
```

현재 `wsr_lora.py`는 이 기저를 직접 생성하지 않고 `--basis_dir`에서 기존 Phase-1
결과를 검증하고 로드한다. 각 target layer에는 입력 차원과 같은
`d_in × d_in` 크기의 전체 `U`가 필요하다.

기본 target과 basis 디렉터리의 대응은 다음과 같다.

| Target module | Basis 하위 디렉터리 |
|---|---|
| `q_proj` | `attn_q` |
| `k_proj` | `attn_k` |
| `v_proj` | `attn_v` |
| `up_proj` | `ffn_up` |
| `down_proj` | `ffn_down` |

예를 들어 0번 layer의 query basis는
`BASIS_DIR/attn_q/layer_00_svd.pt`에 있어야 하며, 파일 내부에는 `U` tensor가
있어야 한다. `BASIS_DIR/metadata.json`은 최소한 다음 조건을 만족해야 한다.

- `decomp`가 `"svd"`일 것
- `total_samples`가 실행 시 `--basis_samples`와 같을 것
- `layer_types`가 요청한 target module의 basis type을 모두 포함할 것

기저는 shape만 맞추는 것으로 충분하지 않다. 원칙적으로 학습을 시작할 동일한
`W_0` 모델과 revision의 안전 데이터 activation으로 만든 기저를 사용해야 한다.

### 2. 함수 보존 PiSSA 초기화

`W_0`의 truncated SVD를 다음과 같이 계산한다.

```text
W_0 = P Sigma Q^T
```

상위 `r`개 성분과 scaling `s`를 이용해 초기 저랭크 인자를 만든다.

```text
B_0 = P_r sqrt(Sigma_r / s)
A_0 = sqrt(Sigma_r / s) Q_r^T
W_res = W_0 - s B_0 A_0
```

WSR-LoRA는 입력 기저에서 학습하도록 `A_0`를 회전한다.

```text
A_tilde_0 = A_0 U
```

초기 유효 weight는 다음과 같으므로 원래 모델 함수가 보존된다.

```text
W_res + s B_0 A_tilde_0 U^T
= W_res + s B_0 A_0 U U^T
= W_0
```

즉 reparameterization 직후에는 adapter가 모델 출력을 바꾸지 않는다.

### 3. Factor별 안전 중요도 계산

PiSSA 초기점에서 safety response에 대한 response-only cross-entropy를 계산한다.
Prompt token의 label은 `-100`으로 가려지므로 loss와 gradient는 assistant response
token에 대해서만 계산된다.

각 safety mini-batch의 gradient 절댓값을 layer별로 누적한다.

```text
I_B       = sum_batches abs(d L_safe / d B)
I_A_tilde = sum_batches abs(d L_safe / d A_tilde)
```

`reparam=True`일 때 `A` parameter 자체가 `A_tilde`이므로 `I_A_tilde`도 안전 입력
기저의 회전 좌표에서 계산된다.

### 4. 고정 mask 생성

각 layer의 `B`와 `A_tilde`에 대해 중요도가 가장 큰 상위 `rho` 비율을 서로
독립적으로 선택한다.

```text
M_B       = TopRho(I_B)
M_A_tilde = TopRho(I_A_tilde)
K_B       = 1 - M_B
K_A_tilde = 1 - M_A_tilde
```

`M=1`인 원소가 고정 대상이고, 실제 코드에는 update 가능한 위치를 나타내는
`keep=K`가 저장된다. 이는 모델 전체에 대한 global top-rho가 아니라 **layer별,
factor별 top-rho**이다.

### 5. Downstream 학습

Base model, `W_res`, `U`는 모두 고정하고 `B`와 `A_tilde`만 학습한다. 열벡터 입력
`h`에 대한 forward는 다음과 같다.

```text
y = W_res h + s B A_tilde U^T h
```

코드는 계산 순서를 조정해 token마다 별도의 `d_in × d_in` 행렬곱을 수행하지 않는다.
Backward 시 parameter hook이 gradient에 keep mask를 곱한다.

```text
g_B       <- K_B       elementwise-multiply g_B
g_A_tilde <- K_A_tilde elementwise-multiply g_A_tilde
```

Mask는 forward에서 factor 값을 0으로 만드는 용도가 아니다. 고정된 원소도 forward에
계속 참여하며, gradient만 0이 되어 안전 정렬된 PiSSA 초기값에 머문다. 이 불변성을
지키기 위해 현재 구현은 `weight_decay=0`만 허용한다.

### 6. 일반 모델로 병합

학습이 끝나면 각 custom layer를 다음 weight의 일반 `nn.Linear`로 교체한다.

```text
W_* = W_res + s B_* A_tilde_* U^T
```

최종 변화량은 다음과 같다.

```text
W_* - W_0 = s (B_* A_tilde_* - B_0 A_tilde_0) U^T
```

결과는 PEFT adapter가 아니라 일반 Hugging Face full model이며, 실행 설정은 출력
디렉터리의 `wsrlora_run_config.json`에 함께 저장된다. 추론 시 basis, mask 또는
WSR 전용 module은 필요하지 않다.

## 전체 의사코드

```text
for each target layer l:
    U_l <- load Phase-1 safety input basis
    P, Sigma, Q <- truncated_svd(W_0,l, rank=r)
    B_l <- P_r sqrt(Sigma_r / s)
    A_l <- sqrt(Sigma_r / s) Q_r^T
    W_res,l <- W_0,l - s B_l A_l
    A_tilde_l <- A_l U_l

I_B, I_A <- 0, 0
for each safety batch b_s:
    L_safe <- response_only_ce(model, b_s)
    I_B <- I_B + abs(grad_B(L_safe))
    I_A <- I_A + abs(grad_A_tilde(L_safe))

for each layer l:
    K_B,l <- 1 - top_rho(I_B,l)
    K_A,l <- 1 - top_rho(I_A,l)
    register gradient hooks using K_B,l and K_A,l

freeze every parameter except B and A_tilde
for each downstream batch b_t:
    L_task <- response_only_ce(model, b_t)
    backpropagate L_task
    g_B <- K_B elementwise-multiply g_B
    g_A <- K_A elementwise-multiply g_A
    optimizer_step(weight_decay=0)

for each target layer l:
    W_*,l <- W_res,l + s B_l A_tilde_l U_l^T
save merged Hugging Face model
```

## 가장 간단한 실행 방법

아래 정도만 준비하면 실행을 시작할 수 있다. 현재 코드는 device를 `cuda`로 고정하므로
NVIDIA CUDA GPU가 필요하다.

### 1. 파일 배치

최소한 다음 파일과 입력을 한 작업 디렉터리에서 접근할 수 있게 둔다.

```text
workdir/
├── wsr_lora.py
├── pissa_wsr_lora.py
├── safety.json
├── downstream.json
└── basis/
    ├── metadata.json
    ├── attn_q/
    ├── attn_k/
    ├── attn_v/
    ├── ffn_up/
    └── ffn_down/
```

`wsr_lora.py`와 `pissa_wsr_lora.py`는 같은 디렉터리에 두는 것이 가장 간단하다.
모델은 이 디렉터리에 복사하지 않고 Hugging Face model ID나 로컬 checkpoint 경로로
지정해도 된다.

### 2. 환경 설치

대상 GPU와 CUDA에 맞는 PyTorch를 먼저 설치한 다음 나머지 패키지를 설치한다.

```bash
python -m venv .venv
source .venv/bin/activate

# CUDA 환경에 맞는 torch는 별도로 먼저 설치
python -m pip install "transformers>=4.46,<5" "accelerate>=0.34" \
  "datasets>=2.18" "safetensors>=0.4" "huggingface_hub>=0.25" \
  "sentencepiece>=0.2" "protobuf>=4.25"
```

WSR-LoRA는 custom `nn.Module` 구현이므로 `peft`가 필수는 아니다. 같은 폴더의
SaLoRA를 실행할 때는 `peft`를 추가로 설치해야 한다.

### 3. 최소 명령으로 실행

```bash
CUDA_VISIBLE_DEVICES=0 python wsr_lora.py \
  --model_name MODEL_ID_OR_CHECKPOINT \
  --safety_data ./safety.json \
  --gsm8k_json ./downstream.json \
  --basis_dir ./basis \
  --basis_samples 4994 \
  --reparam \
  --output_dir ./output_wsrlora
```

이 명령은 나머지 인자에 코드 기본값을 사용한다. `--basis_samples`는 임의의 숫자가
아니라 `basis/metadata.json`의 `total_samples`와 동일하게 바꿔야 한다.

정상 종료되면 `output_wsrlora/`에 병합된 Hugging Face 모델과
`wsrlora_run_config.json`이 생성된다.

### 4. Basis가 아직 없을 때

회전된 주 WSR-LoRA에는 Phase-1 basis가 필수다. 우선 코드 동작만 확인해야 한다면
`--basis_dir`, `--basis_samples`, `--reparam`을 빼고 실행할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 python wsr_lora.py \
  --model_name MODEL_ID_OR_CHECKPOINT \
  --safety_data ./safety.json \
  --gsm8k_json ./downstream.json \
  --rho 0.1 \
  --output_dir ./output_no_rotation
```

이 경우에도 factor 안전 중요도와 gradient freeze는 적용되지만, `U` 기저 회전을
사용하지 않는 `WSR-LoRA(no_rotation)` ablation이다. 논문용 주 알고리즘과 완전히
같은 설정은 아니다.

작은 smoke test가 필요하면 아래처럼 데이터와 sequence 길이를 줄일 수 있다.

```text
--safety_samples 8 --train_samples 8 --batch_size 1 --grad_accum 1 --max_length 128
```

### 5. 실행 코드를 다시 작성할 때 전달할 정보

상대방이 GPT 등을 이용해 launcher를 다시 만들 경우 다음 정보만 명확히 전달하면 된다.

- 시작 모델의 정확한 model ID 또는 checkpoint와 가능하면 revision
- safety JSON과 downstream JSON의 경로 및 field 형식
- Phase-1 basis를 만든 모델, sample 수, `metadata.json`과 basis 디렉터리 경로
- target module 목록과 원하는 `rank`, `alpha`, `rho`
- `--reparam` 사용 여부와 `weight_decay=0` 조건
- GPU 번호, batch size, gradient accumulation, 출력 경로

## 상세 실행 예시

아래는 회전 기저와 두 factor mask를 모두 사용하는 주 WSR-LoRA 구현의 직접 실행
형태다. 실제 실험값은 비교 protocol에 맞게 명시적으로 지정하는 것이 좋다.

```bash
CUDA_VISIBLE_DEVICES=0 python wsr_lora.py \
  --model_name MODEL_OR_LOCAL_CHECKPOINT \
  --safety_data /path/to/safety.json \
  --gsm8k_json /path/to/downstream.json \
  --basis_dir /path/to/phase1_run/basis \
  --basis_samples 4994 \
  --target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
  --rank 16 --alpha 16 --dropout 0 \
  --rho 0.1 --reparam --mask_B 1 --mask_A 1 \
  --safety_samples 512 --basis_batch_size 2 \
  --lr 3e-5 --epochs 1 --batch_size 2 --grad_accum 8 \
  --max_length 1024 --weight_decay 0 --seed 42 \
  --output_dir /path/to/output
```

필요한 입력 JSON 형식은 다음과 같다.

```json
[
  {"prompt": "harmful prompt", "llama3_output": "safe refusal response"}
]
```

```json
[
  {"question": "downstream prompt", "answer": "target response"}
]
```

Downstream target에는 `answer` 대신 `response`를 사용할 수도 있다.

## Ablation과 옵션의 의미

| 설정 | 의미 |
|---|---|
| `--reparam` | `A_tilde=A U` 회전을 사용하는 주 WSR-LoRA. 반드시 `--basis_dir` 필요 |
| `--reparam` 생략 | 입력 회전 없이 원래 `B`, `A` 좌표에 같은 factor 중요도/mask 적용 |
| `--mask_B 0` | `B`는 모두 학습하고 `A`에만 안전 mask 적용 |
| `--mask_A 0` | `A`는 모두 학습하고 `B`에만 안전 mask 적용 |
| `--no_freeze` | 안전 중요도와 mask를 모두 생략한 PiSSA-LoRA baseline |
| `--mask_cache PATH` | 동일 provenance의 중요도 mask를 재사용 |

Mask cache는 model, rank, alpha, rho, safety data, basis 경로와 seed 등의 context가
정확히 일치할 때만 재사용된다.

## 해석 시 주의점

- 이 방법은 factor 원소를 PiSSA 초기값에 고정하는 **원소별 affine constraint**이다.
  Projection 방식이 아니다.
- `B A_tilde`는 bilinear product이므로 일부 factor 원소를 고정해도 대응하는 dense
  weight 원소가 각각 불변이라고 보장할 수는 없다. 정확한 표현은
  “first-order safety-important factor coordinates의 변화를 억제한다”이다.
- [`pissa_wsr_lora.py`](pissa_wsr_lora.py)의 `PissaWsrLinear`는
  `(1-M) elementwise-multiply ((BA-B0A0)U)` 형태의 dense product mask를 매 step
  구성하는 이전 구현이다. 현재 주 구현인 [`wsr_lora.py`](wsr_lora.py)는 `B`와
  `A_tilde`의 gradient를 직접 mask해 이 dense materialization을 피한다.
- 현재 shared-basis 경로 해석은 `.layers.<index>.` 형태의 Llama 계열 module naming을
  전제로 한다. 다른 architecture에는 `_basis_location()`과 target mapping 수정이
  필요할 수 있다.
- 코드의 기본값은 `alpha=32`, `dropout=0.05`, `lr=1e-4`, `epochs=3`이다. 논문이나
  baseline 비교에서는 launcher의 암묵적 기본값에 의존하지 말고 실제 protocol 값을
  명시해야 한다.

## SaLoRA와의 핵심 차이

SaLoRA는 factor 중요도 상위 원소를 고정하지 않는다. 안전 response에서 얻은 출력
부분공간 `U_C`로 `C_S = I - U_C U_C^T`를 만들고, LoRA update 전체를 `C_S`로
투영한다. 즉 WSR-LoRA는 **입력 기저의 factor별 gradient freeze**, SaLoRA는
**출력 여공간으로의 구조적 projection**이라는 차이가 있다.
