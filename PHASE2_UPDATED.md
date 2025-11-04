# Phase 2 업데이트: Fine-tuning + Importance Scoring

## 🔄 변경 개요

Phase 2가 **분석만 하는 것**에서 **fine-tuning + 중요도 계산을 동시에**하도록 업데이트되었습니다.

### ❌ 이전 Phase 2
```
원본 모델 (변화 없음)
    ↓
안전 데이터로 gradient 계산 (역전파 안함)
    ↓
중요도 점수만 계산
    결과: 모델 가중치는 업데이트 안 됨 ❌
```

### ✅ 새로운 Phase 2
```
원본 모델
    ↓
basis_coeff로 재매개변수화 (학습 가능)
    ↓
안전 데이터로 fine-tuning (optimizer.step() 실행)
    ↓
동시에 중요도 점수 계산 (|gradient|)
    결과: 모델이 안전 데이터로 학습됨 ✓
```

---

## 📊 핵심 개념: Weight Space vs Basis Space

### Weight Space (원본)
- `W_original`: 모델의 원래 가중치 (고정, 참고용)
- 크기: (d_out, d_in) = (4096, 14336)

### Basis Space (새로운)
- `basis_coeff`: 학습 가능한 계수 (학습됨)
  - 크기: (d_out, rank) = (4096, 14336)
  - `requires_grad=True` ← **학습 가능**
  
- `U_matrix`: 고정된 basis 행렬 (Phase 1에서 계산됨)
  - 크기: (d_in, rank) = (14336, 14336)
  - `requires_grad=False` ← **고정**

### 관계식
```
W_reconstructed = basis_coeff @ U^T

Gradient 흐름:
loss.backward() 
  → ∂L/∂basis_coeff (계산됨, 사용됨)
  → ∂L/∂W_original (계산 안함, 불필요)
  → ∂L/∂U_matrix (계산 안함, U는 고정)
```

---

## 🔧 코드 구조

### 1️⃣ `reparameterize_weights()`

**목표**: weight를 basis 공간으로 변환

```python
# Step 1: 원본 weight 저장 (고정)
W_original = target_module.weight.data.clone()  # (4096, 14336)
self.original_weights[layer_idx] = W_original

# Step 2: Basis 추출 (고정)
U = self.basis_data[layer_idx]['U']  # (14336, 14336)
target_module.U_matrix = U  # requires_grad=False

# Step 3: basis_coeff 초기화 (학습 가능)
basis_coeff_init = W_original @ U  # (4096, 14336)
target_module.basis_coeff = nn.Parameter(basis_coeff_init)  # requires_grad=True

# Forward에서 사용:
# W = basis_coeff @ U^T
```

### 2️⃣ `compute_importance()`

**목표**: Fine-tuning과 동시에 중요도 계산

#### Phase 1: 학습 준비
```python
# Optimizer 설정 (basis_coeff 파라미터만)
optimizer = AdamW([basis_coeff], lr=1e-5)

# Forward hook: weight 동적 복원
def hook(module, input, output):
    W = module.basis_coeff @ module.U_matrix.T
    module.weight.data = W
    return output
```

#### Phase 2: 훈련 루프
```python
for epoch in range(epochs):
    for batch in dataloader:
        # Forward: weight = basis_coeff @ U^T (hook에서 자동)
        outputs = model(input_ids, attention_mask)
        
        # Loss 계산
        loss = CrossEntropyLoss(logits, targets)
        
        # Backward: basis_coeff.grad 계산
        optimizer.zero_grad()
        loss.backward()
        
        # Importance 수집 (backward 후)
        grad_abs = |basis_coeff.grad|  # Element-wise absolute value
        importances.append(grad_abs)
        
        # Update: basis_coeff 업데이트
        optimizer.step()
```

#### Phase 3: Importance 평균
```python
# 모든 배치의 gradient 절댓값 수집 (num_batches, d_out, rank)
layer_importances = stack(importances)

# 배치 축 평균
importance_mean = layer_importances.mean(dim=0)  # (d_out, rank)

# Input 차원별 sum (각 input의 누적 영향)
importance_per_input = importance_mean.sum(dim=0)  # (rank,)
```

---

## 📈 로그 출력 해석

### 재매개변수화 단계
```
Layer 31:
  ✓ W_original (고정):     torch.Size([4096, 14336])
  ✓ basis_coeff (학습):    torch.Size([4096, 14336])
  ✓ U_matrix (고정):       torch.Size([14336, 14336])
  ✓ Forward: W = basis_coeff @ U^T
```

### 훈련 단계
```
Training Setup
  ✓ Model set to training mode
  ✓ Optimizer created: AdamW
  - Learning rate: 1e-05
  - Weight decay: 0.01
  - Parameters: 1 basis_coeff tensors
  - Layers: [31]
  ✓ 1 forward hooks registered

Fine-tuning with Importance Tracking
[Epoch 1/2] Loss: 0.8234  ← basis_coeff 업데이트 중
[Epoch 2/2] Loss: 0.7891  ← 손실 감소 (fine-tuning 작동)

Computing Importance Scores
✓ Layer 31:
  - Gradient shape (per batch): (d_out, rank) = torch.Size([4096, 14336])
  - Importance aggregated to input-wise (sum): (14336,)
  - Mean: 0.012345
  - Std: 0.005678
```

---

## 🎯 Phase 2 실행

### 기본 명령어
```bash
python train.py \
    --phase 2 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --basis_dir ./checkpoints/phase1_*/basis \
    --safety_samples 5000 \
    --batch_size 4 \
    --safety_epochs 2 \
    --safety_lr 1e-5 \
    --keep_ratio 0.1 \
    --device cuda \
    --seed 42
```

### 파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--safety_epochs` | 1 | Fine-tuning 에포크 수 |
| `--safety_lr` | 1e-5 | basis_coeff 학습률 |
| `--safety_weight_decay` | 0.01 | Weight decay |
| `--keep_ratio` | 0.1 | 유지할 중요도 비율 (상위 10%) |

---

## ✅ 검증 사항

### 1️⃣ basis_coeff 학습 확인
```python
# Phase 2 전후 basis_coeff 변화 확인
before = basis_coeff_init.norm()
# ... 훈련 ...
after = basis_coeff.detach().norm()

print(f"basis_coeff norm changed: {before:.4f} → {after:.4f}")
# 값이 변했으면 학습 진행 중
```

### 2️⃣ Importance 점수 확인
```python
# importance가 양수 값인지 확인
importance_min = importance.min()
importance_max = importance.max()

print(f"Importance range: [{importance_min:.6f}, {importance_max:.6f}]")
# 모두 >= 0이어야 함 (절댓값이므로)
```

### 3️⃣ 마스크 생성 확인
```python
# keep_ratio=0.1일 때, 상위 10%가 마스크되는지 확인
mask_sum = mask.sum().item()
total = len(mask)
actual_ratio = mask_sum / total

print(f"Masked elements: {mask_sum}/{total} ({actual_ratio*100:.1f}%)")
# ~10% 근처여야 함
```

---

## 🚨 주의사항

### ⚠️ Weight Space 혼동 방지

```python
# ❌ 잘못된 사용
W = target_module.weight  # 원본 weight (재구성되지 않은)
# → Hook이 자동으로 재구성하지만, 명시적으로 사용하면 헷갈림

# ✅ 올바른 사용
# Forward pass 중에만 사용 (hook에서 자동 처리)
output = model(input_ids)  # W = basis_coeff @ U^T (자동)
```

### ⚠️ Gradient 흐름 확인

```python
# basis_coeff만 학습되어야 함
optimizer = AdamW([basis_coeff], lr=1e-5)  # ✓ 올바름

# ❌ 잘못된 방식
optimizer = AdamW(model.parameters())  # 전체 파라미터 포함
# → U_matrix와 W_original도 업데이트 시도 (불필요)
```

### ⚠️ Hook 등록 시점

```python
# Hook은 forward 전에 등록되어야 함
register_forward_hook(make_forward_hook)

# 그 후 model(input_ids)를 호출하면 자동으로 weight 재구성
```

---

## 📝 출력 파일

Phase 2 완료 후 다음 파일이 생성됩니다:

```
./checkpoints/phase2_*/
  └─ checkpoints/
      └─ masks/
          ├─ layer_31_mask.pt      # Binary mask (1: frozen, 0: trainable)
          └─ metadata.json         # 메타데이터
```

---

## 🔄 다음 단계 (Phase 3)

Phase 2에서 생성된 mask를 Phase 3에서 사용:

```bash
python train.py \
    --phase 3 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --basis_dir ./checkpoints/phase1_*/basis \
    --masks_dir ./checkpoints/phase2_*/checkpoints/masks \
    --utility_samples 1000 \
    --epochs 3 \
    --device cuda
```

---

## 💡 추가 팁

### Phase 2 손실이 떨어지지 않으면?

```bash
# 학습률 증가
--safety_lr 5e-5

# 더 많은 에포크
--safety_epochs 3

# 더 많은 샘플
--safety_samples 10000
```

### Importance 점수가 너무 크거나 작으면?

```python
# 정규화 추가 (optional)
importance_normalized = (importance - importance.mean()) / (importance.std() + 1e-8)
```

---

## 📚 참고

이 구현은 **WaRP-CIFSL 원본 방식**을 따릅니다:
- Element-wise gradient 절댓값 계산
- Per-input 중요도 aggregation
- Quantile 기반 마스크 생성

