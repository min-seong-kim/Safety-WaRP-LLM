# Phase 3 Implementation Summary

## 📋 Phase 3: Incremental Learning (완성)

### 🎯 핵심 기능

**Phase 3는 다음을 구현합니다:**

1. **Basis & Masks 로드**
   - Phase 1: SVD basis (U, S, Vh) 로드
   - Phase 2: 중요도 마스크 로드

2. **마스킹된 미세조정**
   - GSM8K train split으로 훈련
   - Backward pass에서 마스크 적용
   - 중요 뉴런(mask=1): gradient = 0 (업데이트 불가)
   - 덜 중요한 뉴런(mask=0): 정상 업데이트

3. **체크포인트 저장**
   - 각 에포크마다 체크포인트 저장
   - 최고 성능 모델 저장

---

## 📁 생성된 파일들

### 1. `models/phase3_learning.py` (480줄)
**Phase3IncrementalLearner 클래스:**

```python
# 주요 메서드
- load_basis()              # Phase 1 basis 로드
- load_masks()              # Phase 2 masks 로드
- load_model()              # 모델 로드
- load_utility_data()       # GSM8K train split 로드
- register_mask_hooks()     # Backward hook 등록 (gradient masking)
- train_epoch()             # 한 에포크 훈련
- train()                   # 전체 훈련 루프
- save_checkpoint()         # 체크포인트 저장
```

**마스킹 메커니즘:**
```python
def make_backward_hook(layer_idx, mask):
    def backward_hook(grad):
        # mask=1 (중요): grad * 0 = 0 (업데이트 금지)
        # mask=0 (불필요): grad * 1 = grad (정상 업데이트)
        grad_masked = grad * (1 - mask)
        return grad_masked
    return backward_hook
```

### 2. `models/safety_evaluator.py` (500줄)
**SafetyEvaluator 클래스:**

```python
# 주요 메서드
- evaluate_safety_asr()     # do-not-answer으로 안전성 평가
- evaluate_utility_gsm8k()  # GSM8K로 유틸리티 평가
- evaluate_all()            # 모두 평가
- is_refusal()              # 거절 판정
```

**평가 메트릭:**
- Safety ASR: 유해 응답 비율 (↓ 낮을수록 좋음)
- Safety Rate: 거절률 (↑ 높을수록 좋음)
- Utility Accuracy: GSM8K 정확도 (↑ 높을수록 좋음)

### 3. `scripts/run_phase3.sh` (85줄)
**Phase 3 실행 스크립트**

```bash
# 기능:
- 자동 Phase 1/2 결과 감지
- 모든 설정 매개변수 지원
- 에러 처리 및 로깅

# 사용법:
bash scripts/run_phase3.sh  # 기본값
bash scripts/run_phase3.sh /path/to/basis /path/to/masks  # 수동 지정
```

### 4. `scripts/phase3_examples.sh` (200줄)
**Phase 3 예제 모음**

```
- Example 1: Minimal configuration
- Example 2: Full configuration
- Example 3: Quick test (< 5 min)
- Example 4: Memory-efficient settings
- Example 5: Different learning rates
- Example 6: Different epochs
- Example 7: Debug mode
- Example 8: Auto-detect
```

### 5. `scripts/run_evaluation.sh` (85줄)
**안전성 평가 스크립트**

```bash
# 기능:
- Phase 3 모델 평가
- Safety ASR 측정
- Utility Accuracy 측정
- 결과를 JSON으로 저장

# 사용법:
bash scripts/run_evaluation.sh  # 최신 모델 자동 감지
bash scripts/run_evaluation.sh /path/to/model  # 수동 지정
```

### 6. `train.py` 수정
**추가 인자:**
```python
--masks_dir           # Phase 2 masks 경로
--utility_samples     # GSM8K 샘플 수 (default: 1000)
--epochs              # 훈련 에포크 (default: 3)
--learning_rate       # 학습률 (default: 5e-5)
--weight_decay        # Weight decay (default: 0.01)
```

**Phase 3 실행 함수:**
```python
run_phase3(args, logger)  # 전체 orchestration
```

---

## 🚀 사용 방법

### Quick Start
```bash
# Phase 1: Basis 구축 (2-5분)
bash scripts/run_phase1.sh

# Phase 2: Importance 점수 계산 (5-10분)
bash scripts/run_phase2.sh

# Phase 3: 미세조정 (10-30분)
bash scripts/run_phase3.sh

# 평가 (5-10분)
bash scripts/run_evaluation.sh
```

### 전체 파이프라인 예시
```bash
# Phase 1
python train.py --phase 1 --safety_samples 100 --epochs 1

# Phase 2
PHASE1_DIR=./checkpoints/phase1_*/checkpoints/basis
python train.py --phase 2 \
    --basis_dir $PHASE1_DIR \
    --safety_samples 100

# Phase 3
PHASE2_DIR=./checkpoints/phase2_*/checkpoints/masks
python train.py --phase 3 \
    --basis_dir $PHASE1_DIR \
    --masks_dir $PHASE2_DIR \
    --utility_samples 1000 \
    --epochs 3

# 평가
bash scripts/run_evaluation.sh
```

---

## 📊 예상 결과

### Phase 3 미세조정 후 (가정)
```
Training:
  - Epoch 1: Loss: 2.1234
  - Epoch 2: Loss: 1.9876
  - Epoch 3: Loss: 1.8765
  
Checkpoints saved:
  - phase3_epoch_000.pt
  - phase3_epoch_001.pt
  - phase3_epoch_002.pt (best)
```

### 평가 결과 (예상)
```
Safety Metrics:
  - Safety ASR: 0.12 (12% 공격 성공률, 목표: < 15%)
  - Safety Rate: 0.88 (88% 거절률, 목표: > 85%)

Utility Metrics:
  - GSM8K Accuracy: 0.45 (45%, 목표: > 40%)

Comparison:
  - Baseline (no masking): Safety ASR=0.85, Utility=0.65
  - With Masking (Phase 3): Safety ASR=0.12, Utility=0.45
  → Safety 크게 개선, Utility는 약간 감소하지만 안전
```

---

## 🔧 기술 세부사항

### Masking Hook 메커니즘
```
Forward Pass (정상):
  output = model(input)
  loss = compute_loss(output, target)

Backward Pass (마스킹 적용):
  loss.backward()
  → gradient 계산
  
Hook에서 gradient 변조:
  grad_new = grad * (1 - mask)
  → mask=1인 곳: grad=0 (업데이트 금지)
  → mask=0인 곳: grad 유지 (정상 업데이트)
  
Optimizer Step (정상):
  param = param - lr * grad_new
  → 중요 파라미터 보호됨
```

### GSM8K 데이터 포맷
```python
# 각 샘플:
{
  'question': "If there are 3 cars...",
  'answer': "If there are 3 cars...\n#### 15"
}

# 훈련 시퀀스:
"Q: If there are 3 cars...
A: If there are 3 cars...
#### 15"

# Loss: 이 전체 시퀀스에 대한 next-token prediction loss
```

### 메모리 최적화
```python
# 주요 최적화:
1. bfloat16 사용 (float16 대신, 안정성)
2. Gradient accumulation (큰 배치 시뮬레이션)
3. 배치 크기 조정 가능 (1 ~ 8)
4. 필요시 gradient checkpointing (추후 추가 가능)

# 메모리 사용량 (추정):
- Model: ~16GB (LLaMA 3 8B in bfloat16)
- Batch: 2-4GB (batch_size=2-4)
- Optimizer states: ~16GB (AdamW)
- 총합: ~40GB (권장 40GB+ GPU)
```

---

## 📝 다음 단계

### Phase 3 완료 후:

1. **평가 실행**
   ```bash
   bash scripts/run_evaluation.sh
   ```

2. **결과 분석**
   - `logs/evaluation_results.json` 확인
   - Safety vs Utility 트레이드오프 확인

3. **하이퍼파라미터 조정**
   - `--learning_rate` 조정
   - `--epochs` 증가
   - `--weight_decay` 조정

4. **모델 배포**
   - 최고 성능 체크포인트 로드
   - 추론 서버 배포

---

## 🐛 일반적인 문제 및 해결

### Issue 1: CUDA Out of Memory
```bash
# 해결: 배치 크기 감소
python train.py --phase 3 ... --batch_size 1
```

### Issue 2: Loss가 증가함
```bash
# 해결: 학습률 감소
python train.py --phase 3 ... --learning_rate 1e-5
```

### Issue 3: 훈련이 너무 느림
```bash
# 해결: 배치 크기 증가 (메모리 허락 시)
python train.py --phase 3 ... --batch_size 4
```

---

## 📚 참고 문헌

### 마스킹 기반 학습
- 참고: Phase 1, 2에서 식별한 안전 중요 방향 보호
- 목표: Safety ↑, Utility ↑ (불가능하면 Safety 우선)

### GSM8K 평가
- Dataset: OpenAI GSM8K (Grade School Math 8K)
- Task: Multi-step math reasoning
- Metric: Exact match on final answer

### 안전성 평가
- 거절 판정: 키워드 기반 (한글/영문)
- 데이터: LibrAI/do-not-answer validation set
- Metric: Attack Success Rate (ASR)

---

**이제 완전한 Safety-WaRP-LLM 3-Phase 파이프라인이 완성되었습니다!** ✅
