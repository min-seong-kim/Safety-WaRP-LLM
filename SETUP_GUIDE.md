# Environment Setup Guide

## Python 버전 선택 가이드

### 🟢 권장: Python 3.11
```bash
conda create -n safety-warp python=3.11 -y
```

**이유:**
- ✅ PyTorch 최적 지원
- ✅ 대부분의 라이브러리 호환성 우수
- ✅ 성능 우수 (3.10대비 ~10% 향상)
- ✅ 안정성과 최신성의 균형
- ✅ 2025년까지 지원 보장

---

### 🟢 대안: Python 3.12
```bash
conda create -n safety-warp python=3.12 -y
```

**장점:**
- 최신 Python 버전
- 성능 추가 개선
- 더 나은 타입 힌팅

**주의:**
- 일부 라이브러리 호환성 확인 필요
- 실험적 기능 포함 가능

---

### 🟡 과거 버전: Python 3.10
```bash
conda create -n safety-warp python=3.10 -y
```

**상황:**
- 기존 환경과 호환성 필요시
- 레거시 시스템

**문제점:**
- 최신 패키지 지원 감소
- 성능 저하

---

## 전체 설정 프로세스

### 1단계: Conda 환경 생성

```bash
# Python 3.11로 환경 생성 (권장)
conda create -n safety-warp python=3.11 -y

# 또는 environment.yml 사용
conda env create -f environment.yml
```

### 2단계: 환경 활성화

```bash
conda activate safety-warp
```

### 3단계: PyTorch 설치 (필수)

**Option A: CUDA 11.8 (권장)**
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Option B: CUDA 12.1**
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Option C: CPU only (개발 테스트용)**
```bash
conda install pytorch cpuonly -c pytorch -y
```

### 4단계: 프로젝트 의존성 설치

```bash
cd /path/to/Safety-WaRP-LLM
pip install -r requirements.txt
```

### 5단계: 설치 검증

```bash
# PyTorch 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 핵심 패키지 확인
python -c "from transformers import AutoModelForCausalLM; print('✓ Transformers OK')"
python -c "from datasets import load_dataset; print('✓ Datasets OK')"
python -c "import peft; print('✓ PEFT OK')"
```

---

## 환경별 설정 예제

### 개발 머신 (RTX 4090, 24GB)

```bash
# 환경 생성
conda create -n safety-warp python=3.11 -y
conda activate safety-warp

# PyTorch (CUDA 11.8)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 의존성
pip install -r requirements.txt

# Phase 1 실행 (메모리 최적화)
python train.py --phase 1 \
    --safety_samples 50 \
    --batch_size 2 \
    --dtype float16 \
    --device cuda:0
```

### 고성능 GPU (A100/H100, 40GB+)

```bash
# 환경 생성
conda create -n safety-warp python=3.11 -y
conda activate safety-warp

# PyTorch (CUDA 12.1 - 최신 GPU 권장)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 의존성
pip install -r requirements.txt

# Phase 1 실행 (최적 설정)
python train.py --phase 1 \
    --safety_samples 500 \
    --batch_size 8 \
    --dtype bfloat16 \
    --device cuda:0
```

### 서버 (다중 GPU)

```bash
# 환경 생성
conda create -n safety-warp python=3.11 -y
conda activate safety-warp

# PyTorch with CUDA
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 의존성
pip install -r requirements.txt

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Phase 1 실행
python train.py --phase 1 \
    --safety_samples 1000 \
    --batch_size 16 \
    --dtype bfloat16 \
    --device cuda:0
```

### 개발 (CPU only)

```bash
# 환경 생성
conda create -n safety-warp python=3.11 -y
conda activate safety-warp

# PyTorch CPU
conda install pytorch cpuonly -c pytorch -y

# 의존성
pip install -r requirements.txt

# Phase 1 실행 (매우 느림 - 디버깅용)
python train.py --phase 1 \
    --safety_samples 10 \
    --batch_size 1 \
    --device cpu \
    --debug
```

---

## 환경 관리 팁

### 환경 목록 확인
```bash
conda env list
```

### 환경 정보 확인
```bash
conda info -e
```

### 환경 복제
```bash
conda create --clone safety-warp -n safety-warp-backup
```

### 환경 삭제
```bash
conda remove -n safety-warp --all
```

### 패키지 업그레이드
```bash
conda activate safety-warp
pip install --upgrade -r requirements.txt
```

### requirements.txt 생성 (필요시)
```bash
conda activate safety-warp
pip freeze > requirements_frozen.txt
```

---

## 문제 해결

### CUDA 버전 불일치

```bash
# 현재 CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# NVIDIA CUDA 버전 확인
nvidia-smi

# 해결: PyTorch 재설치
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y --force-reinstall
```

### 메모리 부족

```bash
# CUDA 캐시 정리
python -c "import torch; torch.cuda.empty_cache()"

# 재설치 시 메모리 최적화 옵션
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### ImportError

```bash
# 환경이 활성화되었는지 확인
which python

# 재설치
pip install -r requirements.txt --force-reinstall --no-cache-dir
```

---

## Version Information

- **Created**: October 2025
- **Python Recommended**: 3.11
- **PyTorch**: 2.0+
- **CUDA**: 11.8 또는 12.1

---

## 빠른 체크리스트

- [ ] Conda 설치 확인
- [ ] 환경 생성 (`conda create -n safety-warp python=3.11`)
- [ ] 환경 활성화 (`conda activate safety-warp`)
- [ ] PyTorch 설치 (`conda install pytorch pytorch-cuda=11.8...`)
- [ ] 의존성 설치 (`pip install -r requirements.txt`)
- [ ] 설치 검증 (위의 검증 명령어 실행)
- [ ] Phase 1 테스트 실행

모두 완료되면 `bash scripts/run_phase1.sh`로 Phase 1을 시작하세요!
