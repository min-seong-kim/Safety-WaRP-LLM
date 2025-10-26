#!/bin/bash

# HuggingFace Hub에 모델 업로드하는 스크립트
# 
# Usage:
#   bash scripts/upload_model_to_hf.sh <model_path> <model_id> [token]
#
# Examples:
#   bash scripts/upload_model_to_hf.sh \
#       ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \
#       kmseong/WaRP-Safety-Llama3_8B_Instruct
#
#   # With token
#   bash scripts/upload_model_to_hf.sh \
#       ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \
#       kmseong/WaRP-Safety-Llama3_8B_Instruct \
#       hf_xxxxxxxxxxxxxxxxxxx

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}HuggingFace Hub 모델 업로드${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# 인자 확인
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: 필수 인자가 부족합니다${NC}"
    echo "Usage: bash scripts/upload_model_to_hf.sh <model_path> <model_id> [token]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/upload_model_to_hf.sh \\"
    echo "      ./checkpoints/phase3_20251023_232951/checkpoints/checkpoints/phase3_best.pt \\"
    echo "      kmseong/WaRP-Safety-Llama3_8B_Instruct"
    echo ""
    echo "  # HF_TOKEN 환경변수 설정 후"
    echo "  export HF_TOKEN=your_token"
    echo "  bash scripts/upload_model_to_hf.sh \\"
    echo "      ./checkpoints/phase3_TIMESTAMP/checkpoints/checkpoints/phase3_best.pt \\"
    echo "      kmseong/WaRP-Safety-Llama3_8B_Instruct"
    exit 1
fi

MODEL_PATH="$1"
MODEL_ID="$2"
HF_TOKEN="${3:-}"

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "[Step 1] 설정 확인"
echo "  Model path: $MODEL_PATH"
echo "  Model ID: $MODEL_ID"
if [ -z "$HF_TOKEN" ]; then
    echo "  Token: (환경변수 HF_TOKEN 사용)"
else
    echo "  Token: (직접 전달됨)"
fi
echo ""

# 모델 파일 확인
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: 모델 파일을 찾을 수 없습니다: $MODEL_PATH${NC}"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo -e "${GREEN}✓ 모델 파일 확인: $MODEL_SIZE${NC}"
echo ""

# Python 환경 확인
echo "[Step 2] Python 환경 확인"
python --version
echo ""

# 필요한 패키지 확인
echo "[Step 3] 필수 패키지 확인"
python -c "import torch; print('✓ PyTorch OK')" || { echo -e "${RED}PyTorch 설치 필요${NC}"; exit 1; }
python -c "import transformers; print('✓ Transformers OK')" || { echo -e "${RED}Transformers 설치 필요${NC}"; exit 1; }
python -c "import huggingface_hub; print('✓ HuggingFace Hub OK')" || { 
    echo -e "${YELLOW}HuggingFace Hub 설치 중...${NC}"
    pip install huggingface_hub
}
echo ""

# 업로드 스크립트 실행
echo "[Step 4] 모델 업로드 실행"
echo ""

cd "$PROJECT_ROOT"

if [ -z "$HF_TOKEN" ]; then
    # 환경변수에서 토큰 읽기
    python upload_to_huggingface.py \
        --model_path "$MODEL_PATH" \
        --hf_model_id "$MODEL_ID"
else
    # 직접 전달된 토큰 사용
    python upload_to_huggingface.py \
        --model_path "$MODEL_PATH" \
        --hf_model_id "$MODEL_ID" \
        --hf_token "$HF_TOKEN"
fi

UPLOAD_STATUS=$?

echo ""
if [ $UPLOAD_STATUS -eq 0 ]; then
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}✓ 업로드 완료!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo -e "${GREEN}모델에 접근할 수 있습니다:${NC}"
    echo "  https://huggingface.co/$MODEL_ID"
    echo ""
    echo -e "${YELLOW}다음 단계:${NC}"
    echo "  1. HuggingFace 웹사이트에서 모델 확인"
    echo "  2. README.md 수정 (선택사항)"
    echo "  3. 다른 벤치마크에서 테스트"
    echo ""
else
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}✗ 업로드 실패${NC}"
    echo -e "${RED}================================================${NC}"
    exit 1
fi
