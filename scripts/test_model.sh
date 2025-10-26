#!/bin/bash

# Phase 3 모델 테스트 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}  WaRP LLaMA3 8B Model Test Script${NC}"
echo -e "${YELLOW}================================================${NC}\n"

# Phase 3 체크포인트 찾기
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 최신 Phase 3 결과 찾기
if [ -z "$PHASE3_PATH" ]; then
    PHASE3_DIR=$(ls -td checkpoints/phase3_* 2>/dev/null | head -1)
    if [ -z "$PHASE3_DIR" ]; then
        echo -e "${RED}❌ No Phase 3 checkpoints found!${NC}"
        echo "Please run Phase 3 first:"
        echo "  python train.py -phase 3 ..."
        exit 1
    fi
    MODEL_PATH="$PHASE3_DIR/checkpoints/checkpoints/phase3_best.pt"
else
    MODEL_PATH="$PHASE3_PATH"
fi

echo -e "${GREEN}✓ Model path: $MODEL_PATH${NC}"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo -e "${GREEN}✓ Model size: $MODEL_SIZE${NC}\n"

# Python 환경 확인
echo -e "${YELLOW}Checking Python environment...${NC}"
python --version

# 필요한 패키지 확인
for package in torch transformers; do
    python -c "import $package; print(f'✓ $package installed')" || {
        echo -e "${RED}❌ $package not installed${NC}"
        exit 1
    }
done

echo -e "\n${YELLOW}================================================${NC}"
echo -e "${YELLOW}  Starting Model Tests${NC}"
echo -e "${YELLOW}================================================${NC}\n"

# 테스트 모드 결정
if [ "$1" = "batch" ]; then
    echo -e "${YELLOW}Running BATCH TEST with predefined queries...${NC}\n"
    python test_model.py \
        --model_path "$MODEL_PATH" \
        --batch \
        --device cuda
elif [ -n "$1" ]; then
    echo -e "${YELLOW}Running SINGLE QUERY TEST...${NC}\n"
    echo -e "${GREEN}Query: $1${NC}\n"
    python test_model.py \
        --model_path "$MODEL_PATH" \
        --query "$1" \
        --device cuda
else
    echo -e "${YELLOW}Running INTERACTIVE TEST...${NC}\n"
    echo "Available options:"
    echo "  1. Batch test (predefined queries)"
    echo "  2. Single query test"
    echo "  3. Custom query"
    echo ""
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            python test_model.py \
                --model_path "$MODEL_PATH" \
                --batch \
                --device cuda
            ;;
        2)
            echo ""
            echo "Predefined queries:"
            echo "  - 'What is machine learning?'"
            echo "  - 'If a train travels at 60 mph for 2 hours, how far does it travel?'"
            echo "  - 'Write a haiku about autumn.'"
            echo ""
            read -p "Enter query: " query
            python test_model.py \
                --model_path "$MODEL_PATH" \
                --query "$query" \
                --device cuda
            ;;
        3)
            read -p "Enter your custom query: " custom_query
            python test_model.py \
                --model_path "$MODEL_PATH" \
                --query "$custom_query" \
                --device cuda
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            exit 1
            ;;
    esac
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}  Test Completed!${NC}"
echo -e "${GREEN}================================================${NC}\n"

# 결과 파일 확인
if [ -f "test_results.json" ]; then
    echo -e "${GREEN}✓ Results saved to: test_results.json${NC}"
fi
