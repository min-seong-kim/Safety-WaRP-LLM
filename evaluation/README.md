# AG News 학습 및 평가

이 디렉터리는 AG News fine-tuning과 평가 코드만 정리합니다.

사용 예정 모델:

- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-3.1-8B-Instruct`

base 모델은 plain instruction prompt로 학습/평가하고, instruct/chat 모델은 tokenizer의 chat template을 사용합니다. Llama-2-chat은 chat template이 없을 경우 `[INST] ... [/INST]` fallback을 사용합니다.

## 파일

- `finetune_swebench_agnews_full_params.py`: AG News full-parameter 또는 LoRA SFT
- `evaluate_agnews_hf.py`: AG News test JSONL 평가

## Implementation

AG News train/test JSONL:

```text
/home/users/jongbokwon/dataset/agnews_qwen/agnews_qwen2_5_7b_instruct_train_8000.jsonl
/home/users/jongbokwon/dataset/agnews_qwen/agnews_qwen2_5_7b_instruct_test_200.jsonl
```

같은 instruct-format JSONL을 Qwen base, Qwen instruct, Llama-2 chat, Llama-3.1 instruct 모델에 모두 사용할 수 있습니다. 학습 스크립트는 모델 이름에 `instruct` 또는 `chat`이 있으면 chat 형식으로, 아니면 base prompt 형식으로 자동 변환합니다.

### 학습

```bash
cd /home/users/jongbokwon/Safety-WaRP-LLM/evaluation

python finetune_swebench_agnews_full_params.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --agnews_train_path /home/users/jongbokwon/dataset/agnews_qwen/agnews_qwen2_5_7b_instruct_train_8000.jsonl \
  --output_dir ./outputs/agnews/qwen25_7b_instruct \
  --num_train_samples 8000 \
  --learning_rate 5e-5 \
  --epochs 3 \
  --max_length 1024 \
  --report_to none
```

모델만 바꿔 실행하면 됩니다.

```bash
--model_path Qwen/Qwen2.5-7B
--model_path Qwen/Qwen2.5-7B-Instruct
--model_path meta-llama/Llama-2-7b-chat-hf
--model_path meta-llama/Llama-3.1-8B-Instruct
```

LoRA로 학습하려면 `--lora`를 추가합니다.

### 평가

```bash
python evaluate_agnews_hf.py \
  --model_name_or_path ./outputs/agnews/qwen25_7b_instruct \
  --prompt_format auto \
  --test_jsonl_path /home/users/jongbokwon/dataset/agnews_qwen/agnews_qwen2_5_7b_instruct_test_200.jsonl \
  --output_path ./predictions/agnews/qwen25_7b_instruct_predictions.jsonl
```

`--prompt_format auto`는 모델 이름을 기준으로 base/chat prompt를 자동 선택합니다. AG News 학습과 평가는 Docker가 필요 없습니다.
