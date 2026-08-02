"""로컬 JSON 다운스트림 태스크(SST-2 / AG News) 로더.

`data/sst2_train_8k_seed42.json`, `data/agnews_train_8k_seed42.json` 처럼
`scripts/`(subsets_seed42.manifest.json) 로 고정 샘플링해 둔 분류 데이터셋을
LISA / LoRA-family 러너에서 GSM8K 대신 쓰기 위한 공용 로더다.

기대하는 행 스키마 (둘 중 하나면 된다):
  1) {"question": <프롬프트 전문>, "response": <정답 텍스트>}   ← 우선
  2) {"instruction": ..., "input": ..., "output": ...}          ← fallback 으로 조립

GSM8K 경로와 동일하게 `tokenize_sft_example(question, answer, ...)` 로 넘겨
프롬프트 구간을 -100 마스킹한 SFT 형식으로 토큰화한다. 즉 이 로더가 바꾸는 것은
"어떤 (질문, 정답) 쌍을 쓰는가" 뿐이고, 토크나이즈/콜레이터/학습 로직은 손대지 않는다.
"""
import json
import os
from typing import Dict, List, Tuple

# manifest 에 등록된 파일명 → 태스크 이름. 편의용이며 강제는 아니다.
KNOWN_TASKS = {
    "sst2": "data/sst2_train_8k_seed42.json",
    "agnews": "data/agnews_train_8k_seed42.json",
    # QA 계열 (scripts/prepare_qa_task_data.py 로 생성). 프롬프트/정답 포맷은 각각
    # arc_eval / medqa_eval 하네스에서 그대로 가져오므로 평가와 포맷이 일치한다.
    "arc": "data/arc_challenge_train_task_1119.json",
    "medqa": "data/medqa_train_task_10178.json",
}


def _as_text(value) -> str:
    return "" if value is None else str(value)


def _row_to_pair(row: Dict) -> Tuple[str, str]:
    """한 행에서 (프롬프트, 정답) 을 뽑는다."""
    question = _as_text(row.get("question")).strip()
    response = _as_text(row.get("response")).strip()
    if question and response:
        return question, response

    # fallback: instruction/input/output 조립 (question/response 가 없는 변형 스키마)
    instruction = _as_text(row.get("instruction")).strip()
    input_text = _as_text(row.get("input")).strip()
    output = _as_text(row.get("output") or row.get("label_text")).strip()
    if instruction and input_text:
        question = f"{instruction}\n\nInput:\n{input_text}"
    else:
        question = instruction or input_text
    if not question or not output:
        raise ValueError(
            "행에서 (question, response) 를 만들 수 없습니다. "
            f"사용 가능한 키: {sorted(row.keys())}")
    return question, output


def load_task_pairs(path: str, max_samples: int = 0) -> List[Tuple[str, str]]:
    """JSON 파일 → [(question, response), ...]. max_samples<=0 이면 전체."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"task dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: 최상위가 list 여야 합니다 (got {type(rows).__name__})")
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    return [_row_to_pair(r) for r in rows]


def build_task_dataset(path, tokenizer, max_length, model_ref,
                       tokenize_fn, max_samples: int = 0, desc: str = "tokenizing task data"):
    """JSON → 토큰화된 HF Dataset.

    tokenize_fn 은 각 러너가 이미 쓰고 있는 `tokenize_sft_example` 을 그대로 넘긴다
    (러너마다 import 경로가 달라 여기서 직접 import 하지 않는다).
    """
    from datasets import Dataset as HFDataset

    pairs = load_task_pairs(path, max_samples)
    ds = HFDataset.from_list([{"question": q, "response": a} for q, a in pairs])

    def preprocess(ex):
        return tokenize_fn(ex["question"], ex["response"], tokenizer, max_length, model_ref)

    return ds.map(preprocess, remove_columns=ds.column_names, desc=desc)


def infer_task_name(path: str) -> str:
    """파일명에서 태스크 이름 추정 (로그/summary 기록용)."""
    base = os.path.basename(str(path)).lower()
    for name in KNOWN_TASKS:
        if name in base:
            return name
    return os.path.splitext(base)[0]
