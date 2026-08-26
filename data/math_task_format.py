"""
Hendrycks MATH 전처리 — WaRP Phase 3 와 baseline 러너가 **공유**하는 단일 소스.

`models/phase3_extra_learning.py::_load_hendrycks_math` 안에 중첩 함수로 있던 로직을
그대로 끌어올린 것이다. Phase 3 도 이 모듈을 import 하므로, 여기만 고치면
WaRP arm 과 baseline arm 의 학습 텍스트가 절대 어긋나지 않는다.
(arc/medqa 가 `scripts/prepare_qa_task_data.py` 에서 eval 하네스의 프롬프트 빌더를
그대로 import 하는 것과 같은 이유다.)

타깃 포맷:
    long     = "{rationale}\\nFinal Answer: ${answer}$"     (기본, 100%)
    short    = "Final Answer: ${answer}$"                   (mixed 모드에서 20%)
    minimal  = "${answer}$"                                 (mixed 모드에서 10%)
"""

import re
from typing import Optional

# EleutherAI/hendrycks_math 의 subject → config 이름
SUBJECT_TO_CONFIG = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}
VALID_LEVELS = {f"Level {i}" for i in range(1, 6)}

_MULTI_SPACE_RE = re.compile(r"\n{3,}")


def normalize_csv_arg(value) -> str:
    """CSV 인자에서 바깥쪽 따옴표/공백 제거 ('"all"' → 'all')."""
    if value is None:
        return ""
    value = str(value).strip()
    if len(value) >= 2 and (
        (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")
    ):
        value = value[1:-1].strip()
    return value


def last_boxed_only_string(text: str) -> Optional[str]:
    """solution 에서 마지막 \\boxed{...} (또는 \\fbox{...}) 를 통째로 뽑는다."""
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s is None:
        raise ValueError("remove_boxed received None")
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def extract_final_answer_from_solution(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError(
            f"Could not find final boxed answer in solution: {solution[:300]!r}")
    return remove_boxed(boxed).strip()


def clean_solution_for_reasoning(solution: str, final_answer: str) -> str:
    text = solution.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        text = text.replace(boxed, final_answer)

    text = text.replace("$", "")
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("\\boxed", "")
    text = text.replace("\\fbox", "")
    text = _MULTI_SPACE_RE.sub("\n\n", text)
    return text.strip()


def build_target(solution: str, rng=None, train_on_mixed_formats: bool = False) -> str:
    """solution → 학습 타깃 문자열.

    train_on_mixed_formats=False (기본) 이면 항상 long 포맷이라 rng 는 쓰이지 않는다.
    """
    final_answer = extract_final_answer_from_solution(solution)
    rationale = clean_solution_for_reasoning(solution, final_answer)

    long_target = f"{rationale}\nFinal Answer: ${final_answer}$"
    short_target = f"Final Answer: ${final_answer}$"
    minimal_target = f"${final_answer}$"

    if not train_on_mixed_formats:
        return long_target

    if rng is None:
        raise ValueError("train_on_mixed_formats=True 이면 rng 가 필요합니다")
    draw = rng.random()
    if draw < 0.70:
        return long_target
    if draw < 0.90:
        return short_target
    return minimal_target
