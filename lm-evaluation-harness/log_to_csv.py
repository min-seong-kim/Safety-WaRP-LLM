#!/usr/bin/env python3
"""
eval log 파일을 파싱하여 CSV로 저장합니다.

행(row): gsm8k_flexible, arc_c_acc_norm, mmlu_acc
열(column): 모델명

사용법:
    python log_to_csv.py                          # logs/ 에서 가장 최근 eval_*.log 처리
    python log_to_csv.py logs/eval_XXX.log        # 특정 log 파일 처리
    python log_to_csv.py logs/eval_20260417_210135_results.log
"""

import re
import sys
import csv
from pathlib import Path


def parse_log(log_file: Path):
    """
    로그 파일을 파싱하여 {model: {metric: value}} 딕셔너리 반환.
    metric 키: "gsm8k_flexible", "arc_c_acc_norm", "mmlu_acc"
    """
    lines = log_file.read_text(encoding="utf-8").splitlines()
    results = {}  # model -> {metric: value}
    model_order = []

    for i, line in enumerate(lines):
        if "✓ 완료:" not in line:
            continue
        match = re.search(r"✓ 완료: (.+) \(task: (.+)\)", line)
        if not match:
            continue
        model = match.group(1).strip()
        task = match.group(2).strip()

        if model not in results:
            results[model] = {}
            model_order.append(model)

        # ✓ 완료 위의 테이블 행 수집
        table_lines = []
        j = i - 1
        while j >= 0:
            prev = lines[j].rstrip()
            if prev.startswith("|"):
                table_lines.insert(0, prev)
            elif prev.strip() == "":
                pass
            else:
                break
            j -= 1

        if task == "gsm8k":
            # flexible-extract exact_match 행 찾기
            for tl in table_lines:
                if "flexible-extract" in tl:
                    val, err = _extract_value(tl)
                    if val is not None:
                        results[model]["gsm8k_flexible"] = _fmt(val, err)

        elif task == "arc-c":
            # acc_norm 행 찾기
            for tl in table_lines:
                if "acc_norm" in tl:
                    val, err = _extract_value(tl)
                    if val is not None:
                        results[model]["arc_c_acc_norm"] = _fmt(val, err)
                        break

        elif task == "mmlu":
            # |mmlu | ... |acc| 의 전체 평균 행 찾기 (Groups 테이블의 최상단)
            for tl in table_lines:
                if re.match(r"\|mmlu\s*\|", tl) and "acc" in tl:
                    val, err = _extract_value(tl)
                    if val is not None:
                        results[model]["mmlu_acc"] = _fmt(val, err)
                        break

        elif task == "hendrycks_math_safe":
            # hendrycks_math_safe exact_match 행 찾기
            for tl in table_lines:
                if "exact_match" in tl:
                    val, err = _extract_value(tl)
                    if val is not None:
                        results[model]["math_exact_match"] = _fmt(val, err)
                        break

    return model_order, results


def _extract_value(table_row: str):
    """테이블 행에서 (value, stderr) 튜플 추출."""
    # 형식: |...|...|...|...|...|↑  |0.6072|±  |0.0039|
    parts = [p.strip() for p in table_row.split("|")]
    for k, p in enumerate(parts):
        if p in ("↑", "↓") and k + 1 < len(parts):
            try:
                val = float(parts[k + 1])
                # ± 다음이 stderr
                err = None
                if k + 3 < len(parts) and parts[k + 2] in ("±", ""):
                    try:
                        err = float(parts[k + 3])
                    except ValueError:
                        pass
                return val, err
            except ValueError:
                pass
    return None, None


def _fmt(val: float, err: float | None) -> str:
    """값±오차 형식 문자열 반환."""
    if err is not None:
        return f"{val:.4f}±{err:.4f}"
    return f"{val:.4f}"


def write_csv(log_file: Path, model_order: list, results: dict):
    # 실제로 결과에 등장한 metric만 포함 (등장 순서 고정)
    all_metrics = ["gsm8k_flexible", "arc_c_acc_norm", "mmlu_acc", "math_exact_match"]
    present = {m for r in results.values() for m in r}
    metrics = [m for m in all_metrics if m in present]
    out_file = log_file.parent / f"{log_file.stem}_table.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 헤더: model | metric1 | metric2 | ...
        writer.writerow(["model"] + metrics)
        for model in model_order:
            row = [model]
            for metric in metrics:
                val = results.get(model, {}).get(metric, "")
                row.append(val)
            writer.writerow(row)

    return out_file


def main():
    args = sys.argv[1:]
    log_dir = Path("logs")

    if args:
        log_files = [Path(args[0])]
    else:
        candidates = sorted(log_dir.glob("eval_*.log"))
        if not candidates:
            print("logs/ 에 eval_*.log 파일이 없습니다.")
            sys.exit(1)
        log_files = [candidates[-1]]

    for log_file in log_files:
        if not log_file.exists():
            print(f"파일 없음: {log_file}")
            continue
        model_order, results = parse_log(log_file)
        if not results:
            print(f"결과 없음: {log_file.name}")
            continue
        out = write_csv(log_file, model_order, results)
        print(f"저장됨: {out}")

        # 터미널 미리보기
        all_metrics = ["gsm8k_flexible", "arc_c_acc_norm", "mmlu_acc", "math_exact_match"]
        present = {m for r in results.values() for m in r}
        metrics = [m for m in all_metrics if m in present]
        col_w = max(len(m) for m in model_order) + 2
        print(f"\n{'metric':<20}", end="")
        for m in model_order:
            short = m.split("/")[-1][:20]
            print(f"{short:>22}", end="")
        print()
        for metric in metrics:
            print(f"{metric:<20}", end="")
            for m in model_order:
                val = results.get(m, {}).get(metric, "N/A")
                print(f"{str(val):>22}", end="")
            print()


if __name__ == "__main__":
    main()
