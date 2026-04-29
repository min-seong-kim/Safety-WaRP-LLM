#!/usr/bin/env python3
"""
eval log 파일에서 모델/태스크별 결과 테이블만 추출하여 summary 파일로 저장합니다.

사용법:
    python extract_results.py                        # logs/ 에서 가장 최근 eval_*.log 처리
    python extract_results.py logs/eval_XXX.log      # 특정 log 파일 처리
    python extract_results.py --all                  # logs/ 의 모든 eval_*.log 처리

python extract_results.py logs/eval_20260406_222244.log

"""
import re
import sys
from pathlib import Path


def extract_results(log_file: Path):
    """로그 파일을 파싱하여 (model, task, table_lines) 리스트를 반환합니다."""
    lines = log_file.read_text(encoding="utf-8").splitlines()

    results = []
    for i, line in enumerate(lines):
        if "✓ 완료:" not in line:
            continue
        match = re.search(r"✓ 완료: (.+) \(task: (.+)\)", line)
        if not match:
            continue
        model = match.group(1).strip()
        task = match.group(2).strip()

        # ✓ 완료 바로 위에서 거슬러 올라가며 | 로 시작하는 테이블 행을 수집
        table_lines = []
        j = i - 1
        while j >= 0:
            prev = lines[j].rstrip()
            if prev.startswith("|"):
                table_lines.insert(0, prev)
            elif prev.strip() == "":
                pass  # 빈 줄은 건너뜀
            else:
                break  # 테이블이 아닌 내용을 만나면 중단
            j -= 1

        if table_lines:
            results.append((model, task, table_lines))

    return results


def write_summary(log_file: Path, results):
    out_file = log_file.parent / f"{log_file.stem}_summary.txt"
    lines = [f"=== Results from: {log_file.name} ===\n"]
    for model, task, table_lines in results:
        lines.append(f"[{task}] {model}")
        lines.extend(table_lines)
        lines.append("")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def process(log_file: Path):
    results = extract_results(log_file)
    if not results:
        print(f"  결과 없음: {log_file.name}")
        return

    out_file = write_summary(log_file, results)
    print(f"  저장됨: {out_file.name}  ({len(results)}건)")

    # 터미널에도 출력
    for model, task, table_lines in results:
        print(f"\n[{task}] {model}")
        for tl in table_lines:
            print(tl)


def main():
    log_dir = Path("logs")
    args = sys.argv[1:]

    if "--all" in args:
        log_files = sorted(log_dir.glob("eval_*.log"))
        if not log_files:
            print("logs/ 에서 eval_*.log 파일을 찾을 수 없습니다.")
            sys.exit(1)
        for lf in log_files:
            print(f"\n처리 중: {lf.name}")
            process(lf)
    elif args:
        process(Path(args[0]))
    else:
        # 가장 최근 log 파일 하나만 처리
        log_files = sorted(log_dir.glob("eval_*.log"))
        if not log_files:
            print("logs/ 에서 eval_*.log 파일을 찾을 수 없습니다.")
            sys.exit(1)
        latest = log_files[-1]
        print(f"처리 중: {latest.name}")
        process(latest)


if __name__ == "__main__":
    main()
