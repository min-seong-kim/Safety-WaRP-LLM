#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  revision 실험이 만들어낼 HF 리포 전체 목록을 마크다운으로 출력한다.
#
#  ⚠️ 손으로 목록을 관리하지 않는다. 리포명은 common.sh 의 hf_repo_id() /
#     hf_ssft_repo_id() 한 곳에서만 생성되며, 이 스크립트는 그것을 그대로 부른다.
#     하이퍼파라미터(KEEP_RATIO, SAFELORA_THRESHOLD, ...)를 바꾸면 이름도 바뀌므로
#     바꾼 뒤에는 반드시 다시 돌려서 REPO_LIST.md 를 갱신할 것.
#
#  사용:
#    bash scripts/revision/gen_repo_list.sh > scripts/revision/REPO_LIST.md
#    bash scripts/revision/gen_repo_list.sh | head -40      # 미리보기
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TASK_LABEL() { case "$1" in
  gsm8k) echo "GSM8K" ;; math) echo "MATH" ;; medqa) echo "MedQA" ;;
  arc) echo "ARC-C" ;; agnews) echo "AG News" ;; *) echo "$1" ;; esac; }

METHOD_LABEL() { case "$1" in
  fullft) echo "Full FT" ;; safeinstr) echo "SafeInstr" ;; resta) echo "RESTA" ;;
  safedelta) echo "SafeDelta" ;; wsr_tune) echo "WSR-Tune" ;; lora) echo "Vanilla LoRA" ;;
  asft) echo "AsFT" ;; lisa) echo "LISA" ;; seal) echo "SEAL" ;;
  safelora) echo "SafeLoRA" ;; salora) echo "SaLoRA" ;; wsr_lora) echo "WSR-LoRA" ;;
  *) echo "$1" ;; esac; }

# ── 개수 집계 ──
n_cell=0; n_ssft=0; n_pub=0
for mkey in $MODELS; do
  model_cfg "$mkey" || continue
  if safety_applies bt "$mkey" && [[ -z "$ALIGNED_BT" ]]; then n_ssft=$((n_ssft+1)); fi
  for safety in $SAFETY_SETS; do
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      for m in $METHODS; do
        if cell_wanted "$safety" "$mkey" "$task" "$m"; then n_cell=$((n_cell+1))
        else n_pub=$((n_pub+1)); fi
      done
    done
  done
done

cat <<EOF
# Revision 실험 — 생성될 Hugging Face 리포 전체 목록

> **자동 생성 파일이다. 직접 고치지 말 것.**
> \`bash scripts/revision/gen_repo_list.sh > scripts/revision/REPO_LIST.md\` 로 재생성한다.
> 리포명은 \`scripts/revision/common.sh\` 의 \`hf_repo_id()\` / \`hf_ssft_repo_id()\` 가 유일한
> 생성처다. 하이퍼파라미터를 바꾸면 이름도 바뀌므로 반드시 다시 생성할 것.

| | |
|---|---|
| 네임스페이스 | \`${HF_NAMESPACE}\` |
| **새로 만들 학습 셀** | **${n_cell}개** |
| 논문/rebuttal 에 이미 있어 건너뛰는 셀 | ${n_pub}개 |
| BT 안전정렬 출발모델 리포 | ${n_ssft}개 |
| **합계 (새로 생성)** | **$((n_cell + n_ssft))개** |

### 이번 범위

| | |
|---|---|
| CB(Circuit Breakers) 축 | 모델 6종 전부 |
| BT(BeaverTails) 축 | **\`${BT_MODELS}\` 만**, 태스크 \`${BT_TASKS}\` 전부 · **12기법 전부 신규** |
| 재사용 (\`SKIP_PUBLISHED=${SKIP_PUBLISHED}\`) | **논문 Table 2/4/10 의 full-param 5종만** — 출발모델·lr·epoch·배치가 이번 설정과 일치함을 확인한 것뿐이다 |
| 재사용하지 않는 것 | rebuttal 의 PEFT 수치. 출발모델 sha256 불일치(\`wvnvwn/...ssft-cb\` vs \`kmseong/...Safety-FT-lr5e-5\`), SafeLoRA thr 0.3 미실행(0.15/0.25/0.35 만), AGNEWS 는 epoch1·lr1e-5 동작점 |
| 알려진 잔여 차이 | 논문 MedQA 는 10000 샘플, 이번 신규 셀은 전체 10178 (1.7%). 재사용하는 MedQA 기준행 5개만 해당 |

---

## 명명 규칙

\`\`\`
${HF_NAMESPACE}/{model}-{CB|BT}_SSFT-{method}_{task}[_{hparam}]_lr{lr}
\`\`\`

| 필드 | 값 |
|---|---|
| \`model\` | \`llama2_7b-chat\` · \`llama2_13b-chat\` · \`llama3_2_3b-instruct\` · \`llama3_1_8b-instruct\` · \`qwen2_5_7b-instruct\` · \`gemma2_9b-it\` |
| \`CB\`/\`BT\` | 출발 모델을 안전정렬한 데이터셋 (Circuit Breakers / BeaverTails) |
| \`method\` | 12종. 기법명의 \`_\` 는 \`-\` 로 바꾼다 (\`wsr_tune\` → \`wsr-tune\`) — 필드 구분자 \`_\` 와 충돌하므로 |
| \`task\` | \`gsm8k\` \`math\` \`medqa\` \`arc\` \`agnews\` |
| \`hparam\` | 기법 고유 값. **없는 기법(Full FT / Vanilla LoRA)은 슬롯을 생략한다** |
| \`lr\` | 기법에서 자동 도출. full-param 계열 \`${FULL_LR}\`, LoRA 계열 \`$(lora_lr gsm8k)\` (AG News 만 \`$(lora_lr agnews)\`) |

### 기법별 하이퍼파라미터 태그

| 기법 | 태그 | 값의 근거 |
|---|---|---|
EOF
for m in $METHODS; do
  hp="$(hf_hparam_tag "$m")"
  case "$m" in
    fullft|lora) why="하이퍼파라미터 없음 → 슬롯 생략" ;;
    safeinstr) why="논문 §4.1 (downstream 학습셋의 10%)" ;;
    resta)     why="논문 §4.1" ;;
    safedelta) why="논문 §4.1" ;;
    wsr_tune|wsr_lora) why="논문 기본 freeze ratio ρ" ;;
    asft)      why="사용자 지정 λ" ;;
    lisa)      why="사용자 지정 ρ" ;;
    seal)      why="기존 설정 top-p" ;;
    safelora)  why="사용자 지정 threshold" ;;
    salora)    why="budget-matched 설정" ;;
    *)         why="" ;;
  esac
  printf '| %s | `%s` | %s |\n' "$(METHOD_LABEL "$m")" "${hp:-—}" "$why"
done

cat <<'EOF'

> LoRA 계열 6종은 전부 `r=16 / alpha=32 / dropout=0.05 / targets {q,k,v,up,down}` 로 동일해
> 이름에 넣지 않는다. full-param 계열은 전부 `epochs 3 / effective batch 16 / max_len 1024 /
> seed 42 / bf16`.

---

## 1. 안전정렬 출발 모델 (입력)

이 실험의 모든 셀은 **이미 안전정렬된 모델**에서 출발한다. CB 축 6종과 BT 축 llama2-7b 는
기존 것을 재사용하고, 나머지 BT 5종을 새로 학습해 올린다.

| 모델 키 | base (재사용) | CB 출발모델 (재사용) | BT 출발모델 |
|---|---|---|---|
EOF
for mkey in $MODELS; do
  model_cfg "$mkey" || continue
  if ! safety_applies bt "$mkey"; then bt="— (BT 축 제외)"
  elif [[ -n "$ALIGNED_BT" ]]; then bt="\`$ALIGNED_BT\` (재사용)"
  else bt="**\`$(hf_ssft_repo_id "$mkey" bt)\`** ← 신규"; fi
  printf '| `%s` | `%s` | `%s` | %s |\n' "$mkey" "$BASE" "$ALIGNED_CB" "$bt"
done

cat <<'EOF'

> `base` 는 SafeLoRA / AsFT / RESTA 가 `V = W_aligned − W_base` 를 만들 때 필요하다
> (gated repo 라 HF 토큰 필요). 새로 만드는 리포는 **BT 출발모델 5종뿐**이다.

---

## 2. 학습 셀 리포

모델별로 나눈다. 각 표의 행은 12개 기법, 열은 안전 데이터 축이다.

EOF

for mkey in $MODELS; do
  model_cfg "$mkey" || continue
  mtag="$(hf_model_tag "$mkey")"
  n_this=0
  for safety in $SAFETY_SETS; do
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      for m in $METHODS; do cell_wanted "$safety" "$mkey" "$task" "$m" && n_this=$((n_this+1)); done
    done
  done
  echo "### \`$mkey\`  (\`$mtag\`) — 신규 ${n_this}개"
  echo ""
  for safety in $SAFETY_SETS; do
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      if [[ "$task" == "$PRIMARY_TASK" ]]; then note="Table 2 / Table 4"; else note="Figure 4 확장"; fi
      echo "#### $(hf_safety_tag "$safety") · $(TASK_LABEL "$task")  — $note"
      echo ""
      echo "| 기법 | 리포명 (네임스페이스 생략) | 상태 |"
      echo "|---|---|---|"
      for m in $METHODS; do
        rid="$(hf_repo_id "$safety" "$mkey" "$task" "$m")"
        if cell_wanted "$safety" "$mkey" "$task" "$m"; then st="**신규**"
        else st="기존 (논문/rebuttal)"; fi
        printf '| %s | `%s` | %s |\n' "$(METHOD_LABEL "$m")" "${rid#*/}" "$st"
      done
      echo ""
    done
  done
done

cat <<'EOF'
> 표에는 네임스페이스를 뺀 리포명만 적었다. 실제 id 는 앞에 `NAMESPACE/` 가 붙는다.

---

## 3. 전체 목록 (평문)

**새로 만들 것만** 나열한다. 스크립트에서 그대로 쓰기 좋은 형태.

```
EOF
for mkey in $MODELS; do
  model_cfg "$mkey" || continue
  safety_applies bt "$mkey" && [[ -z "$ALIGNED_BT" ]] && hf_ssft_repo_id "$mkey" bt
done
for mkey in $MODELS; do
  model_cfg "$mkey" || continue
  for safety in $SAFETY_SETS; do
    safety_applies "$safety" "$mkey" || continue
    for task in $(tasks_for_model "$mkey" "$safety"); do
      for m in $METHODS; do
        cell_wanted "$safety" "$mkey" "$task" "$m" && hf_repo_id "$safety" "$mkey" "$task" "$m"
      done
    done
  done
done
cat <<'EOF'
```

---

## 4. 검증

이 목록은 아래를 만족한다 (`gen_repo_list.sh` 가 매번 재확인).

EOF
ALL=$( { for mkey in $MODELS; do model_cfg "$mkey" || continue
           safety_applies bt "$mkey" && [[ -z "$ALIGNED_BT" ]] && hf_ssft_repo_id "$mkey" bt; done
         for mkey in $MODELS; do model_cfg "$mkey" || continue
           for safety in $SAFETY_SETS; do safety_applies "$safety" "$mkey" || continue
             for task in $(tasks_for_model "$mkey" "$safety"); do
               for m in $METHODS; do cell_wanted "$safety" "$mkey" "$task" "$m" && hf_repo_id "$safety" "$mkey" "$task" "$m"; done
             done; done
         done; } )
n_all=$(echo "$ALL" | wc -l)
n_uniq=$(echo "$ALL" | sort -u | wc -l)
maxlen=$(echo "$ALL" | sed 's|^[^/]*/||' | awk '{print length($0)}' | sort -rn | head -1)
bad=$(echo "$ALL" | grep -vcE '^[A-Za-z0-9_-]+/[A-Za-z0-9][A-Za-z0-9._-]*$' || true)
cat <<EOF
- 총 **${n_all}개**, 고유 **${n_uniq}개** — 중복 $(( n_all - n_uniq ))건
- 리포명 최장 **${maxlen}자** (Hugging Face 한도: 네임스페이스 제외 96자)
- HF 허용문자 \`[A-Za-z0-9._-]\` 위반: **${bad}건**

기존 리포와 덮어쓰기 충돌이 없는지는 네트워크가 필요해 별도로 확인한다:

\`\`\`bash
python - <<'PY'
from huggingface_hub import HfApi
import pathlib
existing = {m.id for m in HfApi().list_models(author="${HF_NAMESPACE}")}
block = pathlib.Path("scripts/revision/REPO_LIST.md").read_text().split("## 3. 전체 목록 (평문)")[1].split("\`\`\`")[1]
planned = [l.strip() for l in block.strip().splitlines() if l.strip()]
clash = sorted(set(planned) & existing)
print(f"계획 {len(planned)} · 기존 {len(existing)} · 충돌 {len(clash)}", clash[:5])
PY
\`\`\`

생성 시각: $(date '+%Y-%m-%d %H:%M:%S %Z')
EOF
