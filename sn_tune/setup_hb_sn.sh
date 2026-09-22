#!/usr/bin/env bash
# =============================================================================
# setup_hb_sn.sh — 원공간 뉴런 검출용 conda 환경 `hb_sn` 을 만든다.
#
#   hb (학습용, 정품 transformers)  --clone-->  hb_sn  --패치 덮어쓰기-->  검출용
#
# 왜 별도 환경인가
#   검출은 patched modeling_*.py 가 있어야만 동작하고, 학습은 **정품**이어야 한다.
#   한 환경에 둘 다 둘 수 없으므로 복제해서 검출 쪽만 패치한다.
#   이 스크립트는 `hb` 를 절대 건드리지 않는다.
#
# 사용
#   bash sn_tune/setup_hb_sn.sh                 # 생성 + llama/gemma2/qwen2 패치
#   FAMILIES="llama" bash sn_tune/setup_hb_sn.sh
#   SRC_ENV=hb_repro DST_ENV=hb_sn_repro bash sn_tune/setup_hb_sn.sh
#   FORCE=1 bash sn_tune/setup_hb_sn.sh         # 이미 있어도 패치를 다시 덮어쓴다
#
# 재실행 안전: 환경이 있으면 복제를 건너뛰고 패치/검증만 다시 한다.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/sn_tune/transformers_patch"

SRC_ENV="${SRC_ENV:-hb}"
DST_ENV="${DST_ENV:-hb_sn}"
FAMILIES="${FAMILIES:-llama gemma2 qwen2}"
FORCE="${FORCE:-0}"

# ---------------------------------------------------------------------------
# 0. conda
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda 를 PATH 에서 찾을 수 없다." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_prefix() { conda env list | awk -v n="$1" '$1==n {print $NF}'; }

SRC_PREFIX="$(env_prefix "$SRC_ENV")"
if [[ -z "$SRC_PREFIX" ]]; then
    echo "[ERROR] 원본 환경 '${SRC_ENV}' 이 없다. conda env list 로 확인하라." >&2
    exit 1
fi
echo "[INFO] 원본 환경: ${SRC_ENV} -> ${SRC_PREFIX}"

# ---------------------------------------------------------------------------
# 1. 복제
# ---------------------------------------------------------------------------
DST_PREFIX="$(env_prefix "$DST_ENV")"
if [[ -n "$DST_PREFIX" ]]; then
    echo "[SKIP] '${DST_ENV}' 이 이미 있다 -> ${DST_PREFIX} (복제 건너뜀)"
else
    echo "[INFO] '${SRC_ENV}' -> '${DST_ENV}' 복제 중 (수십 GB, 수 분 걸린다) ..."
    conda create --name "$DST_ENV" --clone "$SRC_ENV" --yes
    DST_PREFIX="$(env_prefix "$DST_ENV")"
    [[ -n "$DST_PREFIX" ]] || { echo "[ERROR] 복제 후에도 '${DST_ENV}' 을 못 찾았다" >&2; exit 1; }
fi

PY_BIN="${DST_PREFIX}/bin/python"
[[ -x "$PY_BIN" ]] || { echo "[ERROR] python 이 없다: ${PY_BIN}" >&2; exit 1; }

SITE_TF="$("$PY_BIN" - <<'PY'
import os, transformers
print(os.path.dirname(transformers.__file__))
PY
)"
echo "[INFO] 대상 transformers: ${SITE_TF}"
echo "[INFO] 버전: $("$PY_BIN" -c 'import transformers; print(transformers.__version__)')"

# ---------------------------------------------------------------------------
# 2. 패치 덮어쓰기 (원본은 .orig 로 한 번만 백업)
# ---------------------------------------------------------------------------
for fam in $FAMILIES; do
    src="${PATCH_DIR}/modeling_${fam}.py"
    dst="${SITE_TF}/models/${fam}/modeling_${fam}.py"

    if [[ ! -f "$src" ]]; then
        echo "[WARN] vendored 패치본 없음, 건너뜀: ${src}"
        continue
    fi
    if [[ ! -f "$dst" ]]; then
        echo "[WARN] 대상 파일 없음, 건너뜀: ${dst}"
        continue
    fi

    if cmp -s "$src" "$dst"; then
        if [[ "$FORCE" != "1" ]]; then
            echo "[SKIP] ${fam}: 이미 패치본과 동일"
            continue
        fi
    fi

    # 최초 1회만 백업한다. 두 번째부터 덮으면 백업이 '패치본'이 되어 복구가 불가능해진다.
    if [[ ! -f "${dst}.orig" ]]; then
        cp -p "$dst" "${dst}.orig"
        echo "[INFO] ${fam}: 정품 백업 -> $(basename "${dst}").orig"
    fi

    # cp 로 바이트 그대로 복사한다. gemma2/qwen2 는 CRLF 이므로 텍스트 변환을 태우면 안 된다.
    cp -p "$src" "$dst"
    echo "[OK]   ${fam}: 패치 설치"
done

# ---------------------------------------------------------------------------
# 3. 검증
# ---------------------------------------------------------------------------
echo
echo "[INFO] 검증 (hb_sn = 패치 있어야 함) ..."
rc=0
for fam in $FAMILIES; do
    echo "--- ${fam}"
    ( cd "$REPO_ROOT" && "$PY_BIN" -m sn_tune.verify_patch --model_family "$fam" --expect present ) || rc=1
done

echo
echo "[INFO] 검증 (${SRC_ENV} = 정품이어야 함) ..."
SRC_PY="${SRC_PREFIX}/bin/python"
for fam in $FAMILIES; do
    echo "--- ${fam}"
    ( cd "$REPO_ROOT" && "$SRC_PY" -m sn_tune.verify_patch --model_family "$fam" --expect absent ) || rc=1
done

echo
if [[ $rc -eq 0 ]]; then
    echo "[DONE] '${DST_ENV}' 준비 완료. 검출은 이 환경에서, 학습은 '${SRC_ENV}' 에서 돌려라."
else
    echo "[DONE] 경고와 함께 종료 — 위 FAIL 항목을 먼저 해결하라." >&2
fi
exit $rc
