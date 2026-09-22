#!/usr/bin/env bash
# =============================================================================
# setup_patched_transformers.sh — 검출용 패치 transformers 를 **오버레이**로 만든다.
#
# `setup_hb_sn.sh` 의 가벼운 대안이다. conda 환경을 통째로 복제(수십 GB + 채널 ToS
# 동의 필요)하는 대신, `transformers` 패키지 디렉토리만 복사해서 패치를 얹고
# PYTHONPATH 로 앞세운다. 검출 프로세스에서만 패치본이 보이고, 같은 conda 환경의
# 학습 프로세스는 정품을 계속 쓴다.
#
#   bash sn_tune/setup_patched_transformers.sh
#   PYTHONPATH=<출력경로> python -m sn_tune.detect_warp_rotated ...
#
# 재실행 안전: 이미 있으면 패치만 다시 얹고 검증한다.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/sn_tune/transformers_patch"

SRC_ENV="${SRC_ENV:-hb}"
OVERLAY="${OVERLAY:-${REPO_ROOT}/.transformers_patched}"
FAMILIES="${FAMILIES:-llama gemma2 qwen2}"
FORCE="${FORCE:-0}"

command -v conda >/dev/null 2>&1 || { echo "[ERROR] conda 가 PATH 에 없다." >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
SRC_PREFIX="$(conda env list | awk -v n="$SRC_ENV" '$1==n {print $NF}')"
[[ -n "$SRC_PREFIX" ]] || { echo "[ERROR] 환경 '${SRC_ENV}' 이 없다." >&2; exit 1; }

SRC_PY="${SRC_PREFIX}/bin/python"
SRC_TF="$("$SRC_PY" -c 'import os,transformers; print(os.path.dirname(transformers.__file__))')"
TF_VER="$("$SRC_PY" -c 'import transformers; print(transformers.__version__)')"
echo "[INFO] 원본 transformers ${TF_VER}: ${SRC_TF}"

DST_TF="${OVERLAY}/transformers"
if [[ -d "$DST_TF" && "$FORCE" != "1" ]]; then
    echo "[SKIP] 오버레이가 이미 있다: ${DST_TF} (패치/검증만 다시 한다)"
else
    echo "[INFO] 복사 중 -> ${DST_TF} ..."
    rm -rf "$DST_TF"
    mkdir -p "$OVERLAY"
    cp -a "$SRC_TF" "$DST_TF"
    # dist-info 도 같이 옮겨야 importlib.metadata(version 조회)가 깨지지 않는다.
    for d in "${SRC_TF}/.."/transformers-*.dist-info; do
        [[ -d "$d" ]] && cp -a "$d" "$OVERLAY/" || true
    done
fi

for fam in $FAMILIES; do
    src="${PATCH_DIR}/modeling_${fam}.py"
    dst="${DST_TF}/models/${fam}/modeling_${fam}.py"
    [[ -f "$src" ]] || { echo "[WARN] 패치본 없음: ${src}"; continue; }
    [[ -f "$dst" ]] || { echo "[WARN] 대상 없음: ${dst}"; continue; }
    [[ -f "${dst}.orig" ]] || cp -p "$dst" "${dst}.orig"
    # cp 로 바이트 그대로. gemma2/qwen2 는 CRLF 라 텍스트 변환을 태우면 안 된다.
    cp -p "$src" "$dst"
    echo "[OK]   ${fam}: 패치 설치"
done

echo
echo "[INFO] 검증 — 오버레이는 패치 있어야 / 원본 env 는 정품이어야"
rc=0
for fam in $FAMILIES; do
    echo "--- ${fam} (오버레이)"
    ( cd "$REPO_ROOT" && PYTHONPATH="$OVERLAY" "$SRC_PY" -m sn_tune.verify_patch --model_family "$fam" --expect present ) || rc=1
    echo "--- ${fam} (원본 ${SRC_ENV})"
    ( cd "$REPO_ROOT" && "$SRC_PY" -m sn_tune.verify_patch --model_family "$fam" --expect absent ) || rc=1
done

echo
if [[ $rc -eq 0 ]]; then
    echo "[DONE] 준비 완료. 검출은 PYTHONPATH=${OVERLAY} 를 앞세워 돌려라."
    echo "       학습은 PYTHONPATH 없이 그대로 (정품 transformers)."
else
    echo "[DONE] 경고와 함께 종료 — 위 FAIL 을 먼저 해결하라." >&2
fi
exit $rc
