#!/usr/bin/env bash
# 이 박스(B200 컨테이너)에서 conda / Claude Code / 캐시 경로를 잡아주는 공용 스니펫.
#
#   source scripts/env.sh          # conda + claude 만 쓸 수 있게
#   source scripts/env.sh hb       # + hb 활성화 (학습용)
#   source scripts/env.sh harmbench
#
# 이 서버는 48시간마다 할당이 끝나고 재할당된다. 그때 $HOME(/home/edgeai_lab)은
# 컨테이너 overlay 라 통째로 날아가지만 lustre 의 $WARP_BASE 는 남는다.
# 그래서 conda / claude 설치본 · 인증 · 캐시를 전부 $WARP_BASE 아래 두고,
# 새 컨테이너에서는 심볼릭 링크만 다시 걸면 되게 해 뒀다:
#
#   bash /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM/scripts/link_env.sh
#
export WARP_BASE=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong
export WARP_PERSIST="$WARP_BASE/.persist"

# ── conda ────────────────────────────────────────────────────────────────
export CONDA_ROOT="${CONDA_ROOT:-$WARP_BASE/miniconda3}"
export CONDA_SH="${CONDA_SH:-$CONDA_ROOT/etc/profile.d/conda.sh}"

# ── Claude Code ──────────────────────────────────────────────────────────
# 설정·인증·세션 기록이 전부 여기 들어간다(.claude.json 포함). 이 변수가 있으면
# 재할당 후에도 로그인 상태가 그대로 유지된다. 설치 바이너리는 link_env.sh 가
# ~/.local/share/claude 로 링크해 준다(자동 업데이트분도 여기 쌓임).
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$WARP_PERSIST/claude-config}"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH";; esac

# ── HF 캐시 ──────────────────────────────────────────────────────────────
# 기존에 받아둔 모델(63개 repo)과 토큰을 그대로 재사용한다.
export HF_HOME="${HF_HOME:-$WARP_BASE/.hf_cache}"

# ── /tmp 이 noexec ───────────────────────────────────────────────────────
# Triton 이 JIT 한 .so 를 mmap 못 해서 죽는다. exec 가능한 곳으로 돌려놔야 한다.
# (conda env 의 activate.d 훅도 같은 값을 다시 잡아준다.)
export TMPDIR="${TMPDIR:-$WARP_BASE/.tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WARP_BASE/.triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WARP_BASE/.inductor_cache}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

[ -f "$CONDA_SH" ] && . "$CONDA_SH"

if [ -n "${1:-}" ]; then
    conda activate "$1" || echo "conda activate $1 실패" >&2
fi
