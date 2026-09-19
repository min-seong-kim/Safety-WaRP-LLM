#!/usr/bin/env bash
# 서버를 재할당받아 $HOME 이 초기화됐을 때 딱 한 번 실행하면 되는 부트스트랩.
#
#   bash /NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/Safety-WaRP-LLM/scripts/link_env.sh
#
# conda, Claude Code(설치본+로그인 상태), 캐시 경로를 전부 다시 연결한다.
# 재실행해도 안전하다(idempotent).
set -euo pipefail
BASE=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong
PERSIST="$BASE/.persist"

echo "[1/4] 작업 디렉토리 링크"
for d in miniconda3 HarmBench lm-evaluation-harness Safety-WaRP-LLM; do
    ln -sfn "$BASE/$d" "$HOME/$d"
    echo "      ~/$d -> $BASE/$d"
done
mkdir -p "$BASE/.tmp" "$BASE/.triton_cache" "$BASE/.inductor_cache"

echo "[2/4] Claude Code 복원"
if [ -d "$PERSIST/claude-share" ]; then
    mkdir -p "$HOME/.local/share" "$HOME/.local/bin"
    # 설치본 자체를 링크해 두면 자동 업데이트로 받은 새 버전도 persist 에 쌓인다.
    if [ -e "$HOME/.local/share/claude" ] && [ ! -L "$HOME/.local/share/claude" ]; then
        mv "$HOME/.local/share/claude" "$HOME/.local/share/claude.local.$(date +%s)"
        echo "      기존 로컬 설치본은 claude.local.* 로 밀어둠"
    fi
    ln -sfn "$PERSIST/claude-share" "$HOME/.local/share/claude"
    # 가장 최신 버전 바이너리를 ~/.local/bin/claude 로
    LATEST="$(ls -1 "$PERSIST/claude-share/versions" 2>/dev/null | sort -V | tail -1)"
    if [ -n "$LATEST" ]; then
        ln -sfn "$HOME/.local/share/claude/versions/$LATEST" "$HOME/.local/bin/claude"
        echo "      ~/.local/bin/claude -> $LATEST"
    else
        echo "      !! versions/ 가 비어 있다. claude install 로 다시 받아야 한다." >&2
    fi
    echo "      CLAUDE_CONFIG_DIR -> $PERSIST/claude-config (로그인 상태 유지)"
else
    echo "      !! $PERSIST/claude-share 가 없다. Claude 는 새로 설치해야 한다." >&2
fi

echo "[3/4] 셸 연결"
MARK="# >>> warp env >>>"
if ! grep -qF "$MARK" "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<BRC

$MARK
source $BASE/Safety-WaRP-LLM/scripts/env.sh
# <<< warp env <<<
BRC
    echo "      ~/.bashrc 에 env.sh 소싱 추가"
else
    echo "      ~/.bashrc 는 이미 연결돼 있음"
fi
# 로그인 셸(bash -l, ssh)은 ~/.bashrc 를 읽지 않는다. 그래서 CLAUDE_CONFIG_DIR 이
# 빠지고 로그인 상태가 날아간 것처럼 보인다 -> ~/.bash_profile 로 넘겨준다.
if ! grep -qF "$MARK" "$HOME/.bash_profile" 2>/dev/null; then
    cat >> "$HOME/.bash_profile" <<'BP'

# >>> warp env >>>
# 로그인 셸도 ~/.bashrc 를 읽게 한다(CLAUDE_CONFIG_DIR / conda 가 여기서 온다).
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
# <<< warp env <<<
BP
    echo "      ~/.bash_profile 생성 (로그인 셸용)"
else
    echo "      ~/.bash_profile 은 이미 연결돼 있음"
fi

echo "[4/4] 확인"
# shellcheck disable=SC1090
source "$BASE/Safety-WaRP-LLM/scripts/env.sh"
printf '      conda  : %s\n' "$(conda --version 2>/dev/null || echo '없음')"
printf '      envs   : %s\n' "$(conda env list 2>/dev/null | awk '!/^#/ && NF {printf "%s ", $1}')"
printf '      claude : %s\n' "$("$HOME/.local/bin/claude" --version 2>/dev/null || echo '없음')"
printf '      config : %s\n' "$CLAUDE_CONFIG_DIR"
echo
echo "완료. 새 셸을 열거나 'source ~/.bashrc' 하면 된다."
