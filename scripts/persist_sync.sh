#!/usr/bin/env bash
# CLAUDE_CONFIG_DIR 을 걸기 *전에* 시작된 claude 세션은 아직 휘발성 ~/.claude 에
# 기록을 쌓는다. 그 세션을 끝내기 전에 한 번 돌려서 persist 로 넘겨준다.
# claude 를 재시작한 뒤로는 필요 없다(처음부터 persist 에 쓴다).
#
#   bash scripts/persist_sync.sh
set -euo pipefail
PERSIST=/NHNHOME/26msit001_A/BASE/edge_ai_lab/minseong/.persist/claude-config

[ -d "$HOME/.claude" ] || { echo "~/.claude 없음 — 이미 persist 만 쓰고 있다."; exit 0; }
[ "$(readlink -f "$HOME/.claude")" = "$(readlink -f "$PERSIST")" ] && {
    echo "~/.claude 가 이미 persist 다. 할 일 없음."; exit 0; }

if command -v rsync >/dev/null 2>&1; then
    rsync -a --update "$HOME/.claude/" "$PERSIST/"
else
    cp -a --update=older "$HOME/.claude/." "$PERSIST/"
fi
[ -f "$HOME/.claude.json" ] && cp -a --update=older "$HOME/.claude.json" "$PERSIST/.claude.json"
chmod 700 "$PERSIST"; chmod 600 "$PERSIST/.credentials.json" 2>/dev/null || true
echo "동기화 완료 -> $PERSIST"
