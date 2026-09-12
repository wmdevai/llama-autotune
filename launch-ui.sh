#!/usr/bin/env bash
# Launcher per la Web UI di llama-autotune.
#
# Avvia il server in background e apre il browser; alla chiusura della pagina
# il server si ferma automaticamente.
#
# La configurazione per-macchina (percorso di llama.cpp e browser) è in
# ~/.config/llama-autotune/launcher.env, generata da ./setup.sh.
set -e

cd "$(dirname "$0")"

CONF="${XDG_CONFIG_HOME:-$HOME/.config}/llama-autotune/launcher.env"
if [ -f "$CONF" ]; then
    # shellcheck disable=SC1090
    . "$CONF"
fi

export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp/build/bin}"

exec .venv/bin/llama-autotune web --browser "${AUTOTUNE_BROWSER:-default}"
