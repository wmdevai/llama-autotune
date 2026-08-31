#!/usr/bin/env bash
# Launcher per la Web UI di llama-autotune.
# Avvia il server in background e apre Brave; alla chiusura della pagina
# il server si ferma automaticamente.
set -e

cd "$(dirname "$0")"

export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp/build/bin}"

exec .venv/bin/llama-autotune web --browser brave
