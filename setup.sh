#!/usr/bin/env bash
# setup.sh — prepara llama-autotune su una nuova macchina.
#
#   ./setup.sh [/percorso/a/llama.cpp/build/bin]
#
# Fa:
#   1. crea/aggiorna l'ambiente Python (.venv)
#   2. rileva la build di llama.cpp (llama-bench) e il browser
#   3. scrive ~/.config/llama-autotune/launcher.env
#   4. crea il launcher .desktop nel menu applicazioni
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/llama-autotune"
CONF_FILE="$CONF_DIR/launcher.env"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APPS_DIR/llama-autotune.desktop"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; }


setup_venv() {
    if [ -x "$REPO_DIR/.venv/bin/llama-autotune" ]; then
        info "Ambiente .venv già presente."
        return 0
    fi

    if command -v uv >/dev/null 2>&1; then
        info "Creo l'ambiente con uv sync --dev ..."
        (cd "$REPO_DIR" && uv sync --dev)
    elif command -v python3 >/dev/null 2>&1; then
        info "uv non trovato: uso python3 -m venv + pip ..."
        python3 -m venv "$REPO_DIR/.venv"
        "$REPO_DIR/.venv/bin/pip" install --upgrade pip >/dev/null
        "$REPO_DIR/.venv/bin/pip" install \
            -e "$REPO_DIR" pytest pytest-cov httpx ruff pyright
    else
        error "Serve 'uv' (consigliato) oppure python3. Installane uno e riprova."
        return 1
    fi
}


detect_llama_dir() {
    local candidates=()

    [ -n "${1:-}" ] && candidates+=("$1")
    [ -n "${LLAMA_CPP_DIR:-}" ] && candidates+=("$LLAMA_CPP_DIR")
    candidates+=(
        "$HOME/.local/src/llama.cpp/build/bin"
        "$HOME/llama.cpp/build/bin"
        "/usr/local/bin"
        "/usr/bin"
    )

    local dir
    for dir in "${candidates[@]}"; do
        if [ -x "$dir/llama-bench" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
    done

    if command -v llama-bench >/dev/null 2>&1; then
        dirname "$(command -v llama-bench)"
        return 0
    fi

    return 1
}


detect_browser() {
    local b
    for b in brave brave-browser chromium chromium-browser \
             google-chrome google-chrome-stable microsoft-edge \
             vivaldi opera firefox; do
        if command -v "$b" >/dev/null 2>&1; then
            printf '%s\n' "$b"
            return 0
        fi
    done
    printf 'default\n'
}


write_conf() {
    mkdir -p "$CONF_DIR"
    cat > "$CONF_FILE" <<EOF
# Generato da setup.sh il $(date +%Y-%m-%d)
LLAMA_CPP_DIR="$LLAMA_DIR"
AUTOTUNE_BROWSER="$BROWSER"
EOF
    info "Configurazione scritta: $CONF_FILE"
}


write_desktop() {
    mkdir -p "$APPS_DIR"
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=llama-autotune
Comment=Web UI per l'ottimizzazione di llama.cpp
Exec="$REPO_DIR/launch-ui.sh"
Icon=web-browser
Terminal=false
Categories=Utility;Development;
EOF
    info "Launcher creato: $DESKTOP_FILE"

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
    fi
}


main() {
    info "Repository: $REPO_DIR"

    setup_venv

    if ! LLAMA_DIR="$(detect_llama_dir "${1:-}")"; then
        warn "llama-bench non trovato."
        warn "Indica la directory con: ./setup.sh /percorso/a/llama.cpp/build/bin"
        LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/.local/src/llama.cpp/build/bin}"
    else
        info "llama.cpp: $LLAMA_DIR"
        if [ ! -x "$LLAMA_DIR/llama-server" ]; then
            warn "llama-server non trovato in $LLAMA_DIR (serve per il comando 'launch')."
        fi
    fi

    BROWSER="$(detect_browser)"
    info "Browser: $BROWSER"

    write_conf
    write_desktop

    printf '\n'
    info "Fatto. Avvia 'llama-autotune' dal menu applicazioni,"
    info "oppure da terminale: $REPO_DIR/launch-ui.sh"
}


main "$@"
