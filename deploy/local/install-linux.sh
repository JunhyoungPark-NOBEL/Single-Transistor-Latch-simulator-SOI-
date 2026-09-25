#!/bin/bash
# STL Simulator installer for Linux:  bash install-linux.sh [--yes] [--port N] [--install-dir DIR] [--no-browser]
# Runs installer-files/stl-local.sh (shared with macOS); see that file for all options.
DIR=$(cd "$(dirname "$0")" && pwd -P)
if [ ! -f "$DIR/installer-files/stl-local.sh" ]; then
    echo "installer-files/stl-local.sh 이(가) 없습니다. ZIP 압축을 전부 푼 폴더에서 실행하세요." >&2
    echo "installer-files/stl-local.sh is missing: run this from the fully extracted kit folder." >&2
    exit 1
fi
# Started from a file manager without a terminal: reopen in a terminal window so the password can be typed.
if [ ! -t 0 ] && [ -z "${STL_INSTALL_PASSWORD:-}" ] && [ -z "${STL_NO_TERMINAL:-}" ] && \
   { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
    case " $* " in *" --password-stdin "*) ;; *)
        term=""
        for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal xterm; do
            if command -v "$t" >/dev/null 2>&1; then term=$t; break; fi
        done
        if [ -n "$term" ]; then
            export STL_NO_TERMINAL=1 STL_PAUSE=1
            if [ "$term" = "gnome-terminal" ]; then exec gnome-terminal -- bash "$DIR/install-linux.sh" "$@"; fi
            exec "$term" -e bash "$DIR/install-linux.sh" "$@"
        fi ;;
    esac
fi
exec bash "$DIR/installer-files/stl-local.sh" install "$@"
