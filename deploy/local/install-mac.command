#!/bin/bash
# STL Simulator installer for macOS: double-click (Finder opens it in Terminal), or run
#   bash install-mac.command [--yes] [--port N] [--install-dir DIR] [--no-browser]
# Runs installer-files/stl-local.sh (shared with Linux); see that file for all options.
DIR=$(cd "$(dirname "$0")" && pwd -P)
if [ ! -f "$DIR/installer-files/stl-local.sh" ]; then
    echo "installer-files/stl-local.sh 을(를) 읽을 수 없습니다. ZIP 압축을 전부 푼 폴더에서 실행하세요." >&2
    echo "'터미널이 다운로드 폴더에 접근하려고 합니다' 창에서 '허용 안 함'을 눌렀다면: 시스템 설정 → 개인정보 보호 및 보안" >&2
    echo "→ 파일 및 폴더 → 터미널 → 다운로드 폴더를 켜고 다시 실행하세요 (또는 키트 폴더를 데스크탑으로 옮겨서 실행)." >&2
    echo "installer-files/stl-local.sh cannot be read: run this from the fully extracted kit folder. If you denied" >&2
    echo "Terminal access to the Downloads folder, allow it in System Settings -> Privacy & Security -> Files and Folders." >&2
    printf 'Enter를 누르면 창을 닫습니다 (press Enter to close) '; read -r _
    exit 1
fi
export STL_PAUSE="${STL_PAUSE:-1}"
exec bash "$DIR/installer-files/stl-local.sh" install "$@"
