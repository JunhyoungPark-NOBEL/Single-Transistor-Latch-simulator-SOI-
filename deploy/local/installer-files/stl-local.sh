#!/bin/bash
# STL Simulator: local installer and control script for macOS and Linux (bash 3.2 compatible).
#
#   From the installer kit:  bash installer-files/stl-local.sh install [options]
#   Installed copy:          ~/STL-Simulator/bin/stl-sim.sh COMMAND [options]
#   Commands: install (kit only), start, stop, restart, open, status, logs, rollback, update [KIT_FOLDER],
#             uninstall, help
#   Options:  --yes             answer "yes" to every question (never to installing Docker itself)
#             --port N          port on 127.0.0.1 (default 8000, else the first free one of 8001-8010)
#             --install-dir D   install folder (default ~/STL-Simulator)
#             --password-stdin  read the password from the first line of stdin (automation)
#             --no-browser      do not open the browser        --no-shortcut   no launcher
#             --remove-cache / --keep-cache   (uninstall) delete / keep the result cache volume
#             --remove-build-cache / --keep-build-cache   (uninstall) delete / keep Docker's build cache
#   Environment: STL_INSTALL_PASSWORD  password for non-interactive installs (removed from the environment at once)
#                STL_BUILD_EXTRA_ARGS  extra `docker build` arguments (e.g. --build-context for a proxy CA image)
#                STL_PAUSE=1           wait for Enter before exiting (set by install-mac.command)
#
# Design (docs/LOCAL_INSTALL.md): the app is an encrypted payload (installer-files/stl-simulator.stlenc).  It is
# decrypted inside a throwaway python:3.11-slim container (docker run -i, no mounts, no network): the password is
# the first line of the container's stdin (printf is a bash builtin: never an argument, never in the container's
# environment, never written or logged), the payload follows, and the plaintext tar goes straight from the
# container's stdout into `docker build -`.  After the build the installer drops BuildKit's cached copy of that tar.

CONTAINER=stl-simulator
REPO=stl-simulator
VOLUME=stl-simulator-cache
HELPER_IMAGE=python:3.11-slim
LABEL=org.stl-simulator
PORT_MIN=8000
PORT_MAX=8010
ENGINE_TIMEOUT=180
HEALTH_TIMEOUT=240
BOOT="import os,base64;exec(base64.b64decode(os.environ['STL_HELPER_B64']))"
DOCKER_DOWNLOAD_URL="https://www.docker.com/products/docker-desktop/"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
OS_NAME=$(uname -s)
KIT_MODE=0
[ -f "$SCRIPT_DIR/kit-info.txt" ] && KIT_MODE=1

YES=0
PORT_ARG=""
DIR_ARG=""
PW_STDIN=0
NO_BROWSER=0
NO_SHORTCUT=0
CACHE_CHOICE=""
BUILD_CACHE_CHOICE=""
ROLLBACK=0
KIT_ARG=""
PW=""
LOG_FILE=""
TMP_LOG=""
INSTALL_DIR=""

# ------------------------------------------------------------------------------------------ output
say() {
    printf '%s\n' "$1"
    [ -n "${2:-}" ] && printf '  %s\n' "$2"
    log "$1${2:+ | $2}"
}
step() { printf '\n==> %s\n    %s\n' "$1" "$2"; log "== $1 | $2"; }
warn() { printf '[주의] %s\n' "$1" >&2; [ -n "${2:-}" ] && printf '       %s\n' "$2" >&2; log "WARN $1${2:+ | $2}"; }
log() {
    [ -n "$LOG_FILE" ] || return 0
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG_FILE" 2>/dev/null
    return 0
}
pause_end() {
    if [ "${STL_PAUSE:-0}" = "1" ] && tty_ok; then
        printf '\n'
        read -r -p "Enter를 누르면 창을 닫습니다 (press Enter to close) " _ < /dev/tty
    fi
}
die() {
    printf '\n[오류] %s\n' "$1" >&2
    [ -n "${2:-}" ] && printf '       %s\n' "$2" >&2
    log "ERROR $1${2:+ | $2}"
    [ -n "$LOG_FILE" ] && [ "$LOG_FILE" != "$TMP_LOG" ] && printf '       로그 / log: %s\n' "$LOG_FILE" >&2
    PW=""
    pause_end
    exit 1
}
tty_ok() { [ -t 0 ] || ( : < /dev/tty ) 2>/dev/null; }

# ask QUESTION_KO QUESTION_EN DEFAULT(y|n) [strict]  -> 0 = yes.
# strict (installing or starting system software): only an explicit answer typed by a person counts as "yes";
# neither --yes nor a missing terminal does.
ask() {
    if [ "${4:-}" = "strict" ]; then tty_ok || return 1
    elif [ "$YES" = "1" ]; then return 0
    elif ! tty_ok; then [ "$3" = "y" ]; return; fi
    local hint="[y/N]" ans
    [ "$3" = "y" ] && hint="[Y/n]"
    printf '%s\n  %s %s ' "$1" "$2" "$hint"
    read -r ans < /dev/tty
    case "$ans" in
        [Yy]*|네|예|응) return 0 ;;
        [Nn]*|아니*) return 1 ;;
        *) [ "$3" = "y" ] ;;
    esac
}

# ------------------------------------------------------------------------------------------ helpers
sha256_of() {  # prints nothing when no SHA-256 tool exists (the payload's own HMAC still protects it)
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'; fi
}
now_stamp() { date '+%Y%m%d-%H%M%S'; }
is_macos() { [ "$OS_NAME" = "Darwin" ]; }

kv_get() {  # kv_get FILE KEY
    [ -f "$1" ] || return 0
    local k v
    while IFS='=' read -r k v || [ -n "$k" ]; do
        v=$(printf '%s' "$v" | tr -d '\r')
        if [ "$k" = "$2" ]; then printf '%s' "$v"; return 0; fi
    done < "$1"
}

open_url() {
    if is_macos; then open "$1" >/dev/null 2>&1 && return 0
    elif command -v xdg-open >/dev/null 2>&1 && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
        # Started from a launcher with Terminal=true this script is the terminal's session leader: when it exits, the
        # kernel hangs up its process group and would kill xdg-open before the browser starts. Ignore SIGHUP
        # (inherited through fork and exec) and, where available, start it in a session of its own.
        if command -v setsid >/dev/null 2>&1; then
            ( trap '' HUP; setsid xdg-open "$1" >/dev/null 2>&1 < /dev/null & ) && return 0
        else
            ( trap '' HUP; xdg-open "$1" >/dev/null 2>&1 < /dev/null & ) && return 0
        fi
    fi
    return 1
}

http_get() {  # prints the body of a local URL; no proxy
    if command -v curl >/dev/null 2>&1; then curl -fsS --max-time 5 --noproxy '*' "$1" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then wget -q -T 5 --no-proxy -O - "$1" 2>/dev/null
    else
        docker exec "$CONTAINER" python -c "import urllib.request,sys;sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).read().decode())" 2>/dev/null
    fi
}

port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# ------------------------------------------------------------------------------------------ docker
find_docker() {
    command -v docker >/dev/null 2>&1 && return 0
    local d
    for d in /usr/local/bin /opt/homebrew/bin "$HOME/.docker/bin" /Applications/Docker.app/Contents/Resources/bin \
             /usr/bin /snap/bin; do
        if [ -x "$d/docker" ]; then PATH="$d:$PATH"; export PATH; return 0; fi
    done
    return 1
}

offer_docker_install() {
    say "Docker가 설치되어 있지 않습니다. STL Simulator는 Docker 안에서 실행됩니다." \
        "Docker is not installed. The simulator runs inside Docker."
    if is_macos; then
        if command -v brew >/dev/null 2>&1 && ask "Homebrew로 Docker Desktop을 설치할까요?" \
                "Install Docker Desktop with Homebrew?" y strict; then
            brew install --cask docker-desktop || brew install --cask docker || die "Homebrew 설치에 실패했습니다." \
                "Homebrew install failed; download Docker Desktop from $DOCKER_DOWNLOAD_URL"
            open -a Docker >/dev/null 2>&1
            say "Docker Desktop을 한 번 실행해 약관에 동의하고, 고래 아이콘이 멈추면 이 설치 파일을 다시 실행하세요." \
                "Start Docker Desktop once, accept its terms, then run this installer again."
        else
            say "Docker Desktop 내려받기 페이지를 엽니다. 설치하고 한 번 실행한 뒤 이 설치 파일을 다시 실행하세요." \
                "Opening the Docker Desktop download page; install it, start it once, then run this installer again."
            open_url "$DOCKER_DOWNLOAD_URL" || say "  $DOCKER_DOWNLOAD_URL"
        fi
    else
        say "공식 설치 스크립트(https://get.docker.com)로 Docker Engine을 설치할 수 있습니다 (sudo 필요)." \
            "Docker Engine can be installed with the official script https://get.docker.com (needs sudo)."
        if ask "지금 설치할까요?" "Install Docker Engine now?" n strict; then
            local tmp
            tmp=$(mktemp "${TMPDIR:-/tmp}/get-docker.XXXXXX") || die "임시 파일을 만들 수 없습니다." "mktemp failed"
            if command -v curl >/dev/null 2>&1; then curl -fsSL https://get.docker.com -o "$tmp"
            else wget -q -O "$tmp" https://get.docker.com; fi || { rm -f "$tmp"; die "get.docker.com을 받지 못했습니다." "download failed"; }
            sudo sh "$tmp" || { rm -f "$tmp"; die "Docker 설치에 실패했습니다." "Docker install failed"; }
            rm -f "$tmp"
            if ask "현재 사용자($USER)를 docker 그룹에 추가할까요? (sudo 없이 docker 사용)" \
                    "Add $USER to the docker group (use docker without sudo)?" y strict; then
                sudo usermod -aG docker "$USER"
            fi
            say "Docker 설치가 끝났습니다. 로그아웃 후 다시 로그인하고 이 설치 파일을 다시 실행하세요." \
                "Docker is installed. Log out and in again, then run this installer again."
        else
            say "Docker 설치 안내: https://docs.docker.com/engine/install/" "Install guide: https://docs.docker.com/engine/install/"
        fi
    fi
    pause_end
    exit 1
}

engine_error() { docker info --format '{{.ServerVersion}}' 2>&1 >/dev/null | head -n 3; }
engine_up() { docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; }

ensure_engine() {
    engine_up && return 0
    local err
    err=$(engine_error)
    case "$err" in
        *"permission denied"*)
            die "docker에 접근할 권한이 없습니다. 'sudo usermod -aG docker $USER' 실행 후 로그아웃·로그인하세요." \
                "Permission denied on the Docker socket: add yourself to the docker group, log out and in again." ;;
    esac
    say "Docker 엔진이 꺼져 있어 시작합니다..." "The Docker engine is not running; starting it..."
    if is_macos; then
        open -g -a Docker >/dev/null 2>&1 || open -a Docker >/dev/null 2>&1 || \
            die "Docker Desktop을 시작하지 못했습니다. 응용 프로그램에서 Docker를 직접 실행하세요." "Could not start Docker Desktop; start it from Applications."
    elif command -v systemctl >/dev/null 2>&1; then
        if systemctl --user cat docker-desktop.service >/dev/null 2>&1; then
            systemctl --user start docker-desktop
        elif ask "sudo systemctl start docker 로 Docker를 시작할까요?" "Start Docker with sudo systemctl start docker?" y strict; then
            sudo systemctl start docker
        else
            die "Docker를 시작한 뒤 다시 실행하세요: sudo systemctl start docker" "Start Docker, then run again."
        fi
    else
        die "Docker 엔진을 시작한 뒤 다시 실행하세요 (예: sudo service docker start)." "Start the Docker engine, then run again."
    fi
    if is_macos; then
        say "Docker Desktop 창이 뜨면 약관에 동의(Accept)하세요. 로그인(Sign in)은 건너뛰어도 됩니다. 처음에는 1–3분 걸립니다." \
            "If the Docker Desktop window opens, accept its terms (signing in can be skipped); the first start takes 1-3 min."
    fi
    local waited=0 total=0
    while ! engine_up; do
        if [ "$waited" -ge "$ENGINE_TIMEOUT" ]; then
            is_macos && say "Docker Desktop 창에 약관 동의나 업데이트 안내가 떠 있는지 확인하세요." \
                "Check the Docker Desktop window for a license or update prompt."
            if [ "$YES" = "0" ] && tty_ok && ask "Docker가 아직 준비되지 않았습니다. 더 기다릴까요?" "Docker is not ready yet. Keep waiting?" y; then
                waited=0
                continue
            fi
            die "Docker 엔진이 ${total}초 안에 시작되지 않았습니다: $(engine_error)" \
                "The Docker engine did not start within ${total} s."
        fi
        [ $((total % 15)) -eq 0 ] && say "  Docker 시작을 기다리는 중... (${total}s)" "  waiting for Docker..."
        sleep 3
        waited=$((waited + 3))
        total=$((total + 3))
    done
    say "Docker 엔진이 준비되었습니다." "Docker engine is ready."
}

# Linux Docker Engine before 28.0 lets computers on the same network reach ports published on 127.0.0.1 by routing
# to the container address (fixed in 28.0: https://docs.docker.com/engine/release-notes/28/).  Docker Desktop and
# rootless Docker keep containers in a VM or a namespace of their own and are not affected.
check_engine_version() {
    is_macos && return 0
    local v major
    v=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
    major=${v%%.*}
    case "$major" in ''|*[!0-9]*) return 0 ;; esac
    [ "$major" -ge 28 ] && return 0
    case "$(docker info --format '{{.OperatingSystem}} {{.SecurityOptions}}' 2>/dev/null)" in
        *"Docker Desktop"*|*rootless*) return 0 ;;
    esac
    warn "Docker Engine ${v}은(는) 28.0보다 오래된 버전입니다. 이 버전에서는 같은 네트워크의 다른 컴퓨터가 시뮬레이터에 접속할 수 있습니다." \
         "Docker Engine $v is older than 28.0: other computers on the same network could reach the simulator."
    say "Docker Engine 28 이상으로 업그레이드하세요: https://docs.docker.com/engine/install/" \
        "Please upgrade Docker Engine to 28 or later: https://docs.docker.com/engine/install/"
    ask "그래도 설치할까요? (믿을 수 있는 네트워크에서만)" "Install anyway (trusted network only)?" n strict || \
        die "Docker Engine을 28 이상으로 업그레이드한 뒤 다시 실행하세요." "Upgrade Docker Engine to 28 or later, then run the installer again."
}

check_platform() {
    local ostype arch ncpu mem
    ostype=$(docker info --format '{{.OSType}}' 2>/dev/null)
    [ "$ostype" = "linux" ] || die "Docker가 Linux 컨테이너 모드가 아닙니다 ($ostype)." "Docker must run Linux containers."
    arch=$(docker info --format '{{.Architecture}}' 2>/dev/null)
    ncpu=$(docker info --format '{{.NCPU}}' 2>/dev/null)
    mem=$(docker info --format '{{.MemTotal}}' 2>/dev/null)
    case "$ncpu" in ''|*[!0-9]*) ncpu=2 ;; esac
    case "$mem" in ''|*[!0-9]*) mem=0 ;; esac
    MEM_MB=$((mem / 1048576))
    WORKERS=$((ncpu - 1))
    [ "$WORKERS" -gt 8 ] && WORKERS=8
    if [ "$MEM_MB" -gt 0 ]; then
        local by_mem=$(((MEM_MB - 1024) / 300))
        [ "$WORKERS" -gt "$by_mem" ] && WORKERS=$by_mem
    fi
    [ "$WORKERS" -lt 1 ] && WORKERS=1
    say "Docker: $arch, CPU $ncpu, 메모리 ${MEM_MB} MB → 계산 프로세스 ${WORKERS}개" \
        "Docker: $arch, $ncpu CPUs, ${MEM_MB} MB memory -> $WORKERS compute workers"
    case "$arch" in aarch64|arm64) say "Apple Silicon/ARM용으로 직접 빌드합니다." "Building natively for arm64." ;; esac
    if [ "$MEM_MB" -gt 0 ] && [ "$MEM_MB" -lt 3500 ]; then
        warn "Docker에 할당된 메모리가 적습니다 (${MEM_MB} MB). Docker Desktop → Settings → Resources에서 4 GB 이상 권장." \
             "Docker has little memory (${MEM_MB} MB); 4 GB or more is recommended."
    fi
}

check_disk() {  # check_disk NEED_GB
    local dir root avail_kb
    root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    dir="$HOME"
    if ! is_macos && [ -n "$root" ] && [ -d "$root" ]; then dir="$root"; fi
    avail_kb=$(df -Pk "$dir" 2>/dev/null | awk 'NR==2 {print $4}')
    case "$avail_kb" in ''|*[!0-9]*) return 0 ;; esac
    if [ "$avail_kb" -lt $(($1 * 1048576)) ]; then
        warn "디스크 여유 공간이 $((avail_kb / 1048576)) GB입니다. 설치에는 약 $1 GB가 필요합니다." \
             "Only $((avail_kb / 1048576)) GB free; about $1 GB are needed."
        ask "그래도 계속할까요?" "Continue anyway?" n || die "디스크 공간을 확보한 뒤 다시 실행하세요." "Free some disk space and run again."
    fi
}

container_exists() { docker container inspect "$CONTAINER" >/dev/null 2>&1; }
container_running() { [ "$(docker container inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" = "running" ]; }
image_exists() { [ -n "$1" ] && docker image inspect "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------------------------------------ state
# S_<key> variables are read and written through eval (bash 3.2 has no associative arrays)
# shellcheck disable=SC2034
S_INSTALLED=""
STATE_KEYS="PORT CURRENT_VERSION CURRENT_IMAGE CURRENT_BUILD PREVIOUS_VERSION PREVIOUS_IMAGE WORKERS INSTALLED"
load_state() {
    local f="$INSTALL_DIR/install-state.txt" k
    for k in $STATE_KEYS; do eval "S_$k=\$(kv_get \"\$f\" \"$k\")"; done
}
save_state() {
    local f="$INSTALL_DIR/install-state.txt" k v
    : > "$f.tmp" || return 1
    for k in $STATE_KEYS; do
        eval "v=\${S_$k:-}"
        printf '%s=%s\n' "$k" "$v" >> "$f.tmp"
    done
    mv -f "$f.tmp" "$f"
}

normalize_dir() {  # typed or dropped path: trim, remove quotes, undo "\ " escapes (Terminal drag and drop), expand ~
    local d
    d=$(printf '%s' "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    case "$d" in
        \"*\") d=${d#\"}; d=${d%\"} ;;
        \'*\') d=${d#\'}; d=${d%\'} ;;
    esac
    case "$d" in *\\*) d=$(printf '%s' "$d" | sed 's/\\\(.\)/\1/g') ;; esac
    # shellcheck disable=SC2088  # the literal text "~/" typed by the user is matched on purpose
    case "$d" in
        "~") d=$HOME ;;
        "~/"*) d="$HOME/${d#??}" ;;
    esac
    printf '%s' "$d"
}

resolve_install_dir() {
    [ -n "$DIR_ARG" ] && DIR_ARG=$(normalize_dir "$DIR_ARG")
    if [ -n "$DIR_ARG" ]; then INSTALL_DIR="$DIR_ARG"
    elif [ "$KIT_MODE" = "0" ] && [ -f "$SCRIPT_DIR/../install-state.txt" ]; then INSTALL_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
    else
        local labelled
        labelled=$(docker container inspect -f "{{index .Config.Labels \"$LABEL.dir\"}}" "$CONTAINER" 2>/dev/null)
        if [ -n "$labelled" ] && [ -f "$labelled/install-state.txt" ]; then INSTALL_DIR="$labelled"
        else INSTALL_DIR="$HOME/STL-Simulator"; fi
    fi
    case "$INSTALL_DIR" in /*) ;; *) INSTALL_DIR="$(pwd -P)/$INSTALL_DIR" ;; esac
    INSTALL_DIR="${INSTALL_DIR%/}"
    if [ -z "$INSTALL_DIR" ] || [ "$INSTALL_DIR" = "$HOME" ] || [ "$INSTALL_DIR" = "/" ]; then
        die "설치 폴더로 쓸 수 없는 경로입니다: '$INSTALL_DIR'" "Unsuitable install folder."
    fi
    case "/$INSTALL_DIR/" in
        */\~*) die "경로에 '~'로 시작하는 폴더 이름이 있습니다: $INSTALL_DIR  (전체 경로로 입력하세요. 예: $HOME/STL-Simulator)" \
                   "The path has a folder name starting with '~'; type the full path, e.g. $HOME/STL-Simulator" ;;
    esac
}

MARKER=.stl-simulator-folder   # written into the install folder as soon as it is created
is_our_dir() { [ -f "$1/$MARKER" ] || [ -f "$1/install-state.txt" ] || [ -f "$1/bin/stl-sim.sh" ]; }

# ------------------------------------------------------------------------------------------ password
helper_b64() { base64 < "$SCRIPT_DIR/stl_payload.py" | tr -d '\n\r'; }

run_helper() {  # run_helper MODE < payload   (the password $PW goes first on the container's stdin)
    { printf '%s\n' "$PW"; cat; } | STL_HELPER_B64="$HELPER_B64" STL_EXPECT_SHA256="$KIT_ARCHIVE_SHA" \
        docker run --rm -i --network none -e STL_PW_STDIN=1 -e STL_HELPER_B64 -e STL_EXPECT_SHA256 \
        "$HELPER_IMAGE" python3 -c "$BOOT" "$1"
}

ensure_helper_image() {
    image_exists "$HELPER_IMAGE" && return 0
    say "기본 이미지($HELPER_IMAGE)를 내려받는 중..." "Pulling $HELPER_IMAGE..."
    docker pull "$HELPER_IMAGE" >> "${LOG_FILE:-/dev/null}" 2>&1 || \
        die "$HELPER_IMAGE 을(를) 받지 못했습니다. 인터넷·프록시·방화벽을 확인하세요." \
            "Could not pull $HELPER_IMAGE: check the internet connection, proxy or firewall."
}

get_password() {
    local tries=0 max=3 code
    if [ -n "${STL_INSTALL_PASSWORD:-}" ]; then
        PW="$STL_INSTALL_PASSWORD"; max=1
    elif [ "$PW_STDIN" = "1" ]; then
        IFS= read -r PW || true; max=1
    fi
    unset STL_INSTALL_PASSWORD
    while [ "$tries" -lt "$max" ]; do
        tries=$((tries + 1))
        if [ "$max" -gt 1 ] || [ -z "$PW" ]; then
            tty_ok || die "비밀번호를 입력받을 수 없습니다 (터미널 없음)." "No terminal to read the password from (use --password-stdin)."
            printf '비밀번호를 입력하세요 (입력 내용은 보이지 않습니다) / Password: '
            IFS= read -r -s PW < /dev/tty
            printf '\n'
        fi
        PW=$(printf '%s' "$PW" | tr -d '\r')
        if [ -z "$PW" ]; then say "비밀번호가 비어 있습니다." "The password is empty."; continue; fi
        say "비밀번호 확인 중..." "Checking the password..."
        run_helper verify < "$PAYLOAD" 2>> "${LOG_FILE:-/dev/null}"
        code=$?
        case "$code" in
            0) say "비밀번호가 확인되었습니다." "Password accepted."; return 0 ;;
            3) if printf '%s' "$PW" | LC_ALL=C grep -q '[^ -~]'; then
                   say "비밀번호가 맞지 않습니다. 한글로 입력된 것 같습니다: 한/영 키를 누른 뒤 다시 입력하세요." \
                       "Wrong password. Non-English (e.g. Hangul) input detected: switch the keyboard to English and retry."
               else
                   say "비밀번호가 맞지 않습니다. (한/영 · Caps Lock 확인)" "Wrong password (check the input language and Caps Lock)."
               fi
               PW="" ;;
            4) die "설치 파일이 손상되었습니다. 키트를 다시 받으세요." "The payload is damaged; download the kit again." ;;
            *) die "비밀번호 확인 중 Docker 오류가 났습니다 (코드 $code)." "Docker error while checking the password (exit $code)." ;;
        esac
    done
    PW=""
    die "비밀번호가 맞지 않습니다. 아무것도 설치하지 않았습니다." "Wrong password; nothing was installed."
}

# ------------------------------------------------------------------------------------------ build + run
explain_build_failure() {
    local blog="$1" dlog="$2"
    if [ -s "$dlog" ] && grep -q "error" "$dlog"; then
        die "설치 파일을 푸는 중 오류가 났습니다: $(tail -n 1 "$dlog")" "Decryption failed (see $dlog)."
    fi
    if grep -Eiq "failed to resolve|dial tcp|i/o timeout|TLS handshake|x509|certificate|proxyconnect|ECONNRESET|ETIMEDOUT|EAI_AGAIN|network is unreachable|Temporary failure in name resolution|Could not fetch URL|failed to do request|connection refused|npm ERR! network|Read timed out" "$blog"; then
        say "" ""
        say "인터넷 연결 문제로 보입니다. 처음 설치할 때는 Docker Hub, npm, PyPI에서 부품을 내려받습니다." \
            "This looks like a network problem: the first build downloads from Docker Hub, npm and PyPI."
        say "- 회사/학교 프록시가 있으면 Docker Desktop → Settings → Resources → Proxies에 설정하세요." \
            "- Behind a proxy: set it in Docker Desktop -> Settings -> Resources -> Proxies."
        say "- 방화벽/보안 프로그램이 docker.io, registry.npmjs.org, pypi.org, files.pythonhosted.org를 막는지 확인하세요." \
            "- Check that a firewall does not block docker.io, registry.npmjs.org, pypi.org, files.pythonhosted.org."
        say "- 'x509/certificate' 오류는 보안 프로그램의 HTTPS 검사 때문입니다 (docs/LOCAL_INSTALL.md 참고)." \
            "- 'x509/certificate' errors come from HTTPS inspection (see docs/LOCAL_INSTALL.md)."
    elif grep -Eiq "no space left on device" "$blog"; then
        say "디스크 공간이 부족합니다. Docker Desktop → Troubleshoot → Clean/Purge 또는 불필요한 이미지를 지우세요." \
            "Out of disk space: free space or clean Docker images."
    fi
    printf '\n--- build log (last 25 lines) ---\n'
    tail -n 25 "$blog"
    die "이미지 빌드에 실패했습니다." "The image build failed (log: $blog)."
}

filter_build_output() {  # show each build step once and the warm-up lines; everything goes to the log anyway
    local line id seen=" " start=$SECONDS re_step='^#[0-9]+ \[[A-Za-z0-9_-]+ +[0-9]+/[0-9]+\] '
    while IFS= read -r line; do
        case "$line" in
            *"[warmup]"*)
                printf '          %s\n' "$(printf '%s' "$line" | sed 's/^#[0-9]* [0-9.]* //' | cut -c1-110)"
                continue ;;
            *"ERROR"*|*"error:"*)
                printf '  %s\n' "$(printf '%s' "$line" | cut -c1-160)"
                continue ;;
        esac
        if [[ $line =~ $re_step ]]; then
            id=${line%% *}
            case "$seen" in *" $id "*) continue ;; esac
            seen="$seen$id "
            printf '  [%02d:%02d] %s\n' $(((SECONDS - start) / 60)) $(((SECONDS - start) % 60)) \
                "$(printf '%s' "${line#* }" | cut -c1-110)"
        fi
    done
}

build_image() {
    local tag="$1" stamp blog dlog progress="" status
    stamp=$(now_stamp)
    blog="$INSTALL_DIR/logs/build-$KIT_VERSION-$stamp.log"
    dlog="$INSTALL_DIR/logs/decrypt-$stamp.log"
    docker buildx version >/dev/null 2>&1 && progress="--progress=plain"
    say "이미지를 만듭니다: $tag  (처음에는 5–15분 걸립니다. 정상입니다.)" \
        "Building $tag (the first build takes 5-15 minutes; this is normal)."
    say "  자세한 기록 / full log: $blog" ""
    # decrypt (container stdout) | docker build from the tar stream on stdin; nothing lands on this disk
    # shellcheck disable=SC2086  # STL_BUILD_EXTRA_ARGS is split into words on purpose
    run_helper decrypt < "$PAYLOAD" 2> "$dlog" | \
        docker build $progress --label "$LABEL=1" --label "$LABEL.version=$KIT_VERSION" \
            --label "$LABEL.build=$KIT_BUILD_ID" ${STL_BUILD_EXTRA_ARGS:-} -t "$tag" - 2>&1 | \
        tee "$blog" | filter_build_output
    status=${PIPESTATUS[1]}
    PW=""
    prune_build_context
    [ "$status" = "0" ] || explain_build_failure "$blog" "$dlog"
    [ -s "$dlog" ] || rm -f "$dlog"
    docker tag "$tag" "$REPO:local" >> "$LOG_FILE" 2>&1
    say "이미지 빌드 완료 ($tag)." "Image built."
}

# BuildKit caches the build context of `docker build -`, i.e. the decrypted source archive (as the downloaded
# "http url http://buildkit-session/..." record and its unpacked "copy /context /" snapshot).  Nothing needs them
# after the build: delete exactly those records (other projects' build cache is not touched).  The regexes use "."
# for the spaces so that the filter has no spaces or quotes (Windows PowerShell 5.1 mangles such native arguments;
# the Windows installer uses the same strings).
prune_build_context() {
    docker buildx version >/dev/null 2>&1 || return 0
    docker buildx prune -f --filter 'description~=^http.url.http://buildkit-session/' >> "${LOG_FILE:-/dev/null}" 2>&1
    docker buildx prune -f --filter 'description~=^copy./context./$' >> "${LOG_FILE:-/dev/null}" 2>&1
    return 0
}

run_container() {  # run_container IMAGE PORT -> 0 ok, 2 port busy, 1 other error
    local out
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    out=$(docker run -d --name "$CONTAINER" --restart unless-stopped -p "127.0.0.1:$2:8000" \
        -e "STL_WORKERS=$S_WORKERS" -e FORWARDED_ALLOW_IPS=127.0.0.1 -e STL_ALLOWED_HOSTS=127.0.0.1,localhost \
        -v "$VOLUME:/app/server/.cache" \
        --label "$LABEL=1" --label "$LABEL.dir=$INSTALL_DIR" "$1" 2>&1)
    local code=$?
    log "docker run $1 on 127.0.0.1:$2 -> $code $out"
    [ "$code" = "0" ] && return 0
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    case "$out" in *"port is already allocated"*|*"address already in use"*|*"bind"*) return 2 ;; esac
    printf '%s\n' "$out" >&2
    return 1
}

wait_health() {  # wait_health PORT
    local waited=0 body
    while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
        body=$(http_get "http://127.0.0.1:$1/api/health")
        case "$body" in *'"ok":true'*|*'"ok": true'*) return 0 ;; esac
        if ! container_running || [ "$(docker container inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null)" != "0" ]; then
            log "container stopped or crashed: $(docker logs --tail 30 "$CONTAINER" 2>&1)"
            return 1
        fi
        [ $((waited % 20)) -eq 0 ] && [ "$waited" -gt 0 ] && say "  시작을 기다리는 중... (${waited}s)" "  waiting for the server..."
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

start_on_port() {  # start_on_port IMAGE -> sets S_PORT
    local p code
    if [ -n "$PORT_ARG" ]; then
        run_container "$1" "$PORT_ARG"; code=$?
        [ "$code" = "0" ] && { S_PORT=$PORT_ARG; return 0; }
        [ "$code" = "2" ] && die "포트 $PORT_ARG 을(를) 다른 프로그램이 쓰고 있습니다." "Port $PORT_ARG is in use."
        return 1
    fi
    for p in ${S_PORT:-} $(seq "$PORT_MIN" "$PORT_MAX"); do
        if [ "$p" != "${S_PORT:-}" ] && port_in_use "$p"; then continue; fi
        run_container "$1" "$p"; code=$?
        [ "$code" = "0" ] && { S_PORT=$p; return 0; }
        [ "$code" = "2" ] || return 1
    done
    die "포트 ${PORT_MIN}–${PORT_MAX} 이 모두 사용 중입니다. --port 로 다른 번호를 지정하세요." "Ports $PORT_MIN-$PORT_MAX are all busy; use --port."
}

prune_images() {
    local t
    for t in $(docker images "$REPO" --format '{{.Tag}}' 2>/dev/null); do
        case "$t" in local-*) ;; *) continue ;; esac
        [ "$REPO:$t" = "${S_CURRENT_IMAGE:-}" ] && continue
        [ "$REPO:$t" = "${S_PREVIOUS_IMAGE:-}" ] && continue
        docker rmi "$REPO:$t" >> "$LOG_FILE" 2>&1 && log "pruned $REPO:$t"
    done
    docker image prune -f --filter "label=$LABEL=1" >> "$LOG_FILE" 2>&1
}

# ------------------------------------------------------------------------------------------ files
bin_ext() { if is_macos; then printf '.command'; else printf '.sh'; fi; }

write_bin() {
    local ext c
    ext=$(bin_ext)
    mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/logs"
    cat "$SCRIPT_PATH" > "$INSTALL_DIR/bin/stl-sim.sh" && chmod 755 "$INSTALL_DIR/bin/stl-sim.sh"
    for c in start stop open status update rollback uninstall logs; do
        # shellcheck disable=SC2016  # the wrapper expands $0 and $@ itself
        printf '#!/bin/bash\n# STL Simulator: %s\nexec bash "$(cd "$(dirname "$0")" && pwd -P)/stl-sim.sh" %s "$@"\n' "$c" "$c" \
            > "$INSTALL_DIR/bin/$c$ext"
        chmod 755 "$INSTALL_DIR/bin/$c$ext"
    done
}

write_version() {
    {
        printf 'STL Simulator (local)\n'
        printf 'version:     %s\n' "$S_CURRENT_VERSION"
        printf 'build id:    %s\n' "$S_CURRENT_BUILD"
        printf 'commit date: %s\n' "$KIT_COMMIT_DATE"
        printf 'kit built:   %s\n' "$KIT_BUILT"
        printf 'installed:   %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
        printf 'image:       %s\n' "$S_CURRENT_IMAGE"
        printf 'previous:    %s\n' "${S_PREVIOUS_IMAGE:-none}"
        printf 'address:     http://127.0.0.1:%s\n' "$S_PORT"
    } > "$INSTALL_DIR/version.txt"
}

write_readme() {
    local ext
    ext=$(bin_ext)
    {
        printf 'STL Simulator 사용법 (로컬 설치, 버전 %s)\n' "$S_CURRENT_VERSION"
        printf '==============================================\n\n'
        printf '주소: http://127.0.0.1:%s   (브라우저 주소창에 입력)\n' "$S_PORT"
        printf '설치 폴더: %s\n\n' "$INSTALL_DIR"
        if is_macos; then
            printf '열기:        이 폴더의 "STL Simulator 열기.command" 더블클릭\n'
        else
            printf '열기:        프로그램 메뉴의 "STL Simulator" 또는 bin/open%s\n' "$ext"
        fi
        printf '시작 / 중지: bin/start%s  /  bin/stop%s\n' "$ext" "$ext"
        printf '상태 확인:   bin/status%s\n' "$ext"
        printf '업데이트:    새 설치 키트를 받으면 그 키트의 설치 파일을 다시 실행 (설정·포트 유지)\n'
        printf '되돌리기:    bin/rollback%s  (바로 전 버전으로)\n' "$ext"
        printf '제거:        bin/uninstall%s\n' "$ext"
        printf '기록(로그):  logs 폴더\n\n'
        printf '%s\n' '- Docker가 켜져 있어야 합니다. 시뮬레이터는 Docker가 시작될 때 자동으로 함께 시작됩니다' \
            '  (중지 명령으로 멈춘 경우 제외). 쓰지 않을 때 메모리를 아끼려면 stop 을 실행하세요.' \
            '- 이 컴퓨터에서만 접속됩니다 (127.0.0.1). 다른 사람과 주소를 공유할 수 없습니다.' \
            '- 문제가 있으면 logs 폴더를 담당자에게 보내 주세요 (비밀번호는 기록되지 않습니다).' ''
        printf 'English: open http://127.0.0.1:%s ; bin/ has start, stop, open, status, rollback, uninstall;\n' "$S_PORT"
        printf 'to update, run the installer of a new kit.\n'
    } > "$INSTALL_DIR/README-사용법.txt"
}

desktop_dir() {
    local d=""
    command -v xdg-user-dir >/dev/null 2>&1 && d=$(xdg-user-dir DESKTOP 2>/dev/null)
    [ -n "$d" ] && [ "$d" != "$HOME" ] && [ -d "$d" ] && { printf '%s' "$d"; return 0; }
    [ -d "$HOME/Desktop" ] && printf '%s' "$HOME/Desktop"
}

make_shortcuts() {
    [ "$NO_SHORTCUT" = "1" ] && return 0
    if is_macos; then
        local f="$INSTALL_DIR/STL Simulator 열기.command"
        printf '#!/bin/bash\n# STL Simulator: start (if needed) and open in the browser\nSTL_PAUSE=1 exec bash "%s/bin/stl-sim.sh" open\n' \
            "$INSTALL_DIR" > "$f" && chmod 755 "$f" && say "실행 아이콘: $f" "Launcher: $f"
        return 0
    fi
    local apps="$HOME/.local/share/applications" esc desk
    # shellcheck disable=SC2016  # sed script: escape \ " ` $ for the desktop-entry Exec key
    esc=$(printf '%s' "$INSTALL_DIR/bin/stl-sim.sh" | sed 's/\\/\\\\/g; s/"/\\"/g; s/`/\\`/g; s/\$/\\$/g')
    mkdir -p "$apps" 2>/dev/null || return 0
    {
        printf '[Desktop Entry]\nType=Application\nName=STL Simulator\n'
        printf 'Comment=STL Simulator (local, http://127.0.0.1:%s)\n' "$S_PORT"
        printf 'Exec=env STL_PAUSE=1 bash "%s" open\nTerminal=true\nIcon=applications-science\nCategories=Science;Education;\n' "$esc"
    } > "$apps/stl-simulator.desktop" 2>/dev/null || return 0
    chmod 755 "$apps/stl-simulator.desktop"
    desk=$(desktop_dir)
    if [ -n "$desk" ]; then
        cp "$apps/stl-simulator.desktop" "$desk/stl-simulator.desktop" 2>/dev/null && \
            chmod 755 "$desk/stl-simulator.desktop" && \
            { command -v gio >/dev/null 2>&1 && gio set "$desk/stl-simulator.desktop" metadata::trusted true >/dev/null 2>&1; true; }
    fi
    say "프로그램 메뉴에 'STL Simulator'를 추가했습니다." "Added 'STL Simulator' to the applications menu."
}

remove_shortcuts() {
    local desk
    rm -f "$HOME/.local/share/applications/stl-simulator.desktop" 2>/dev/null
    desk=$(desktop_dir)
    [ -n "$desk" ] && rm -f "$desk/stl-simulator.desktop" 2>/dev/null
    return 0
}

# ------------------------------------------------------------------------------------------ commands
read_kit() {
    local f="$SCRIPT_DIR/kit-info.txt"
    KIT_VERSION=$(kv_get "$f" VERSION)
    KIT_BUILD_ID=$(kv_get "$f" BUILD_ID)
    KIT_COMMIT_DATE=$(kv_get "$f" COMMIT_DATE)
    KIT_BUILT=$(kv_get "$f" BUILT)
    KIT_ARCHIVE_SHA=$(kv_get "$f" ARCHIVE_SHA256)
    PAYLOAD="$SCRIPT_DIR/$(kv_get "$f" PAYLOAD)"
    case "$KIT_VERSION" in ''|*[!0-9A-Za-z._-]*) die "kit-info.txt가 올바르지 않습니다." "Bad kit-info.txt." ;; esac
    [ -f "$PAYLOAD" ] || die "설치 파일($PAYLOAD)이 없습니다. ZIP 압축을 전부 풀었는지 확인하세요." \
        "The payload file is missing; extract the whole zip."
    local want got
    want=$(kv_get "$f" PAYLOAD_SHA256)
    got=$(sha256_of "$PAYLOAD")
    [ -z "$want" ] || [ -z "$got" ] || [ "$want" = "$got" ] || die "설치 파일이 손상되었습니다 (다운로드가 덜 되었을 수 있음). 키트를 다시 받으세요." \
        "The payload is damaged or incomplete; download the kit again."
}

cmd_install() {
    [ "$KIT_MODE" = "1" ] || die "설치는 설치 키트의 install 파일로 실행하세요." "Run install from the installer kit."
    if [ "$(id -u)" = "0" ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        die "sudo 없이 실행하세요 (설치 파일과 바로가기가 root 계정에 만들어집니다). docker 권한이 없다면: sudo usermod -aG docker $SUDO_USER → 로그아웃 후 다시 로그인" \
            "Do not run the installer with sudo. For docker access: sudo usermod -aG docker $SUDO_USER, then log out and in."
    fi
    read_kit
    TMP_LOG=$(mktemp "${TMPDIR:-/tmp}/stl-install.XXXXXX") || TMP_LOG=""
    LOG_FILE="$TMP_LOG"
    printf '\n'
    say "STL Simulator 설치 (버전 $KIT_VERSION)" "STL Simulator installer (version $KIT_VERSION)"
    say "이 프로그램은: Docker 확인 → 비밀번호 확인 → 시뮬레이터 이미지 만들기(처음 5–15분) → 실행 → 브라우저 열기 순서로 진행합니다." \
        "Steps: check Docker -> check the password -> build the simulator image (first time 5-15 min) -> start -> open the browser."

    step "[1/6] Docker 확인" "Checking Docker"
    find_docker || offer_docker_install
    ensure_engine
    check_engine_version
    check_platform

    step "[2/6] 설치 위치" "Install folder"
    resolve_install_dir
    local existing=0 ans
    if ! is_our_dir "$INSTALL_DIR" && [ -z "$DIR_ARG" ] && [ "$YES" = "0" ] && tty_ok; then
        printf '설치 폴더 [%s]\n  Install folder (Enter = default): ' "$INSTALL_DIR"
        read -r ans < /dev/tty
        if [ -n "$ans" ]; then DIR_ARG="$ans"; resolve_install_dir; fi
    fi
    if is_our_dir "$INSTALL_DIR"; then
        load_state
        if [ -n "${S_CURRENT_IMAGE:-}" ]; then
            existing=1
            say "기존 설치를 찾았습니다 (버전 ${S_CURRENT_VERSION:-?}) → 업데이트/복구합니다: $INSTALL_DIR" \
                "Existing install found (version ${S_CURRENT_VERSION:-?}); updating/repairing: $INSTALL_DIR"
        else
            say "설치 폴더: $INSTALL_DIR (이전에 끝나지 않은 설치를 이어서 합니다)" "Install folder (resuming an unfinished install): $INSTALL_DIR"
        fi
    elif [ -d "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
        die "폴더가 비어 있지 않습니다: $INSTALL_DIR (다른 폴더를 지정하세요)" "The folder is not empty; choose another one (--install-dir)."
    else
        say "설치 폴더: $INSTALL_DIR" "Install folder: $INSTALL_DIR"
    fi
    if [ "$existing" = "1" ]; then check_disk 3; else check_disk 5; fi

    step "[3/6] 비밀번호 확인" "Password"
    HELPER_B64=$(helper_b64)
    ensure_helper_image
    get_password

    mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/logs" || die "폴더를 만들 수 없습니다: $INSTALL_DIR" "Cannot create the folder."
    printf 'STL Simulator install folder (used by the installer and uninstaller)\n' > "$INSTALL_DIR/$MARKER"
    LOG_FILE="$INSTALL_DIR/logs/install-$(now_stamp).log"
    if [ -n "$TMP_LOG" ]; then cat "$TMP_LOG" >> "$LOG_FILE" 2>/dev/null; rm -f "$TMP_LOG"; TMP_LOG=""; fi

    step "[4/6] 시뮬레이터 이미지 만들기" "Building the simulator image"
    local tag="$REPO:local-$KIT_VERSION" old_image="${S_CURRENT_IMAGE:-}" old_version="${S_CURRENT_VERSION:-}"
    build_image "$tag"

    step "[5/6] 실행" "Starting"
    S_WORKERS=$WORKERS
    if ! start_on_port "$tag" || ! wait_health "$S_PORT"; then
        docker logs --tail 40 "$CONTAINER" >> "$LOG_FILE" 2>&1
        if [ -n "$old_image" ] && [ "$old_image" != "$tag" ] && image_exists "$old_image"; then
            warn "새 버전이 시작되지 않아 이전 버전($old_version)으로 되돌립니다." "The new version did not start; rolling back to $old_version."
            if run_container "$old_image" "$S_PORT" && wait_health "$S_PORT"; then
                docker tag "$old_image" "$REPO:local" >/dev/null 2>&1
                docker rmi "$tag" >> "$LOG_FILE" 2>&1
                die "업데이트에 실패해 이전 버전으로 되돌렸습니다. logs 폴더를 담당자에게 보내 주세요." \
                    "The update failed; the previous version is running again. Please send the logs folder."
            fi
        fi
        docker rm -f "$CONTAINER" >/dev/null 2>&1
        die "시뮬레이터가 시작되지 않았습니다. logs 폴더를 담당자에게 보내 주세요." \
            "The simulator did not start (its output is in the install log in the logs folder)."
    fi
    if [ -n "$old_image" ] && [ "$old_image" != "$tag" ]; then
        S_PREVIOUS_IMAGE=$old_image; S_PREVIOUS_VERSION=$old_version
    fi
    S_CURRENT_IMAGE=$tag; S_CURRENT_VERSION=$KIT_VERSION; S_CURRENT_BUILD=$KIT_BUILD_ID
    # shellcheck disable=SC2034  # saved by save_state
    S_INSTALLED=$(date '+%Y-%m-%dT%H:%M:%S%z')
    save_state
    say "시뮬레이터가 실행 중입니다: http://127.0.0.1:$S_PORT" "The simulator is running: http://127.0.0.1:$S_PORT"

    step "[6/6] 마무리" "Finishing"
    write_bin
    write_version
    write_readme
    make_shortcuts
    prune_images
    local url="http://127.0.0.1:$S_PORT/"
    if [ "$NO_BROWSER" = "0" ]; then open_url "$url" || say "브라우저에서 $url 을 여세요." "Open $url in your browser."; fi
    local ext; ext=$(bin_ext)
    printf '\n'
    say "설치 완료! 주소: $url" "Done! Address: $url"
    say "  시작/중지: $INSTALL_DIR/bin/start$ext, stop$ext   상태: status$ext" "  start / stop / status scripts in $INSTALL_DIR/bin"
    say "  업데이트: 새 키트의 설치 파일 실행   되돌리기: bin/rollback$ext   제거: bin/uninstall$ext" \
        "  update: run a new kit's installer; rollback: bin/rollback$ext; uninstall: bin/uninstall$ext"
    say "  사용법: $INSTALL_DIR/README-사용법.txt" "  how-to: README-사용법.txt"
    pause_end
}

require_install() {
    resolve_install_dir
    is_our_dir "$INSTALL_DIR" || die "설치된 STL Simulator를 찾지 못했습니다 ($INSTALL_DIR)." "No installation found in $INSTALL_DIR."
    load_state
    LOG_FILE="$INSTALL_DIR/logs/control.log"
    mkdir -p "$INSTALL_DIR/logs" 2>/dev/null
}

cmd_start() {
    require_install
    find_docker || die "Docker를 찾지 못했습니다." "Docker CLI not found."
    ensure_engine
    if container_running; then :
    elif container_exists; then
        docker start "$CONTAINER" >> "$LOG_FILE" 2>&1 || die "컨테이너를 시작하지 못했습니다." "Could not start the container."
    else
        local img="${S_CURRENT_IMAGE:-$REPO:local}"
        image_exists "$img" || die "시뮬레이터 이미지가 없습니다. 설치 키트로 다시 설치하세요." "The image is missing; run the installer kit again."
        [ -n "${S_WORKERS:-}" ] || S_WORKERS=1
        start_on_port "$img" || die "컨테이너를 시작하지 못했습니다." "Could not start the container."
        save_state
    fi
    say "시작하는 중..." "Starting..."
    wait_health "$S_PORT" || die "시뮬레이터가 응답하지 않습니다. 'docker logs $CONTAINER'를 확인하세요." "The simulator does not respond."
    say "실행 중: http://127.0.0.1:$S_PORT" "Running: http://127.0.0.1:$S_PORT"
}

cmd_open() {
    cmd_start
    open_url "http://127.0.0.1:$S_PORT/" || say "브라우저에서 http://127.0.0.1:$S_PORT 을 여세요." "Open http://127.0.0.1:$S_PORT in your browser."
}

cmd_stop() {
    require_install
    if ! find_docker || ! engine_up; then
        say "Docker가 꺼져 있습니다 (이미 중지됨)." "Docker is not running (already stopped)."
        return 0
    fi
    docker stop "$CONTAINER" >> "$LOG_FILE" 2>&1
    say "중지했습니다. 다시 시작: bin/start$(bin_ext)" "Stopped. Start again with bin/start$(bin_ext)."
}

cmd_status() {
    require_install
    say "설치 폴더: $INSTALL_DIR" "Install folder"
    say "버전: ${S_CURRENT_VERSION:-?}  (이전: ${S_PREVIOUS_VERSION:-없음})" "Version (previous)"
    if ! find_docker || ! engine_up; then say "Docker: 꺼짐" "Docker is not running"; return 0; fi
    local st
    st=$(docker container inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)
    say "컨테이너: ${st:-없음}   주소: http://127.0.0.1:${S_PORT:-?}" "Container: ${st:-none}"
    if [ "$st" = "running" ]; then
        case "$(http_get "http://127.0.0.1:$S_PORT/api/health")" in
            *'"ok":true'*) say "상태: 정상" "Health: ok" ;;
            *) say "상태: 응답 없음 (시작 중일 수 있음)" "Health: no answer (maybe still starting)" ;;
        esac
    fi
}

cmd_logs() {
    require_install
    docker logs --tail "${1:-200}" "$CONTAINER"
}

cmd_rollback() {
    require_install
    find_docker || die "Docker를 찾지 못했습니다." "Docker CLI not found."
    ensure_engine
    local prev="${S_PREVIOUS_IMAGE:-}"
    image_exists "$prev" || die "되돌릴 이전 버전이 없습니다." "No previous version to roll back to."
    say "이전 버전(${S_PREVIOUS_VERSION})으로 되돌립니다..." "Rolling back to ${S_PREVIOUS_VERSION}..."
    [ -n "${S_WORKERS:-}" ] || S_WORKERS=1
    run_container "$prev" "$S_PORT" || die "이전 버전을 시작하지 못했습니다." "Could not start the previous version."
    wait_health "$S_PORT" || die "이전 버전이 응답하지 않습니다." "The previous version does not respond."
    docker tag "$prev" "$REPO:local" >> "$LOG_FILE" 2>&1
    local ci="$S_CURRENT_IMAGE" cv="$S_CURRENT_VERSION"
    S_CURRENT_IMAGE=$prev; S_CURRENT_VERSION=$S_PREVIOUS_VERSION
    S_PREVIOUS_IMAGE=$ci; S_PREVIOUS_VERSION=$cv
    S_CURRENT_BUILD=$(docker image inspect -f "{{index .Config.Labels \"$LABEL.build\"}}" "$prev" 2>/dev/null)
    save_state
    write_version
    write_readme
    say "되돌렸습니다: 버전 $S_CURRENT_VERSION, http://127.0.0.1:$S_PORT (한 번 더 실행하면 버전 $S_PREVIOUS_VERSION 으로 돌아갑니다)" \
        "Rolled back to $S_CURRENT_VERSION (run rollback again to return to $S_PREVIOUS_VERSION)."
}

cmd_update() {
    if [ "$ROLLBACK" = "1" ]; then cmd_rollback; return; fi
    require_install
    local kit="$KIT_ARG" s
    if [ -z "$kit" ]; then
        say "업데이트하려면 새 설치 키트(ZIP)의 압축을 풀고 그 안의 설치 파일을 실행하세요." \
            "To update, unzip the new installer kit and run its installer."
        say "또는: $INSTALL_DIR/bin/update$(bin_ext) <새 키트 폴더>   /   이전 버전으로: bin/rollback$(bin_ext)" \
            "or: bin/update <new kit folder>; previous version: bin/rollback"
        return 0
    fi
    for s in "$kit/installer-files/stl-local.sh" "$kit/stl-local.sh"; do
        if [ -f "$s" ]; then
            local fwd
            fwd=(install --install-dir "$INSTALL_DIR")
            [ "$YES" = "1" ] && fwd+=(--yes)
            [ "$PW_STDIN" = "1" ] && fwd+=(--password-stdin)
            [ "$NO_BROWSER" = "1" ] && fwd+=(--no-browser)
            [ "$NO_SHORTCUT" = "1" ] && fwd+=(--no-shortcut)
            [ -n "$PORT_ARG" ] && fwd+=(--port "$PORT_ARG")
            exec bash "$s" "${fwd[@]}"
        fi
    done
    die "키트 폴더가 아닙니다: $kit (ZIP이면 먼저 압축을 푸세요)" "Not a kit folder: $kit (unzip it first)."
}

cmd_uninstall() {
    require_install
    ask "STL Simulator를 제거할까요? ($INSTALL_DIR)" "Uninstall the STL Simulator?" n || { say "취소했습니다." "Cancelled."; return 0; }
    if find_docker && engine_up; then
        docker rm -f "$CONTAINER" >/dev/null 2>&1
        local img
        for img in $(docker images "$REPO" --format '{{.Repository}}:{{.Tag}}' 2>/dev/null); do
            case "$img" in "$REPO:<none>") continue ;; esac
            docker rmi -f "$img" >/dev/null 2>&1 && say "  이미지 삭제: $img" "  removed $img"
        done
        docker image prune -f --filter "label=$LABEL=1" >/dev/null 2>&1
        local rm_cache=1
        case "$CACHE_CHOICE" in
            keep) rm_cache=0 ;;
            remove) rm_cache=1 ;;
            *) ask "계산 결과 캐시(Docker 볼륨 $VOLUME)도 지울까요?" "Also delete the result cache volume?" y || rm_cache=0 ;;
        esac
        if [ "$rm_cache" = "1" ]; then docker volume rm "$VOLUME" >/dev/null 2>&1 && say "  캐시 삭제: $VOLUME" "  removed volume $VOLUME"; fi
        prune_build_context
        local rm_bc=1
        case "$BUILD_CACHE_CHOICE" in
            keep) rm_bc=0 ;;
            remove) rm_bc=1 ;;
            *) ask "Docker 빌드 캐시도 지울까요? (시뮬레이터 프로그램의 사본이 들어 있습니다. 다른 Docker 프로젝트의 빌드 캐시도 함께 지워집니다)" \
                   "Also delete Docker's build cache? It holds a copy of the program (other projects' build cache goes too)." y || rm_bc=0 ;;
        esac
        if [ "$rm_bc" = "1" ]; then
            if docker buildx version >/dev/null 2>&1; then docker buildx prune -af >/dev/null 2>&1
            else docker builder prune -af >/dev/null 2>&1; fi && say "  빌드 캐시 삭제" "  removed Docker's build cache"
        else
            BC_KEPT=1
        fi
    else
        warn "Docker가 꺼져 있어 컨테이너/이미지는 지우지 못했습니다. Docker를 켜고 'docker rm -f $CONTAINER; docker rmi $REPO:local'" \
             "Docker is not running; remove the container and images later."
    fi
    remove_shortcuts
    local dir="$INSTALL_DIR"
    LOG_FILE=""
    if is_our_dir "$dir"; then rm -rf "$dir" && say "  폴더 삭제: $dir" "  removed $dir"; fi
    say "제거했습니다. 기본 이미지 python:3.11-slim, node:22-slim 은 남겨 둡니다 (프로그램은 들어 있지 않음)." \
        "Uninstalled. The base images python:3.11-slim and node:22-slim are kept (they hold no part of the program)."
    if [ "${BC_KEPT:-0}" = "1" ]; then
        say "Docker 빌드 캐시에는 아직 프로그램 사본이 남아 있습니다. 지우려면: docker buildx prune -af" \
            "Docker's build cache still holds a copy of the program; delete it with: docker buildx prune -af"
    fi
    pause_end
}

usage() {
    sed -n '2,17p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
}

# ------------------------------------------------------------------------------------------ main
if [ "$KIT_MODE" = "1" ]; then COMMAND=install; else COMMAND=help; fi
case "${1:-}" in
    install|start|stop|restart|open|status|logs|rollback|update|uninstall|help) COMMAND=$1; shift ;;
esac
while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y) YES=1 ;;
        --port) PORT_ARG=${2:-}; shift ;;
        --port=*) PORT_ARG=${1#--port=} ;;
        --install-dir) DIR_ARG=${2:-}; shift ;;
        --install-dir=*) DIR_ARG=${1#--install-dir=} ;;
        --password-stdin) PW_STDIN=1 ;;
        --no-browser) NO_BROWSER=1 ;;
        --no-shortcut) NO_SHORTCUT=1 ;;
        --remove-cache) CACHE_CHOICE=remove ;;
        --keep-cache) CACHE_CHOICE=keep ;;
        --remove-build-cache) BUILD_CACHE_CHOICE=remove ;;
        --keep-build-cache) BUILD_CACHE_CHOICE=keep ;;
        --rollback) ROLLBACK=1 ;;
        -h|--help) COMMAND=help ;;
        -*) die "알 수 없는 옵션: $1" "Unknown option: $1" ;;
        *) KIT_ARG=$1 ;;
    esac
    shift
done
case "$PORT_ARG" in ''|[0-9]|[0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9][0-9][0-9]) ;;
    *) die "--port 는 숫자여야 합니다." "--port must be a number." ;; esac
trap 'PW=""; [ -n "$TMP_LOG" ] && rm -f "$TMP_LOG"' EXIT
trap 'PW=""; printf "\n"; exit 130' INT TERM

case "$COMMAND" in
    install) cmd_install ;;
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    open) cmd_open ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    rollback) cmd_rollback ;;
    update) cmd_update ;;
    uninstall) cmd_uninstall ;;
    *) usage ;;
esac
