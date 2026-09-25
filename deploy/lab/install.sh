#!/usr/bin/env bash
# STL simulator — one-time setup of a lab Linux server (Ubuntu/Debian or Rocky/RHEL). Guide: docs/DEPLOY_LAB.md
#
#   sudo bash install.sh                      interactive: asks for the branch, the site address and the password
#   sudo bash /opt/stl-sim/bin/install.sh     re-run later with the installed copy (settings are kept)
#   sudo bash install.sh --help               all options (non-interactive use, custom paths and ports)
#
# Steps: check docker + compose + git · create the service user (default "stl", member of the docker group) ·
# create a read-only deploy key and show where to add it on GitHub · clone the repository into $STL_HOME/app ·
# write $STL_HOME/stl.env (mode 600) with the password (typed, never shown) and a random session secret · install the
# deploy kit (updater = command `stl-lab`, compose file, Caddyfile, systemd units) root-owned into $STL_HOME/bin and
# $STL_HOME/kit · first deploy · print the URL and how to check the status.
#
# The deploy kit is pinned: pushes change the application, never the kit. A re-run shows what changed in the kit
# since it was installed (a diff), checks it, and asks before installing it — this is the review step for changes to
# docker-compose.yml, the Caddyfile or the updater (whoever controls those controls the server).
#
# Safe to re-run: the checkout, stl.env (password, session secret, settings) and the deploy key are kept; missing
# values are filled in, options given on the command line replace the stored ones. Use --reset-password to change
# the password. A re-run respects a rollback and --pause (it does not force-deploy the branch head).

set -Eeuo pipefail
umask 027

DEFAULT_REPO=git@github.com:JunhyoungPark-NOBEL/Single-Transistor-Latch-simulator-SOI-.git
LAB=deploy/lab
KIT_FILES="docker-compose.yml Caddyfile tailscale-funnel.json update.sh install.sh systemd/stl-update.service systemd/stl-update.timer systemd/stl-compose.service systemd/stl-refresh.service systemd/stl-refresh.timer"
UNITS="stl-update.service stl-update.timer stl-compose.service stl-refresh.service stl-refresh.timer"

# GitHub's SSH host keys (https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
# — SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU and SHA256:p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM).
# ssh.github.com:443 (SSH over the HTTPS port, for networks that block outbound port 22) serves the same keys.
GITHUB_KNOWN_HOSTS='github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
github.com ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=
[ssh.github.com]:443 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl
[ssh.github.com]:443 ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg='

# the checksum of this very file, taken before anything else (a re-exec'ed copy deletes its temporary file)
SELF_SUM=$(sha256sum <"${BASH_SOURCE[0]}" 2>/dev/null | cut -c1-64 || true)
REEXEC=0 KIT_LABEL=""
if [[ -n ${STL_INSTALL_REEXEC-} ]]; then
	rm -f -- "$STL_INSTALL_REEXEC"
	REEXEC=1 KIT_LABEL=${STL_INSTALL_KIT_LABEL-}
	unset STL_INSTALL_REEXEC STL_INSTALL_KIT_LABEL
fi
ORIG_ARGS=("$@")

# ------------------------------------------------------------------------------------------------ output
if [[ -t 1 ]]; then B=$'\e[1m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'; else B='' G='' Y='' R='' N=''; fi
say() { printf '%s\n' "$*"; }
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok() { printf '%s  ✓ %s%s\n' "$G" "$*" "$N"; }
warn() { printf '%s  ! %s%s\n' "$Y" "$*" "$N" >&2; }
die() {
	printf '%s  ✗ %s%s\n' "$R" "$*" "$N" >&2
	exit 1
}
trap 'printf "%s  ✗ unexpected error (line %s)%s\n" "$R" "$LINENO" "$N" >&2' ERR

usage() {
	cat <<'EOF'
Usage: sudo bash install.sh [options]

  --home DIR            installation directory (default /opt/stl-sim)
  --user NAME           service user (default stl; created if missing and added to the docker group).
                        --user "$SUDO_USER" uses your own account instead.
  --repo URL            repository (default the SSH URL of the GitHub repository; file:// or https:// also work;
                        port 22 blocked: ssh://git@ssh.github.com:443/<owner>/<repo>.git)
  --branch NAME         deploy branch (default main; claude/stl-simulator-web-j0yy9i until PR #1 is merged)
  --site ADDRESS        site name(s) for the proxy, comma-separated (DNS name, campus hostname or IP)
  --tls acme|internal   acme = Let's Encrypt (real domain, ports 80/443 open to the internet);
                        internal = Caddy's own CA (campus only / IP address). Default internal.
  --http-port N --https-port N --bind ADDR     host ports / address of the proxy (default 80, 443, 0.0.0.0)
  --workers N           compute processes (default: CPUs − 1, at most 8, limited by memory)
  --app-uid N           uid/gid of the app container (default: the first free one from 61000)
  --project NAME --image NAME --subnet CIDR --debug-port N --compose-override FILE      advanced (stl.env §6)
  --kit-from REF        which commit's deploy kit to install: head (default: the deploy branch's newest commit),
                        deployed (the commit that is live) or a commit id
  --password-stdin      read the access password from standard input (one line) instead of asking
  --reset-password      ask for a new access password even if one is set (signs everyone out)
  --non-interactive     never prompt (defaults for everything not given; needs --password-stdin on first run;
                        kit changes are installed without asking — review them yourself first)
  --no-systemd          do not install systemd units (containers without systemd, no sudo: use cron instead)
  --no-deploy           set everything up but do not deploy
  -h, --help            this help
EOF
}

# ------------------------------------------------------------------------------------------------ options
STL_HOME=/opt/stl-sim SVC_USER="" INTERACTIVE=1 PASSWORD_STDIN=0 RESET_PASSWORD=0 NO_SYSTEMD=0 NO_DEPLOY=0 KIT_FROM=""
declare -A OPT=() # stl.env keys given on the command line
while (($#)); do
	case $1 in
	--home) STL_HOME=${2:?--home needs a value} && shift ;;
	--user) SVC_USER=${2:?--user needs a value} && shift ;;
	--repo) OPT[STL_REPO_URL]=${2:?} && shift ;;
	--branch) OPT[DEPLOY_BRANCH]=${2:?} && shift ;;
	--site) OPT[STL_SITE_ADDRESS]=${2:?} && shift ;;
	--tls) OPT[STL_TLS_MODE]=${2:?} && shift ;;
	--http-port) OPT[STL_HTTP_PORT]=${2:?} && shift ;;
	--https-port) OPT[STL_HTTPS_PORT]=${2:?} && shift ;;
	--bind) OPT[STL_BIND_ADDR]=${2:?} && shift ;;
	--workers) OPT[STL_WORKERS]=${2:?} && shift ;;
	--app-uid) OPT[STL_APP_UID]=${2:?} && shift ;;
	--project) OPT[STL_PROJECT]=${2:?} && shift ;;
	--image) OPT[STL_IMAGE]=${2:?} && shift ;;
	--subnet) OPT[STL_NET_SUBNET]=${2:?} && shift ;;
	--debug-port) OPT[STL_DEBUG_PORT]=${2:?} && shift ;;
	--compose-override) OPT[STL_COMPOSE_OVERRIDE]=${2:?} && shift ;;
	--kit-from) KIT_FROM=${2:?} && shift ;;
	--password-stdin) PASSWORD_STDIN=1 ;;
	--reset-password) RESET_PASSWORD=1 ;;
	--non-interactive | -y) INTERACTIVE=0 ;;
	--no-systemd) NO_SYSTEMD=1 ;;
	--no-deploy) NO_DEPLOY=1 ;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		usage >&2
		die "unknown option: $1"
		;;
	esac
	shift
done
{ : </dev/tty; } 2>/dev/null || INTERACTIVE=0 # no terminal (cron, CI, pipes without a tty): never prompt

[[ $STL_HOME == /* ]] || STL_HOME=$PWD/$STL_HOME
[[ $STL_HOME =~ ^/[A-Za-z0-9._/-]+$ ]] || die "--home must be an absolute path of letters, digits, . _ - /"
STL_HOME=${STL_HOME%/}
# a neutral working directory: the service user may not be able to read the caller's (e.g. a mode-750 home), and
# git then fails with misleading errors
cd /
APP=$STL_HOME/app
ENV_FILE=$STL_HOME/stl.env
STATE=$STL_HOME/state
# re-run without --user: the service user of the existing installation (owner of state/), else "stl"
if [[ -z $SVC_USER ]]; then
	SVC_USER=stl
	if [[ -d $STATE ]]; then SVC_USER=$(stat -c %U "$STATE"); fi
fi
SUDO=""
[[ $EUID -ne 0 ]] || SUDO="sudo "

# ------------------------------------------------------------------------------------------------ helpers
ask() { # ask PROMPT DEFAULT → answer on stdout
	local answer
	if ((!INTERACTIVE)); then
		printf '%s' "$2"
		return 0
	fi
	read -r -p "  $1 [$2]: " answer </dev/tty || true
	printf '%s' "${answer:-$2}"
}
confirm() { # confirm PROMPT (default yes)
	local answer
	((INTERACTIVE)) || return 0
	read -r -p "  $1 [Y/n]: " answer </dev/tty || true
	[[ ! $answer =~ ^[Nn] ]]
}

as_user() { # run a command as the service user, with that user's HOME
	if [[ $(id -un) == "$SVC_USER" ]]; then
		"$@"
	elif command -v runuser >/dev/null 2>&1; then
		runuser -u "$SVC_USER" -- env HOME="$SVC_HOME" "$@"
	else
		sudo -u "$SVC_USER" -H -- "$@"
	fi
}

version_ge() { # version_ge HAVE NEED
	[[ $(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1) == "$2" ]]
}

env_get() { # same parser as update.sh
	local key=$1 def=${2-} line val
	line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$ENV_FILE" 2>/dev/null | tail -n 1 || true)
	line=${line%$'\r'}
	if [[ -z $line ]]; then
		printf '%s' "$def"
		return 0
	fi
	val=${line#*=}
	val=${val#"${val%%[![:space:]]*}"}
	case $val in
	\'*)
		val=${val#\'}
		val=${val%%\'*}
		;;
	\"*)
		val=${val#\"}
		val=${val%%\"*}
		;;
	*)
		val=${val%%[[:space:]]#*}
		val=${val%"${val##*[![:space:]]}"}
		;;
	esac
	[[ -n $val ]] || val=$def
	printf '%s' "$val"
}

# env_set KEY VALUE: replace (or append) KEY='VALUE' in stl.env. Pure bash (printf is a builtin), so a value never
# appears in any process's argument list; the file is rewritten through a mode-600 temporary file.
env_set() {
	local key=$1 val=$2 tmp line found=0
	[[ $val != *"'"* ]] && [[ ! $val =~ [[:cntrl:]] ]] || die "the value of $key must not contain a single quote or control characters"
	tmp=$(mktemp "$ENV_FILE.XXXXXX")
	chmod 600 "$tmp"
	while IFS= read -r line || [[ -n $line ]]; do
		if [[ $line =~ ^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*= ]]; then
			((found)) && continue
			found=1
			if [[ -n $val ]]; then printf "%s='%s'\n" "$key" "$val" >>"$tmp"; else printf '%s=\n' "$key" >>"$tmp"; fi
		else
			printf '%s\n' "$line" >>"$tmp"
		fi
	done <"$ENV_FILE"
	if ((!found)); then printf "%s='%s'\n" "$key" "$val" >>"$tmp"; fi
	chown "$SVC_USER:$SVC_GROUP" "$tmp"
	mv -f "$tmp" "$ENV_FILE"
}

check_password() {
	local p=$1
	[[ -n $p ]] || {
		warn "비밀번호가 비어 있습니다 / the password is empty"
		return 1
	}
	((${#p} >= 12)) || {
		warn "12자 이상이어야 합니다(16자 이상 권장) / use at least 12 characters (16+ recommended)"
		return 1
	}
	[[ $p != *"'"* ]] || {
		warn "작은따옴표(')는 쓸 수 없습니다 / a single quote (') is not allowed"
		return 1
	}
	[[ ! $p =~ [[:cntrl:]] ]] && [[ $p == "${p#[[:space:]]}" ]] && [[ $p == "${p%[[:space:]]}" ]] || {
		warn "앞뒤 공백·제어 문자는 쓸 수 없습니다 / no leading/trailing spaces or control characters"
		return 1
	}
	((${#p} >= 16)) || warn "16자 이상을 권장합니다 / 16+ characters are recommended"
	return 0
}

PASSWORD=""
read_password() {
	local p1 p2
	if ((PASSWORD_STDIN)); then
		IFS= read -r p1 || true
		check_password "$p1" || die "the password from stdin was rejected"
		PASSWORD=$p1
		return 0
	fi
	((INTERACTIVE)) || die "no password set: use --password-stdin (or run interactively)"
	say "  링크와 이 비밀번호를 아는 사람만 접속할 수 있습니다. 입력한 글자는 화면에 보이지 않습니다."
	say "  Anyone with the link AND this password can use the site. Typing is not echoed."
	while :; do
		IFS= read -rs -p "  접속 비밀번호 / access password: " p1 </dev/tty || true
		printf '\n' >/dev/tty
		check_password "$p1" || continue
		IFS= read -rs -p "  한 번 더 / once more: " p2 </dev/tty || true
		printf '\n' >/dev/tty
		[[ $p1 == "$p2" ]] && break
		warn "두 입력이 다릅니다 / the two entries differ"
	done
	PASSWORD=$p1
}

new_secret() {
	if command -v openssl >/dev/null 2>&1; then
		openssl rand -hex 32
	else
		head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
	fi
}

ssh_host_of() { # git@host:path | ssh://[user@]host[:port]/path → "host port" (port empty = 22)
	local u=$1 port=""
	case $u in
	ssh://*)
		u=${u#ssh://}
		u=${u#*@}
		u=${u%%/*}
		if [[ $u == *:* ]]; then port=${u##*:} && u=${u%%:*}; fi
		printf '%s %s' "$u" "$port"
		;;
	*://*) ;;
	*@*:*)
		u=${u#*@}
		printf '%s ' "${u%%:*}"
		;;
	esac
}
github_path_of() { # owner/repo of a github.com URL
	local u=$1
	u=${u%.git}
	u=${u#*github.com}
	u=${u#:443}
	u=${u#[:/]}
	printf '%s' "$u"
}

# own_dir MODE OWNER:GROUP DIR…: create or fix directories (without root: owned by the invoking user)
own_dir() {
	local mode=$1 og=$2
	shift 2
	if [[ $EUID -eq 0 ]]; then
		install -d -m "$mode" -o "${og%%:*}" -g "${og#*:}" "$@"
	else
		install -d -m "$mode" "$@"
	fi
}
# put_file MODE SRC DEST: install a kit file (root-owned when run as root)
put_file() {
	if [[ $EUID -eq 0 ]]; then install -m "$1" -o root -g root "$2" "$3"; else install -m "$1" "$2" "$3"; fi
}

# ------------------------------------------------------------------------------------------------ 1. checks
step "1/7 점검 / checks"
if [[ $EUID -ne 0 ]]; then
	if ((NO_SYSTEMD)) && [[ $SVC_USER == "$(id -un)" ]]; then
		warn "running without root: no service user, no systemd units (use cron, printed at the end)"
	else
		die "run with sudo: sudo bash $0 …   (or without root: --no-systemd --user $(id -un) --home ~/stl-sim)"
	fi
fi
OS_ID=unknown OS_LIKE=""
if [[ -r /etc/os-release ]]; then
	# shellcheck disable=SC1091
	OS_ID=$(. /etc/os-release && printf '%s' "${ID:-unknown}")
	# shellcheck disable=SC1091
	OS_LIKE=$(. /etc/os-release && printf '%s' "${ID_LIKE:-}")
fi
case "$OS_ID $OS_LIKE" in
*debian* | *ubuntu*) PKG="apt-get install -y" ;;
*rhel* | *fedora* | *rocky* | *centos* | *almalinux*) PKG="dnf install -y" ;;
*) PKG="(package manager) install" ;;
esac
ok "OS: $OS_ID"

missing=()
for c in git flock timeout runuser sha256sum; do command -v "$c" >/dev/null 2>&1 || missing+=("$c"); done
if ((${#missing[@]})); then
	die "missing commands: ${missing[*]} → sudo $PKG git openssh-client util-linux coreutils   (Rocky: openssh-clients)"
fi
command -v python3 >/dev/null 2>&1 || warn "python3 not found: the updater's safety check of the compose configuration is skipped → sudo $PKG python3"
command -v docker >/dev/null 2>&1 ||
	die "Docker is not installed → https://docs.docker.com/engine/install/ (Ubuntu: 'Install using the apt repository'; Rocky: the RHEL/CentOS page), then: sudo systemctl enable --now docker"
docker info >/dev/null 2>&1 || die "the Docker daemon is not reachable → sudo systemctl enable --now docker   (without sudo: an admin must add you to the docker group)"
dver=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 0)
version_ge "$dver" 27.0 || die "Docker Engine $dver is too old (need ≥ 27; install Docker CE from docs.docker.com, not the distribution's docker.io/podman)"
cver=$(docker compose version --short 2>/dev/null || echo 0)
cver=${cver#v}
version_ge "$cver" 2.24 || die "docker compose plugin missing or too old ($cver; need ≥ 2.24) → sudo $PKG docker-compose-plugin"
docker buildx version >/dev/null 2>&1 || die "docker buildx plugin missing → sudo $PKG docker-buildx-plugin"
ok "docker $dver, compose $cver, buildx, git"
cpus=$(nproc)
mem_mb=$(awk '/MemTotal/ { print int($2 / 1024) }' /proc/meminfo)
disk_gb=$(df -Pk "$(dirname "$STL_HOME")" | awk 'NR == 2 { print int($4 / 1048576) }')
ok "CPUs $cpus, memory $((mem_mb / 1024)) GB, free disk $disk_gb GB"
((mem_mb >= 1800)) || warn "less than 2 GB of memory: use --workers 1"
((disk_gb >= 10)) || warn "less than 10 GB free disk: images and build cache need ~5–10 GB"

# ------------------------------------------------------------------------------------------------ 2. user
step "2/7 서비스 계정 / service user"
if ! id "$SVC_USER" >/dev/null 2>&1; then
	[[ $EUID -eq 0 ]] || die "user $SVC_USER does not exist"
	confirm "create the system user '$SVC_USER' (home $STL_HOME/home) that runs the site?" || die "aborted"
	nologin=$(command -v nologin || echo /usr/sbin/nologin)
	useradd --system --user-group --home-dir "$STL_HOME/home" --no-create-home --shell "$nologin" "$SVC_USER"
	ok "created user $SVC_USER"
fi
SVC_GROUP=$(id -gn "$SVC_USER")
SVC_HOME=$(getent passwd "$SVC_USER" | cut -d: -f6)
if [[ $SVC_USER != root ]] && ! id -nG "$SVC_USER" | tr ' ' '\n' | grep -qx docker; then
	[[ $EUID -eq 0 ]] || die "$SVC_USER is not in the docker group (an admin runs: sudo usermod -aG docker $SVC_USER, then log in again)"
	getent group docker >/dev/null || die "there is no 'docker' group (install Docker CE)"
	usermod -aG docker "$SVC_USER"
	ok "added $SVC_USER to the docker group"
fi
[[ $SVC_USER == root ]] && warn "running the site as root works but a dedicated user (default: stl) is safer"
# Layout. Root-owned (the service user cannot change them): $STL_HOME itself, bin/ (updater, installer) and kit/
# (compose file, Caddyfile, units). Service user: the git checkouts, state, logs, the deploy key, stl.env.
own_dir 750 "root:$SVC_GROUP" "$STL_HOME"
own_dir 755 root:root "$STL_HOME/bin" "$STL_HOME/kit"
own_dir 750 "$SVC_USER:$SVC_GROUP" "$APP" "$STL_HOME/build" "$STATE" "$STL_HOME/logs"
own_dir 700 "$SVC_USER:$SVC_GROUP" "$STL_HOME/ssh"
if [[ $SVC_HOME == "$STL_HOME/home" ]]; then own_dir 750 "$SVC_USER:$SVC_GROUP" "$SVC_HOME"; fi
[[ -d $SVC_HOME ]] || SVC_HOME=$STL_HOME/home
ok "user $SVC_USER, directory $STL_HOME"

# F3: the app container's uid must not be an account on this host
uid=${OPT[STL_APP_UID]:-$(env_get STL_APP_UID "")}
if [[ -z $uid ]]; then
	uid=61000
	while getent passwd "$uid" >/dev/null || getent group "$uid" >/dev/null; do uid=$((uid + 1)); done
	OPT[STL_APP_UID]=$uid
fi
{ [[ $uid =~ ^[0-9]+$ ]] && ((uid >= 1000)); } || die "--app-uid must be a number ≥ 1000"
if getent passwd "$uid" >/dev/null || getent group "$uid" >/dev/null; then
	die "uid/gid $uid (STL_APP_UID) belongs to an account on this server ($(getent passwd "$uid" | cut -d: -f1)$(getent group "$uid" | cut -d: -f1 | sed 's/^/ group /')): that account could tamper with the app — choose a free one with --app-uid"
fi
ok "app container uid/gid $uid (no account on this server)"

# ------------------------------------------------------------------------------------------------ 3. repository access
step "3/7 저장소 접근 / repository access"
REPO_URL=${OPT[STL_REPO_URL]:-$(env_get STL_REPO_URL "")}
if [[ -z $REPO_URL ]]; then REPO_URL=$(ask "repository URL" "$DEFAULT_REPO"); fi
BRANCH=${OPT[DEPLOY_BRANCH]:-$(env_get DEPLOY_BRANCH "")}
if [[ -z $BRANCH ]]; then
	say "  배포 브랜치: 이 브랜치에 push된 커밋이 자동으로 공개됩니다. 권장: PR #1 병합 후 main."
	say "  (병합 전에는 claude/stl-simulator-web-j0yy9i — 그 브랜치에 push하는 모든 작업이 곧바로 공개됩니다)"
	say "  Deploy branch: commits pushed to it go live automatically. Recommended: main, after merging PR #1."
	BRANCH=$(ask "deploy branch" main)
fi
if ! out=$(git check-ref-format --branch "$BRANCH" 2>&1); then die "invalid branch name: $BRANCH ($out)"; fi

KEY=$STL_HOME/ssh/deploy_key
KNOWN=$STL_HOME/ssh/known_hosts
GIT_SSH_CMD="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20"
read -r host sport <<<"$(ssh_host_of "$REPO_URL")" || true
LS_ERR=""
repo_ok() { # can the service user read the branch? (the error, if any, is kept in LS_ERR)
	local out
	if [[ -n $host ]]; then
		out=$(as_user env GIT_SSH_COMMAND="$GIT_SSH_CMD" GIT_TERMINAL_PROMPT=0 timeout 60 git ls-remote --heads "$REPO_URL" "$BRANCH" 2>&1) || {
			LS_ERR=$out
			return 1
		}
	else
		out=$(as_user env GIT_TERMINAL_PROMPT=0 timeout 60 git ls-remote --heads "$REPO_URL" "$BRANCH" 2>&1) || {
			LS_ERR=$out
			return 1
		}
	fi
	[[ $out == *"refs/heads/$BRANCH"* ]] || {
		LS_ERR="the repository has no branch '$BRANCH'"
		return 1
	}
}
show_ls_err() { printf '%s\n' "$LS_ERR" | tail -n 4 | sed -e 's/^/      git: /' >&2; }
if [[ -n $host ]]; then
	{ command -v ssh-keygen && command -v ssh; } >/dev/null 2>&1 ||
		die "ssh is missing → sudo $PKG openssh-client   (Rocky: openssh-clients)"
	if [[ ! -f $KEY ]]; then
		as_user ssh-keygen -q -t ed25519 -N '' -C "stl-lab-deploy@$(hostname -s)" -f "$KEY"
		ok "created the deploy key $KEY"
	fi
	if [[ $host == github.com ]] || [[ $host == ssh.github.com ]]; then
		printf '%s\n' "$GITHUB_KNOWN_HOSTS" | as_user tee "$KNOWN" >/dev/null
	elif [[ ! -s $KNOWN ]]; then
		command -v ssh-keyscan >/dev/null 2>&1 || die "ssh-keyscan is missing → sudo $PKG openssh-client"
		# shellcheck disable=SC2016 # positional parameters of the inner shell
		as_user sh -c 'ssh-keyscan -T 10 -p "$1" "$2" > "$3" 2>/dev/null' _ "${sport:-22}" "$host" "$KNOWN" || true
		[[ -s $KNOWN ]] || die "cannot read the SSH host key of $host (port ${sport:-22})"
		warn "trusting the SSH host key of $host on first use:"
		ssh-keygen -lf "$KNOWN" | sed -e 's/^/      /'
		confirm "is this the right fingerprint?" || die "aborted"
	fi
	chmod 600 "$KEY" "$KNOWN"
	chown "$SVC_USER:$SVC_GROUP" "$KNOWN"
	if ! repo_ok; then
		path=$(github_path_of "$REPO_URL")
		say ""
		warn "no read access yet:"
		show_ls_err
		if [[ $LS_ERR == *"timed out"* ]] || [[ $LS_ERR == *"Connection refused"* ]] || [[ $LS_ERR == *"resolve"* ]]; then
			warn "this looks like a network problem, not the key: outbound SSH to $host:${sport:-22} is blocked?"
			warn "  try port 443: --repo ssh://git@ssh.github.com:443/$path.git   (docs/DEPLOY_LAB.md §7)"
		fi
		say ""
		say "  ${B}배포 키 등록 / add the deploy key (read-only)${N}"
		if [[ $host == *github.com ]]; then
			say "   1. 열기 / open:  https://github.com/$path/settings/keys/new"
		else
			say "   1. 저장소의 deploy key 설정 페이지를 엽니다 / open the repository's deploy-key settings on $host"
		fi
		say "   2. Title:  stl-lab-server ($(hostname -s))"
		say "   3. Key:    아래 한 줄 전체 / the whole line below"
		say ""
		say "      $(cat "$KEY.pub")"
		say ""
		say "   4. 'Allow write access'는 체크하지 않습니다 / leave 'Allow write access' UNCHECKED (read-only)"
		say "   5. Add key"
		((INTERACTIVE)) || die "the deploy key is not accepted yet; add it and re-run install.sh"
		until repo_ok; do
			read -r -p "  추가했으면 Enter (그만두려면 q) / press Enter when added (q to quit): " a </dev/tty || true
			[[ $a == q ]] && die "stopped; re-run install.sh after adding the key"
			repo_ok || {
				warn "still no access to $REPO_URL (branch $BRANCH):"
				show_ls_err
			}
		done
	fi
	ok "read access to $REPO_URL ($BRANCH) with the deploy key"
else
	repo_ok || {
		show_ls_err
		die "cannot read branch '$BRANCH' of $REPO_URL"
	}
	ok "read access to $REPO_URL ($BRANCH)"
fi

# ------------------------------------------------------------------------------------------------ 4. checkout
step "4/7 코드 받기 / checkout"
if [[ ! -d $APP/.git ]]; then
	[[ -z $(ls -A "$APP") ]] || die "$APP exists and is not a git checkout"
	as_user env GIT_TERMINAL_PROMPT=0 git -c core.sshCommand="$GIT_SSH_CMD" clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP"
	as_user git -C "$APP" checkout --quiet --detach
	ok "cloned into $APP"
fi
if [[ -n $host ]]; then as_user git -C "$APP" config core.sshCommand "$GIT_SSH_CMD"; fi
as_user git -C "$APP" remote set-url origin "$REPO_URL"
as_user env GIT_TERMINAL_PROMPT=0 git -C "$APP" fetch --quiet --no-tags origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
HEAD_REF=refs/remotes/origin/$BRANCH
as_user git -C "$APP" cat-file -e "$HEAD_REF:$LAB/update.sh" 2>/dev/null ||
	die "branch '$BRANCH' does not contain $LAB/ yet (main still has the old Streamlit app): merge PR #1 first, or use --branch claude/stl-simulator-web-j0yy9i"
ok "branch $BRANCH at $(as_user git -C "$APP" rev-parse --short=12 "$HEAD_REF")"
# the kit version 1 layout kept stl.env inside the checkout: move it out of the pushed tree
if [[ ! -f $ENV_FILE ]] && [[ -f $APP/$LAB/stl.env ]]; then
	mv "$APP/$LAB/stl.env" "$ENV_FILE"
	ok "moved stl.env out of the git checkout to $ENV_FILE"
fi
if [[ -L $APP/$LAB/.env ]]; then rm -f "$APP/$LAB/.env"; fi

# ------------------------------------------------------------------------------------------------ 4b. deploy kit: which version, what changed
DEPLOYED=$(cat "$STATE/deployed" 2>/dev/null || true)
# default: the branch head — the kit that belongs to the code the updater deploys next. It is shown as a diff,
# checked (bash -n, compose policy, caddy validate) and only then installed.
[[ -n $KIT_FROM ]] || KIT_FROM="head"
case $KIT_FROM in
head) KIT_REF=$HEAD_REF ;;
deployed)
	[[ -n $DEPLOYED ]] || die "--kit-from deployed: nothing is deployed yet"
	KIT_REF=$DEPLOYED
	;;
*) KIT_REF=$KIT_FROM ;;
esac
KIT_SHA=$(as_user git -C "$APP" rev-parse --verify --quiet "$KIT_REF^{commit}") || die "--kit-from: unknown commit '$KIT_FROM'"
[[ -n $KIT_LABEL ]] || KIT_LABEL=$KIT_FROM
STAGE=$(mktemp -d "$STL_HOME/kit.new.XXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
chmod 755 "$STAGE"
mkdir -p "$STAGE/systemd"
chmod 755 "$STAGE/systemd"
for f in $KIT_FILES; do
	as_user git -C "$APP" show "$KIT_SHA:$LAB/$f" >"$STAGE/$f" 2>/dev/null ||
		die "commit $(as_user git -C "$APP" rev-parse --short=12 "$KIT_SHA") has no $LAB/$f (older than this kit version); use --kit-from head"
	chmod 644 "$STAGE/$f"
done
installed_of() { case $1 in update.sh | install.sh) printf '%s' "$STL_HOME/bin/$1" ;; *) printf '%s' "$STL_HOME/kit/$1" ;; esac }
KIT_CHANGED=0 FIRST_KIT=1 KIT_SKIP=0
changed=()
for f in $KIT_FILES; do
	inst=$(installed_of "$f")
	[[ ! -e $inst ]] || FIRST_KIT=0
	cmp -s "$STAGE/$f" "$inst" 2>/dev/null || changed+=("$f")
done
((${#changed[@]} == 0)) || KIT_CHANGED=1
kit_short=$(as_user git -C "$APP" rev-parse --short=12 "$KIT_SHA")
if [[ -n $DEPLOYED ]] && [[ $DEPLOYED != "$KIT_SHA" ]]; then
	say "  (live: ${DEPLOYED:0:12}; kit source: $kit_short = $KIT_LABEL)"
fi
if ((FIRST_KIT)); then
	ok "deploy kit from $kit_short ($KIT_LABEL)"
elif ((KIT_CHANGED)); then
	say ""
	say "  ${B}배포 키트 변경 / the deploy kit changed${N} ($(head -n 1 "$STL_HOME/kit/SOURCE" 2>/dev/null | cut -c1-12) → $kit_short, $KIT_LABEL): ${changed[*]}"
	if ((!REEXEC)); then
		say "  이 파일들은 서버 전체를 제어할 수 있습니다. 누가 왜 바꿨는지 확인하세요."
		say "  These files control the whole server (docker group = root). Check who changed them and why."
		for f in "${changed[@]}"; do diff -u --label "installed/$f" --label "$kit_short/$f" "$(installed_of "$f")" "$STAGE/$f" 2>/dev/null || true; done |
			head -n 400 | sed -e 's/^/    /'
		say "  (전체 / full diff: ${SUDO}${SUDO:+-u $SVC_USER }git -C $APP diff $(head -n 1 "$STL_HOME/kit/SOURCE" 2>/dev/null | cut -c1-12) $kit_short -- $LAB)"
		if ! confirm "이 변경을 설치할까요? / install these changes?"; then
			say "  → 설치된 키트를 그대로 씁니다 / keeping the installed kit (the rest of the setup continues)"
			KIT_SKIP=1 KIT_CHANGED=0
		fi
	fi
else
	ok "deploy kit unchanged (installed from $kit_short)"
fi
if ((!KIT_SKIP)); then
	bash -n "$STAGE/update.sh" || die "$LAB/update.sh of $kit_short has a syntax error; nothing was changed"
	bash -n "$STAGE/install.sh" || die "$LAB/install.sh of $kit_short has a syntax error; nothing was changed"
fi
# a different install.sh: continue with that one (after the review above), so its own steps apply
if ((!REEXEC)) && ((!KIT_SKIP)) && [[ $(sha256sum <"$STAGE/install.sh" | cut -c1-64) != "$SELF_SUM" ]]; then
	say "  → install.sh itself changed: continuing with the $kit_short version"
	next=$(mktemp "$STL_HOME/.install.XXXXXX")
	cp "$STAGE/install.sh" "$next"
	rm -rf -- "$STAGE"
	trap - EXIT
	exec env STL_INSTALL_REEXEC="$next" STL_INSTALL_KIT_LABEL="$KIT_LABEL" bash "$next" "${ORIG_ARGS[@]}" --home "$STL_HOME" --repo "$REPO_URL" \
		--branch "$BRANCH" --kit-from "$KIT_SHA"
fi

# ------------------------------------------------------------------------------------------------ 5. stl.env
step "5/7 설정 파일 / settings ($ENV_FILE)"
if [[ ! -f $ENV_FILE ]]; then
	(umask 077 && as_user git -C "$APP" show "$KIT_SHA:$LAB/stl.env.example" >"$ENV_FILE")
	chown "$SVC_USER:$SVC_GROUP" "$ENV_FILE"
	ok "created $ENV_FILE from stl.env.example"
fi
chmod 600 "$ENV_FILE"
chown "$SVC_USER:$SVC_GROUP" "$ENV_FILE"

# settings: command line > stored value > question/default
set_opt() { # set_opt KEY PROMPT DEFAULT — only asks when the key is empty and was not given
	local key=$1 cur
	if [[ -n ${OPT[$key]+x} ]]; then
		env_set "$key" "${OPT[$key]}"
		return 0
	fi
	cur=$(env_get "$key" "")
	if [[ -z $cur ]] && [[ -n $3 ]]; then env_set "$key" "$(ask "$2" "$3")"; fi
}
env_set STL_REPO_URL "$REPO_URL"
env_set DEPLOY_BRANCH "$BRANCH"
default_site=$(hostname -I 2>/dev/null | awk '{ print $1 }')
[[ -n $default_site ]] || default_site=$(hostname -f 2>/dev/null || hostname)
cur_site=$(env_get STL_SITE_ADDRESS "")
if [[ -z ${OPT[STL_SITE_ADDRESS]+x} ]] && { [[ -z $cur_site ]] || [[ $cur_site == localhost ]]; } && ((INTERACTIVE)); then
	say "  사이트 주소: 실제 도메인(예: stl.example.ac.kr), 교내 호스트 이름, 또는 이 서버의 IP"
	say "  Site address: a real DNS name, a campus hostname, or this server's IP (comma-separate several)"
	OPT[STL_SITE_ADDRESS]=$(ask "site address" "$default_site")
fi
set_opt STL_SITE_ADDRESS "site address" "$default_site"
site=$(env_get STL_SITE_ADDRESS localhost)
site_re='^[][A-Za-z0-9.,:* -]+$'
[[ $site =~ $site_re ]] || die "invalid site address: $site (names, IP addresses, commas and spaces only)"
if [[ -z ${OPT[STL_TLS_MODE]+x} ]] && ((INTERACTIVE)); then
	say "  HTTPS: acme = Let's Encrypt(실제 도메인 + 인터넷에서 80/443 접근 가능), internal = 자체 인증서(교내 전용·IP)"
	say "  HTTPS: acme = Let's Encrypt (real domain, ports 80/443 reachable from the internet); internal = own CA"
	OPT[STL_TLS_MODE]=$(ask "TLS mode (acme/internal)" "$(env_get STL_TLS_MODE internal)")
fi
set_opt STL_TLS_MODE "" ""
[[ $(env_get STL_TLS_MODE internal) =~ ^(acme|internal)$ ]] || die "--tls must be acme or internal"
for k in STL_HTTP_PORT STL_HTTPS_PORT STL_DEBUG_PORT; do
	set_opt "$k" "" ""
	v=$(env_get "$k" 1)
	{ [[ $v =~ ^[0-9]+$ ]] && ((v >= 1 && v <= 65535)); } || die "$k must be a port number"
done
for k in STL_BIND_ADDR STL_PROJECT STL_IMAGE STL_NET_SUBNET STL_COMPOSE_OVERRIDE STL_APP_UID; do set_opt "$k" "" ""; done

workers=$((cpus - 1))
((workers <= 8)) || workers=8
((workers <= (mem_mb - 1024) / 300)) || workers=$(((mem_mb - 1024) / 300))
((workers >= 1)) || workers=1
set_opt STL_WORKERS "compute processes (STL_WORKERS)" "$workers"
w=$(env_get STL_WORKERS "$workers")
{ [[ $w =~ ^[0-9]+$ ]] && ((w >= 1 && w <= 64)); } || die "STL_WORKERS must be a number from 1 to 64"
[[ -n $(env_get STL_MEM_LIMIT "") ]] || env_set STL_MEM_LIMIT "$((w * 300 + 1024))m"

if [[ -z $(env_get STL_ACCESS_PASSWORD "") ]] || ((RESET_PASSWORD)); then
	read_password
	env_set STL_ACCESS_PASSWORD "$PASSWORD"
	PASSWORD=""
	ok "password stored in $ENV_FILE (mode 600)"
else
	ok "password already set (change it: ${SUDO}bash $STL_HOME/bin/install.sh --reset-password)"
fi
if [[ -z $(env_get STL_SESSION_SECRET "") ]]; then
	env_set STL_SESSION_SECRET "$(new_secret)"
	ok "generated STL_SESSION_SECRET"
fi
ok "settings: $ENV_FILE"

# ------------------------------------------------------------------------------------------------ 6. deploy kit + systemd
step "6/7 배포 키트·자동 업데이트 / deploy kit and auto-update"
# check the staged kit before it replaces the installed one: compose configuration within the safety rules (run by
# the service user, as the timer would), and the Caddyfile accepted by Caddy (throwaway container, no network)
kit_ok=1
if ((KIT_SKIP)); then
	: # the owner declined the kit change above: nothing to check or install
elif ! as_user env STL_HOME="$STL_HOME" STL_KIT_DIR="$STAGE" bash "$STAGE/update.sh" --check-config; then
	kit_ok=0
	warn "the compose configuration of kit $kit_short is invalid or breaks the safety rules (see above)"
fi
caddy_img=$(env_get STL_CADDY_IMAGE caddy:2-alpine)
if ((kit_ok)) && ((!KIT_SKIP)) && ! out=$(docker run --rm --network none -v "$STAGE/Caddyfile:/etc/caddy/Caddyfile:ro" \
	-e STL_SITE_ADDRESS="$(env_get STL_SITE_ADDRESS localhost)" -e STL_TLS_MODE="$(env_get STL_TLS_MODE internal)" \
	-e STL_NET_SUBNET="$(env_get STL_NET_SUBNET 172.30.83.0/24)" -e STL_MAX_BODY_KB="$(env_get STL_MAX_BODY_KB 256)" \
	"$caddy_img" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1); then
	kit_ok=0
	warn "Caddy rejects the Caddyfile of kit $kit_short:"
	printf '%s\n' "$out" | grep -v '^{"level":"info"' | tail -n 6 | sed -e 's/^/      /' >&2
elif ((kit_ok)) && ((!KIT_SKIP)); then
	ok "Caddy accepts the Caddyfile ($caddy_img, caddy validate)"
fi
KIT_REFUSED=0
if ((KIT_SKIP)); then
	ok "deploy kit: $STL_HOME/kit (unchanged, from $(head -n 1 "$STL_HOME/kit/SOURCE" 2>/dev/null | cut -c1-12))"
elif ((!kit_ok)); then
	((!FIRST_KIT)) || die "the deploy kit of $kit_short is unusable; fix it on the branch and re-run"
	warn "keeping the installed kit ($(head -n 1 "$STL_HOME/kit/SOURCE" 2>/dev/null | cut -c1-12)); the site is unchanged"
	KIT_CHANGED=0 KIT_REFUSED=1
else
	put_file 755 "$STAGE/update.sh" "$STL_HOME/bin/update.sh"
	put_file 755 "$STAGE/install.sh" "$STL_HOME/bin/install.sh"
	own_dir 755 root:root "$STL_HOME/kit/systemd"
	for f in $KIT_FILES; do
		case $f in update.sh | install.sh) ;; *) put_file 644 "$STAGE/$f" "$STL_HOME/kit/$f" ;; esac
	done
	printf '%s %s %s\n' "$KIT_SHA" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$KIT_LABEL" >"$STAGE/SOURCE"
	put_file 644 "$STAGE/SOURCE" "$STL_HOME/kit/SOURCE"
	owned=""
	[[ $EUID -ne 0 ]] || owned=" (root-owned)"
	if ((KIT_CHANGED)); then ok "deploy kit $kit_short installed: $STL_HOME/bin, $STL_HOME/kit$owned"; else ok "deploy kit: $STL_HOME/kit$owned"; fi
fi
rm -rf -- "$STAGE"
trap - EXIT

STL_CMD="${SUDO}stl-lab"
if [[ $EUID -eq 0 ]] && [[ -d /usr/local/bin ]]; then
	ln -sfn "$STL_HOME/bin/update.sh" /usr/local/bin/stl-lab
	# sudo only finds commands in its secure_path; Rocky/RHEL leave /usr/local/bin out of it
	sp=$(grep -hsE '^[[:space:]]*Defaults[[:space:]]+secure_path' /etc/sudoers /etc/sudoers.d/* 2>/dev/null | tail -n 1 || true)
	if [[ -n $sp ]] && [[ $sp != */usr/local/bin* ]]; then
		ln -sfn "$STL_HOME/bin/update.sh" /usr/bin/stl-lab
		ok "command: stl-lab (/usr/local/bin and /usr/bin: sudo's secure_path has no /usr/local/bin)"
	else
		ok "command: stl-lab  (sudo stl-lab --status, --force, --rollback, --restart, --help)"
	fi
else
	STL_CMD="STL_HOME=$STL_HOME $STL_HOME/bin/update.sh"
fi
USE_SYSTEMD=0
if ((!NO_SYSTEMD)) && [[ -d /run/systemd/system ]] && command -v systemctl >/dev/null 2>&1; then
	USE_SYSTEMD=1
	for u in $UNITS; do
		sed -e "s#/opt/stl-sim#$STL_HOME#g" -e "s#^User=stl\$#User=$SVC_USER#" "$STL_HOME/kit/systemd/$u" >"/etc/systemd/system/$u"
		chmod 644 "/etc/systemd/system/$u"
	done
	systemctl daemon-reload
	# the timers go on before the first deploy: if this session drops during the long first build, the timer
	# finishes the job (the updater's lock keeps runs from overlapping)
	systemctl enable --quiet stl-compose.service
	systemctl enable --quiet --now stl-update.timer stl-refresh.timer
	ok "systemd: stl-update.timer (every 5 min), stl-refresh.timer (weekly base-image refresh), stl-compose.service (boot)"
else
	LOG=$STL_HOME/logs/update.log
	if [[ $EUID -eq 0 ]]; then cron_cmd="sudo crontab -u $SVC_USER -e"; else cron_cmd="crontab -e"; fi
	warn "no systemd units installed: add these cron jobs for $SVC_USER ($cron_cmd):"
	say "      */5 * * * *  STL_HOME=$STL_HOME $STL_HOME/bin/update.sh >> $LOG 2>&1"
	say "      17 4 * * 0   STL_HOME=$STL_HOME $STL_HOME/bin/update.sh --refresh >> $LOG 2>&1"
	say "      @reboot      STL_HOME=$STL_HOME $STL_HOME/bin/update.sh --up >> $LOG 2>&1"
fi

# ------------------------------------------------------------------------------------------------ 7. deploy
step "7/7 배포 / deploy"
rc=0
if ((NO_DEPLOY)); then
	say "  skipped (--no-deploy). Deploy later with: $STL_CMD --force"
elif [[ -n $DEPLOYED ]]; then
	# re-run: apply the kit and stl.env to the running site, then a normal run (respects a rollback and --pause)
	args=(--after-install)
	((!KIT_CHANGED)) || args+=(kit-changed)
	as_user env STL_HOME="$STL_HOME" "$STL_HOME/bin/update.sh" "${args[@]}" || rc=$?
elif ((USE_SYSTEMD)); then
	say "  첫 빌드는 5–15분 걸립니다(npm + pip + numba 컴파일). 접속이 끊겨도 서버에서 계속 진행됩니다."
	say "  The first build takes 5–15 min. It runs as a systemd service: a dropped SSH session does not stop it."
	journalctl -u stl-update.service -f -n 0 -o cat &
	jpid=$!
	systemctl start stl-update.service || true
	kill "$jpid" 2>/dev/null || true
	rc=$(systemctl show -p ExecMainStatus --value stl-update.service 2>/dev/null || echo 1)
	[[ $rc =~ ^[0-9]+$ ]] || rc=1
else
	say "  첫 빌드는 5–15분 걸립니다(npm + pip + numba 컴파일). / The first build takes 5–15 min."
	as_user env STL_HOME="$STL_HOME" "$STL_HOME/bin/update.sh" --force || rc=$?
fi
if ((USE_SYSTEMD)) && ((rc == 0)) && ((!NO_DEPLOY)); then systemctl start stl-compose.service || true; fi

site=$(env_get STL_SITE_ADDRESS localhost)
port=$(env_get STL_HTTPS_PORT 443)
url=https://${site%%,*}
[[ $port == 443 ]] || url=$url:$port
say ""
if ((KIT_REFUSED)); then
	say "${Y}${B}새 배포 키트($kit_short)는 설치하지 않았습니다(위 오류) — 이전 키트로 계속 동작 / the kit of $kit_short was REFUSED (see above); the installed kit stays in use${N}"
fi
if ((rc == 0)) && ((!NO_DEPLOY)); then
	say "${G}${B}사이트 동작 중 / site is up${N}  →  $url/"
elif ((rc == 75)); then
	say "${Y}${B}다른 배포가 진행 중입니다 / another deploy is running${N} — $STL_CMD --status"
elif ((!NO_DEPLOY)); then
	say "${Y}${B}설치는 끝났지만 배포가 되지 않았습니다 / installed, but not deployed (code $rc)${N}"
	say "  원인은 위 로그와 docs/DEPLOY_LAB.md '문제 해결'을 보세요 / see the log above and docs/DEPLOY_LAB.md (troubleshooting)"
fi
if ((USE_SYSTEMD)); then
	logcmd="journalctl -u stl-update -f"
else
	logcmd="tail -f $STL_HOME/logs/update.log   (cron)"
fi
if [[ -n $SUDO ]]; then editcmd="sudoedit $ENV_FILE"; else editcmd="\${EDITOR:-nano} $ENV_FILE"; fi
cat <<EOF

  상태 / status            $STL_CMD --status
  배포 로그 / deploy log   $logcmd        (build logs: $STL_HOME/logs/)
  지금 배포 / deploy now   $STL_CMD --force
  되돌리기 / roll back     $STL_CMD --rollback [commit]
  설정 적용 / apply stl.env changes   $editcmd  →  $STL_CMD --restart
  키트 변경 적용 / apply kit changes  ${SUDO}bash $STL_HOME/bin/install.sh   (shows the diff first)
  공유 전 확인 / check before sharing the link:
      curl -sk -o /dev/null -w '%{http_code}\n' $url/api/meta     → 401
      curl -sk $url/api/health                                    → "access_gate":"on"
EOF
if [[ $(env_get STL_TLS_MODE internal) == internal ]]; then
	cat <<EOF
  TLS internal: 브라우저가 경고합니다. Caddy 루트 인증서를 PC에 설치하기 전에 docs/DEPLOY_LAB.md 4장 (B)의 위험 설명을
  먼저 읽으세요 / browsers warn; read the risks in docs/DEPLOY_LAB.md §4B before installing Caddy's root certificate.
EOF
fi
# exit code: the deploy's, or 1 when the new kit was refused
((rc != 0)) || ((!KIT_REFUSED)) || rc=1
exit "$rc"
