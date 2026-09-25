#!/usr/bin/env bash
# STL simulator — lab-server auto-updater (docs/DEPLOY_LAB.md).
#
# install.sh installs this file root-owned as $STL_HOME/bin/update.sh (command `stl-lab`), and stl-update.timer runs
# it every 5 minutes as the service user. One run:
#   1. git fetch DEPLOY_BRANCH with the read-only deploy key; no new commit → exit 0
#   2. check the commit out into a clean build worktree ($STL_HOME/build)
#   3. docker compose build → image $STL_IMAGE:<commit12>        (the live site keeps serving meanwhile)
#   4. canary: run the new image alone (no network, no capabilities) until its health check passes, then log in and
#      run a real compute job through the API                     (skipped when the image did not change)
#   5. switch: tag the image $STL_IMAGE:live, docker compose up -d, wait for the app's health check and the proxy
#   6. any failure: the previous image is put back (rollback); the failed commit is remembered, retried once after
#      30 min (network hiccups), then skipped until a new commit arrives or --force
#   7. record the deployed commit, prune old images (keep STL_KEEP_IMAGES)
#
# Only the APPLICATION comes from the pushed commit. The deploy kit — docker-compose.yml, Caddyfile,
# tailscale-funnel.json, this script, install.sh and the systemd units — is pinned in root-owned $STL_HOME/kit and
# $STL_HOME/bin and changes only when the owner reviews it and re-runs install.sh: whoever controls the compose file
# controls the host (the service user is in the docker group). A commit that changes the kit is still deployed (its
# image), and the log says that the kit changed. The merged compose configuration is also checked before every
# build and start (no privileged containers, host namespaces, host bind mounts, extra capabilities …).
#
# Usage: stl-lab [--force] | --status | --rollback [COMMIT] | --refresh | --restart | --up | --stop | --pause [NOTE]
#                | --resume | --compose ARGS… | --help
#   (no option)  timer run: deploy the branch head if it is new and not skipped
#   --force      deploy the branch head now, even if it failed before, was rolled back, or updates are paused
#   --refresh    rebuild the deployed commit with fresh base images (docker build --pull) and pull the proxy image;
#                switches only if something changed (stl-refresh.timer runs it weekly)
#   --rollback   go back to the previous deployment (or COMMIT); the rolled-back commit is skipped until a new one
#   --restart    recreate all containers (apply stl.env changes)        --up / --stop   start / stop the stack
# Exit codes: 0 ok / nothing to do · 1 usage or configuration error · 2 git fetch failed · 3 build failed (previous
#   version still serving) · 4 new version unhealthy, not switched or rolled back (previous version serving) ·
#   5 site unhealthy / rollback failed: needs a human · 75 another run holds the lock (systemd counts it as success)
#
# Never prints secrets: stl.env is read only for non-secret settings (and to check that the password is set), docker
# compose runs with a clean environment so only stl.env feeds it, and error output is scrubbed of the secret values.

set -Eeuo pipefail
umask 027

readonly E_OK=0 E_CONFIG=1 E_FETCH=2 E_BUILD=3 E_HEALTH=4 E_DOWN=5 E_BUSY=75
readonly LAB=deploy/lab
# the deploy kit: files of $LAB that install.sh pins on the server (kit/ or bin/)
readonly KIT_FILES="docker-compose.yml Caddyfile tailscale-funnel.json update.sh install.sh systemd/stl-update.service systemd/stl-update.timer systemd/stl-compose.service systemd/stl-refresh.service systemd/stl-refresh.timer"
SELF=$(readlink -f "${BASH_SOURCE[0]}")
unset STL_IMAGE_TAG STL_BUILD_CONTEXT # set explicitly per docker compose call below

# ------------------------------------------------------------------------------------------------ logging
# Log lines go to stderr (journald under systemd); stdout carries only data (--status, command substitutions).
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
info() { printf '%s INFO  %s\n' "$(ts)" "$*" >&2; }
warn() { printf '%s WARN  %s\n' "$(ts)" "$*" >&2; }
err() { printf '%s ERROR %s\n' "$(ts)" "$*" >&2; }
die() {
	local code=$1
	shift
	err "$*"
	exit "$code"
}
indent() { sed -e 's/^/    | /' >&2; }
trap 'err "unexpected error (line $LINENO, exit $?)"' ERR

usage() {
	sed -n '2,/^$/p' "$SELF" | sed -e 's/^# \{0,1\}//'
}

# ------------------------------------------------------------------------------------------------ paths
STL_HOME=${STL_HOME:-$(dirname "$(dirname "$SELF")")}
STL_HOME=$(readlink -f "$STL_HOME")
# a neutral working directory: the caller's may be unreadable for the service user (e.g. a mode-750 home), which
# makes git fail with misleading errors
cd "$STL_HOME" 2>/dev/null || cd /
APP=$STL_HOME/app     # git repository + checkout of the deployed commit (informational; nothing in it is run)
BUILD=$STL_HOME/build # build worktree: the candidate commit, cleaned before every build
KIT=${STL_KIT_DIR:-$STL_HOME/kit} # pinned compose file + Caddyfile (STL_KIT_DIR: install.sh checks a staged kit)
STATE=$STL_HOME/state
LOGS=$STL_HOME/logs
ENV_FILE=$STL_HOME/stl.env

# Run as the service user (owner of state/) when started as root, e.g. `sudo stl-lab --status`.
if [[ $EUID -eq 0 ]] && [[ -d $STATE ]]; then
	owner=$(stat -c %U "$STATE")
	if [[ $owner != root ]] && [[ $owner != UNKNOWN ]]; then
		exec runuser -u "$owner" -- env STL_HOME="$STL_HOME" ${STL_KIT_DIR:+STL_KIT_DIR="$STL_KIT_DIR"} \
			HOME="$(getent passwd "$owner" | cut -d: -f6)" "$SELF" "$@"
	fi
fi

# ------------------------------------------------------------------------------------------------ settings
# env_get KEY [DEFAULT]: a setting from stl.env (last assignment wins; quotes and trailing " # comment" removed).
env_get() {
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
env_nonempty() { [[ -n $(env_get "$1") ]]; }

# scrub KEY… : replace the values of these stl.env secrets in stdin with <redacted> (pure bash: a secret never
# becomes a process argument)
scrub() {
	local text k v
	text=$(cat)
	for k in STL_ACCESS_PASSWORD STL_SESSION_SECRET CLOUDFLARE_TUNNEL_TOKEN TS_AUTHKEY; do
		v=$(env_get "$k")
		((${#v} >= 4)) || continue
		text=${text//"$v"/<redacted>}
	done
	printf '%s\n' "$text"
}

load_settings() {
	if [[ ! -f $ENV_FILE ]]; then
		[[ ! -f $APP/$LAB/stl.env ]] ||
			die $E_CONFIG "stl.env is still at its old place ($APP/$LAB/stl.env): re-run install.sh, which moves it to $ENV_FILE"
		die $E_CONFIG "missing $ENV_FILE — run install.sh (docs/DEPLOY_LAB.md §3)"
	fi
	BRANCH=$(env_get DEPLOY_BRANCH main)
	PROJECT=$(env_get STL_PROJECT stl)
	IMAGE=$(env_get STL_IMAGE stl-lab/app)
	PROFILES=$(env_get COMPOSE_PROFILES "")
	OVERRIDE=$(env_get STL_COMPOSE_OVERRIDE "")
	HEALTH_TIMEOUT=$(env_get STL_HEALTH_TIMEOUT 300)
	KEEP_IMAGES=$(env_get STL_KEEP_IMAGES 3)
	SMOKE=$(env_get STL_SMOKE 1)
	APP_UID=$(env_get STL_APP_UID 61000)
	MEM_LIMIT=$(env_get STL_MEM_LIMIT 4g)
	[[ $HEALTH_TIMEOUT =~ ^[0-9]+$ ]] || HEALTH_TIMEOUT=300
	[[ $KEEP_IMAGES =~ ^[0-9]+$ ]] && ((KEEP_IMAGES >= 1)) || KEEP_IMAGES=3
	[[ $MEM_LIMIT =~ ^[0-9]+[bkmgBKMG]?$ ]] || MEM_LIMIT=4g
}

preflight() {
	local c mode out ovr
	for c in docker git flock timeout; do
		command -v "$c" >/dev/null 2>&1 || die $E_CONFIG "required command not found: $c"
	done
	docker info >/dev/null 2>&1 || die $E_CONFIG "cannot talk to the Docker daemon (is docker running? is $(id -un) in the docker group?)"
	docker compose version >/dev/null 2>&1 || die $E_CONFIG "the docker compose plugin is missing (docs/DEPLOY_LAB.md §1)"
	[[ -d $APP/.git ]] || die $E_CONFIG "no checkout at $APP — run install.sh first"
	for c in docker-compose.yml Caddyfile; do
		[[ -r $KIT/$c ]] || die $E_CONFIG "the installed deploy kit is incomplete ($KIT/$c missing) — re-run install.sh"
	done
	mode=$(stat -c %a "$ENV_FILE")
	[[ $mode =~ ^[0-7]00$ ]] || warn "$ENV_FILE is readable by other users (mode $mode): chmod 600 $ENV_FILE"
	env_nonempty STL_ACCESS_PASSWORD || die $E_CONFIG "STL_ACCESS_PASSWORD is empty in stl.env — the site would refuse every request (install.sh --reset-password)"
	env_nonempty STL_SESSION_SECRET || die $E_CONFIG "STL_SESSION_SECRET is empty in stl.env (generate one: openssl rand -hex 32)"
	case $(env_get STL_TLS_MODE internal) in acme | internal) ;; *) die $E_CONFIG "STL_TLS_MODE must be acme or internal" ;; esac
	# these are pasted into the Caddyfile as text: allow only what they need
	local site_re='^[][A-Za-z0-9.,:* -]+$'
	[[ $(env_get STL_SITE_ADDRESS localhost) =~ $site_re ]] || die $E_CONFIG "STL_SITE_ADDRESS may contain only names, IP addresses, commas and spaces"
	[[ $(env_get STL_NET_SUBNET 172.30.83.0/24) =~ ^[0-9.]+/[0-9]+$ ]] || die $E_CONFIG "STL_NET_SUBNET must look like 172.30.83.0/24"
	{ [[ $(env_get STL_MAX_BODY_KB 256) =~ ^[0-9]+$ ]] && (($(env_get STL_MAX_BODY_KB 256) >= 1)); } || die $E_CONFIG "STL_MAX_BODY_KB must be a whole number of KiB (default 256)"
	{ [[ $APP_UID =~ ^[0-9]+$ ]] && ((APP_UID >= 1000 && APP_UID < 2147483647)); } || die $E_CONFIG "STL_APP_UID must be a number ≥ 1000 that no account on this server uses"
	if ! out=$(git check-ref-format --branch "$BRANCH" 2>&1); then
		die $E_CONFIG "DEPLOY_BRANCH '$BRANCH' is not a valid branch name ($out)"
	fi
	if [[ -n $OVERRIDE ]]; then
		[[ $OVERRIDE == /* ]] || die $E_CONFIG "STL_COMPOSE_OVERRIDE must be an absolute path"
		[[ -f $OVERRIDE ]] || die $E_CONFIG "STL_COMPOSE_OVERRIDE file not found: $OVERRIDE"
		ovr=$(readlink -f "$OVERRIDE")
		case $ovr in "$APP"/* | "$BUILD"/*) die $E_CONFIG "STL_COMPOSE_OVERRIDE must not be inside the git checkouts ($APP, $BUILD): a push could change it" ;; esac
	fi
	if [[ ,$PROFILES, == *,cloudflare,* ]] && ! env_nonempty CLOUDFLARE_TUNNEL_TOKEN; then
		warn "COMPOSE_PROFILES has cloudflare but CLOUDFLARE_TUNNEL_TOKEN is empty: the tunnel cannot start"
	fi
	if [[ ,$PROFILES, == *,tailscale,* ]] && ! env_nonempty TS_AUTHKEY; then
		warn "COMPOSE_PROFILES has tailscale but TS_AUTHKEY is empty (needed until the machine has joined the tailnet)"
	fi
	mkdir -p "$STATE" "$LOGS"
}

# ------------------------------------------------------------------------------------------------ helpers
state_get() { cat "$STATE/$1" 2>/dev/null || true; }
state_set() {
	printf '%s\n' "$2" >"$STATE/$1.tmp"
	mv -f "$STATE/$1.tmp" "$STATE/$1"
}
history_add() { printf '%s %s %s %s\n' "$(ts)" "$1" "$2" "${3-}" >>"$STATE/history"; }
short() { printf '%s' "${1:0:12}"; }
# commit subject for log lines: control characters removed (a commit message must not be able to forge log lines)
subject() { git -C "$APP" log -1 --format=%s "$1" 2>/dev/null | tr -d '\000-\037\177' | cut -c1-80 || true; }
redact() { sed -E 's#(https?://)[^/@]*@#\1***@#g'; }

# compose CONTEXT ARGS…: docker compose with the installed kit, the build context CONTEXT (the build worktree for a
# build, the live checkout otherwise — `up` never builds), and a clean environment so that only stl.env (and
# STL_IMAGE_TAG, when the caller sets it) feed the interpolation.
compose() {
	local ctx=$1
	shift
	local files=(-f "$KIT/docker-compose.yml")
	[[ -z $OVERRIDE ]] || files+=(-f "$OVERRIDE")
	env -i PATH="$PATH" HOME="${HOME:-$STL_HOME}" LANG=C.UTF-8 \
		${DOCKER_HOST:+DOCKER_HOST="$DOCKER_HOST"} ${DOCKER_CONFIG:+DOCKER_CONFIG="$DOCKER_CONFIG"} \
		${DOCKER_CONTEXT:+DOCKER_CONTEXT="$DOCKER_CONTEXT"} \
		COMPOSE_PROFILES="$PROFILES" BUILDKIT_PROGRESS=plain STL_BUILD_CONTEXT="$ctx" \
		${STL_IMAGE_TAG:+STL_IMAGE_TAG="$STL_IMAGE_TAG"} \
		docker compose -p "$PROJECT" --project-directory "$KIT" --env-file "$ENV_FILE" "${files[@]}" "$@"
}

# The merged compose configuration may not give any container power over the host. The kit's own file passes; this
# catches an override file or a future edit of the kit that would (defence in depth).
readonly POLICY_PY='
import json, os, sys
kit, ctx = (os.path.realpath(p) for p in sys.argv[1:3])
cfg = json.load(sys.stdin)
bad = []
known = {"app", "caddy", "debug", "cloudflared", "tailscale"}
binds = {os.path.join(kit, "Caddyfile"), os.path.join(kit, "tailscale-funnel.json")}
for name, s in (cfg.get("services") or {}).items():
    at = "service " + name
    if name not in known:
        bad.append(at + ": not a service of the kit")
    if s.get("privileged"):
        bad.append(at + ": privileged")
    for k in ("network_mode", "pid", "ipc", "userns_mode", "uts", "cgroup", "runtime", "cgroup_parent", "isolation"):
        if s.get(k):
            bad.append("%s: %s: %s" % (at, k, s[k]))
    for k in ("devices", "device_cgroup_rules", "volumes_from", "secrets", "configs", "gpus"):
        if s.get(k):
            bad.append("%s: %s" % (at, k))
    caps = {str(c).upper().replace("CAP_", "", 1) for c in s.get("cap_add") or []}
    if caps - {"NET_BIND_SERVICE"}:
        bad.append("%s: cap_add %s" % (at, ",".join(sorted(caps - {"NET_BIND_SERVICE"}))))
    for o in s.get("security_opt") or []:
        o = str(o).replace(":", "=", 1)
        if "unconfined" in o or o.startswith("label=disable") or o.startswith("no-new-privileges=false"):
            bad.append("%s: security_opt %s" % (at, o))
    for v in s.get("volumes") or []:
        t = v.get("type")
        if t == "bind":
            src = os.path.realpath(v.get("source") or "")
            if src not in binds or not v.get("read_only"):
                bad.append("%s: bind mount of %s (only the kit files, read-only)" % (at, src))
        elif t not in ("volume", "tmpfs"):
            bad.append("%s: %s mount" % (at, t))
    b = s.get("build")
    if b:
        if name != "app":
            bad.append(at + ": build")
        if os.path.realpath(b.get("context") or "") != ctx:
            bad.append("%s: build context %s (expected %s)" % (at, b.get("context"), ctx))
        if b.get("dockerfile") != "Dockerfile" or b.get("dockerfile_inline"):
            bad.append(at + ": build.dockerfile must be the repository Dockerfile")
        for k in ("network", "entitlements", "secrets", "ssh", "privileged", "extra_hosts"):
            if b.get(k):
                bad.append("%s: build.%s" % (at, k))
        if set(b.get("args") or {}) - {"APP_UID"}:
            bad.append("%s: build.args %s" % (at, ",".join(sorted(set(b["args"]) - {"APP_UID"}))))
        for k, v in (b.get("additional_contexts") or {}).items():
            if not str(v).startswith("docker-image://"):
                bad.append("%s: build.additional_contexts %s (only docker-image://)" % (at, k))
    if name == "app":
        if str(s.get("user") or "0").split(":")[0] in ("", "0", "root"):
            bad.append(at + ": runs as root (user: missing)")
        if s.get("ports"):
            bad.append(at + ": publishes ports (only the proxy may)")
        if set(s.get("networks") or {}) != {"backend"}:
            bad.append(at + ": must be on the internal network backend only")
for name, v in (cfg.get("volumes") or {}).items():
    if v.get("external") or v.get("driver_opts") or (v.get("driver") or "local") != "local":
        bad.append("volume %s: external, driver or driver_opts" % name)
for name, n in (cfg.get("networks") or {}).items():
    if n.get("external") or (n.get("driver") or "bridge") != "bridge":
        bad.append("network %s: external or driver %s" % (name, n.get("driver")))
if not (cfg.get("networks") or {}).get("backend", {}).get("internal"):
    bad.append("network backend: must be internal")
for k in ("secrets", "configs"):
    if cfg.get(k):
        bad.append("top-level " + k)
for line in bad:
    print(line)
sys.exit(1 if bad else 0)
'

# policy_check CONTEXT: the configuration `compose CONTEXT …` would run is within the kit's rules
policy_check() {
	local ctx=$1 json out
	if ! json=$(compose "$ctx" config --format json 2>&1); then
		err "docker compose cannot read the configuration ($KIT/docker-compose.yml${OVERRIDE:+ + $OVERRIDE}, $ENV_FILE):"
		printf '%s\n' "$json" | scrub | tail -n 15 | indent
		return 1
	fi
	if ! command -v python3 >/dev/null 2>&1; then
		warn "python3 not found: the compose policy check is skipped"
		return 0
	fi
	if ! out=$(printf '%s' "$json" | python3 -c "$POLICY_PY" "$KIT" "$ctx" 2>&1); then
		err "REFUSED: the compose configuration breaks the kit's safety rules:"
		printf '%s\n' "$out" | scrub | indent
		return 1
	fi
}

image_id() { docker image inspect -f '{{.Id}}' "$1" 2>/dev/null || true; }
# content fingerprint (layers + config): two builds of the same code differ in .Id (build attestations), not in this
image_fp() {
	local out
	out=$(docker image inspect -f '{{json .RootFS.Layers}}{{json .Config}}' "$1" 2>/dev/null) || return 0
	printf '%s' "$out" | sha256sum | cut -c1-64
}

LOCKED=0
lock() { # lock wait|nowait
	((LOCKED)) && return 0
	exec 9>>"$STATE/update.lock"
	if ! flock -n 9; then
		if [[ $1 == nowait ]]; then
			info "another update run is in progress; skipping this one"
			exit $E_BUSY
		fi
		info "waiting for the running update to finish …"
		flock -w 3600 9 || die $E_BUSY "gave up waiting for the lock after 1 h"
	fi
	LOCKED=1
}

CANARY=""
stop_canary() {
	if [[ -n $CANARY ]]; then docker rm -f "$CANARY" >/dev/null 2>&1 || true; fi
	CANARY=""
}
trap stop_canary EXIT

readonly HEALTH_PY="import json,urllib.request as u; d=json.load(u.urlopen('http://127.0.0.1:8000/api/health', timeout=4)); raise SystemExit(0 if d.get('ok') and d.get('access_gate') == 'on' else 1)"

# wait_container NAME_OR_ID LIMIT_S LABEL: until the container's health check passes (0) or it fails/exits (1)
wait_container() {
	local cid=$1 limit=$2 label=$3 t0=$SECONDS st restarts0 restarts
	restarts0=$(docker inspect -f '{{.RestartCount}}' "$cid" 2>/dev/null || echo 0)
	while :; do
		st=$(docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo "gone -")
		restarts=$(docker inspect -f '{{.RestartCount}}' "$cid" 2>/dev/null || echo 0)
		case $st in
		"running healthy")
			info "$label healthy after $((SECONDS - t0)) s"
			return 0
			;;
		"running unhealthy" | exited* | dead* | gone* | restarting*)
			err "$label is ${st% *} (health: ${st#* })"
			show_container "$cid"
			return 1
			;;
		esac
		if ((restarts > restarts0)); then
			err "$label crashed and was restarted"
			show_container "$cid"
			return 1
		fi
		if ((SECONDS - t0 >= limit)); then
			err "$label not healthy after ${limit} s (state: $st)"
			show_container "$cid"
			return 1
		fi
		sleep 2
	done
}

show_container() {
	local out
	out=$(docker inspect -f '{{if .State.Health}}{{range .State.Health.Log}}{{.Output}}{{end}}{{end}}' "$1" 2>/dev/null | tail -n 5 || true)
	[[ -z $out ]] || {
		err "last health-check output:"
		printf '%s\n' "$out" | scrub | indent
	}
	err "last log lines:"
	docker logs --tail 25 "$1" 2>&1 | scrub | indent || true
}

# ------------------------------------------------------------------------------------------------ git
sync_remote_url() {
	local url cur
	url=$(env_get STL_REPO_URL "")
	[[ -n $url ]] || return 0
	cur=$(git -C "$APP" remote get-url origin 2>/dev/null || true)
	if [[ $cur != "$url" ]]; then
		git -C "$APP" remote set-url origin "$url"
		info "origin set to $(printf '%s' "$url" | redact)"
	fi
}

fetch_head() { # prints the commit at the tip of DEPLOY_BRANCH
	local out
	if ! out=$(GIT_TERMINAL_PROMPT=0 timeout 180 git -C "$APP" fetch --quiet --no-tags --prune origin \
		"+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>&1); then
		err "git fetch of '$BRANCH' from $(git -C "$APP" remote get-url origin | redact) failed:"
		printf '%s\n' "$out" | redact | indent
		err "check the network (outbound SSH to github.com — docs/DEPLOY_LAB.md §7), that the branch exists, and the deploy key (GitHub → repository → Settings → Deploy keys)"
		return 1
	fi
	state_set last_fetch "$(date +%s)"
	git -C "$APP" rev-parse --verify --quiet "refs/remotes/origin/$BRANCH^{commit}"
}

check_commit() { # refuse commits the kit cannot or must not deploy
	local sha=$1 f
	git -C "$APP" cat-file -e "$sha:$LAB/docker-compose.yml" 2>/dev/null ||
		{
			err "commit $(short "$sha") has no $LAB/docker-compose.yml (older than the lab-server kit); not deploying it"
			return 1
		}
	for f in "$LAB/stl.env" "$LAB/.env"; do
		if git -C "$APP" cat-file -e "$sha:$f" 2>/dev/null; then
			err "commit $(short "$sha") contains $f — secrets must never be committed; not deploying it"
			return 1
		fi
	done
}

prepare_build_tree() {
	local sha=$1
	if [[ ! -e $BUILD/.git ]]; then
		git -C "$APP" worktree prune || return 1
		mkdir -p "$BUILD" || return 1
		# empty it (the directory itself may belong to a root-owned parent and stays)
		find "$BUILD" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || return 1
		git -C "$APP" worktree add --quiet --detach "$BUILD" "$sha" || return 1
	else
		git -C "$BUILD" checkout --quiet --force --detach "$sha" || return 1
	fi
	git -C "$BUILD" clean -ffdxq
}

# kit_diff COMMIT: the kit files whose version in COMMIT differs from the installed one (space-separated)
kit_diff() {
	local c=$1 f inst out=()
	for f in $KIT_FILES; do
		case $f in update.sh | install.sh) inst=$STL_HOME/bin/$f ;; *) inst=$KIT/$f ;; esac
		if git -C "$APP" cat-file -e "$c:$LAB/$f" 2>/dev/null; then
			git -C "$APP" show "$c:$LAB/$f" 2>/dev/null | cmp -s - "$inst" 2>/dev/null || out+=("$f")
		elif [[ -e $inst ]]; then
			out+=("$f")
		fi
	done
	printf '%s' "${out[*]}"
}

kit_note() { # kit_note COMMIT: log that COMMIT carries a different deploy kit
	local c=$1 d src
	d=$(kit_diff "$c")
	[[ -n $d ]] || return 0
	src=$(state_kit_source)
	warn "commit $(short "$c") changes the deploy kit ($d). The server keeps using the installed kit${src:+ (from $(short "$src"))}:"
	warn "  review:  git -C $APP diff ${src:-<installed>} $(short "$c") -- $LAB"
	warn "  apply:   sudo bash $STL_HOME/bin/install.sh   (shows the diff and asks before installing)"
}
state_kit_source() { head -n 1 "$KIT/SOURCE" 2>/dev/null | cut -d' ' -f1 || true; }

# ------------------------------------------------------------------------------------------------ build / canary
build_image() { # build_image SHA [pull]
	local sha=$1 pull=${2-} tag log t0=$SECONDS args=(build)
	tag=$(short "$sha")
	log=$LOGS/build-$tag.log
	[[ -z $pull ]] || args+=(--pull)
	policy_check "$BUILD" || return 1
	info "building $IMAGE:$tag from $(short "$sha")${pull:+ with fresh base images} (log: $log)"
	if STL_IMAGE_TAG=$tag compose "$BUILD" "${args[@]}" app >"$log" 2>&1; then
		info "build finished in $((SECONDS - t0)) s"
		return 0
	fi
	err "build FAILED after $((SECONDS - t0)) s; last lines of $log:"
	tail -n 25 "$log" | scrub | indent
	return 1
}

canary() { # canary IMAGE: start it alone, wait for health, then log in and run one compute job
	local img=$1 pw
	CANARY=$PROJECT-canary
	docker rm -f "$CANARY" >/dev/null 2>&1 || true
	pw=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n') # throwaway password of the canary only
	info "canary: starting $img without network"
	# same restrictions as the live container (docker-compose.yml): uid, no capabilities, memory cap
	docker run -d --name "$CANARY" --network none --label "stl.canary=$PROJECT" --init \
		--user "$APP_UID:$APP_UID" --cap-drop ALL --security-opt no-new-privileges --memory "$MEM_LIMIT" \
		--env-file <(printf 'STL_ACCESS_PASSWORD=%s\nSTL_REQUIRE_PASSWORD=1\nSTL_WORKERS=1\n' "$pw") \
		--health-cmd "python -c \"$HEALTH_PY\"" --health-interval 5s --health-timeout 5s --health-retries 2 \
		--health-start-period "${HEALTH_TIMEOUT}s" --health-start-interval 2s \
		"$img" >/dev/null || return 1
	wait_container "$CANARY" "$HEALTH_TIMEOUT" "canary" || return 1
	if [[ $SMOKE != 0 ]]; then
		local out
		if ! out=$(docker exec -i "$CANARY" python - 2>&1 <<'PY'
import http.client, json, os, urllib.parse

def req(method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", 8000, timeout=90)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, r.headers, data

st, _, _ = req("GET", "/api/meta")
assert st == 401, f"/api/meta without a session answered HTTP {st} (expected 401: the login gate is not active)"
form = urllib.parse.urlencode({"password": os.environ["STL_ACCESS_PASSWORD"], "next": "/"})
st, h, _ = req("POST", "/login", form, {"Content-Type": "application/x-www-form-urlencoded"})
cookie = (h.get("Set-Cookie") or "").split(";")[0]
assert st == 303 and cookie.startswith("stl_session="), f"login answered HTTP {st} without a session cookie"
st, _, _ = req("GET", "/api/meta", headers={"Cookie": cookie})
assert st == 200, f"/api/meta with a session answered HTTP {st}"
st, _, data = req("POST", "/api/compute/folds?wait=60", json.dumps({"device": {"preset": "paper"}}),
                  {"Cookie": cookie, "Content-Type": "application/json"})
job = json.loads(data or b"{}")
v = ((job.get("result") or {}).get("folds") or {}).get("V_LU")
assert st == 200 and job.get("status") == "done" and isinstance(v, (int, float)), \
    f"compute job: HTTP {st}, status {job.get('status')!r}, error {job.get('error')!r}"
print(f"login gate, session and a compute job (folds: V_LU = {v:.4f} V) work")
PY
		); then
			err "canary smoke test FAILED:"
			printf '%s\n' "$out" | tail -n 8 | indent
			show_container "$CANARY"
			return 1
		fi
		info "canary: $(printf '%s' "$out" | tail -n 1)"
	fi
	stop_canary
}

# ------------------------------------------------------------------------------------------------ switch
check_proxy() { # caddy → app through the proxy's own handler (plain-HTTP tunnel entrance, inside the container)
	local out _
	for _ in $(seq 1 15); do
		out=$(compose "$APP" exec -T caddy wget -q -T 10 -O - http://127.0.0.1:8080/api/health 2>/dev/null || true)
		if [[ $out == *'"ok":true'* ]] && [[ $out == *'"access_gate":"on"'* ]]; then
			info "proxy → app OK (access gate on)"
			return 0
		fi
		sleep 2
	done
	err "the proxy (caddy) cannot reach a healthy, password-protected app"
	compose "$APP" logs --no-color --tail 20 caddy 2>&1 | scrub | indent || true
	return 1
}

# fingerprint of the kit files that are bind-mounted into containers: a replaced file (install.sh) is a new inode,
# which a running container does not see until it is recreated
mounted_kit_fp() { cat "$KIT/Caddyfile" "$KIT/tailscale-funnel.json" 2>/dev/null | sha256sum | cut -c1-16; }

bring_up() { # start/refresh the stack (image :live, installed kit) and wait until app and proxy are healthy
	local cid fp svcs=(caddy)
	policy_check "$APP" || return 1
	compose "$APP" up -d --no-build --remove-orphans 2>&1 | scrub | indent || {
		err "docker compose up failed"
		return 1
	}
	fp=$(mounted_kit_fp)
	if [[ $fp != "$(state_get kit_mounted)" ]]; then
		[[ ,$PROFILES, != *,tailscale,* ]] || svcs+=(tailscale)
		info "proxy configuration (kit) changed: recreating ${svcs[*]}"
		compose "$APP" up -d --no-build --no-deps --force-recreate "${svcs[@]}" 2>&1 | scrub | indent || return 1
	fi
	cid=$(compose "$APP" ps -q app 2>/dev/null | head -n 1)
	[[ -n $cid ]] || {
		err "no app container after docker compose up"
		return 1
	}
	wait_container "$cid" "$HEALTH_TIMEOUT" "app" || return 1
	check_proxy || return 1
	state_set kit_mounted "$fp"
}

# switch_to SHA: make SHA live (image $IMAGE:<sha12> must exist). 0 = live and healthy, 4 = failed and rolled back,
# 5 = failed and the rollback failed too (or there was nothing to roll back to)
switch_to() {
	local sha=$1 tag prev prev_id rc=0
	tag=$(short "$sha")
	prev=$(state_get deployed)
	prev_id=$(image_id "$IMAGE:live")
	info "switching to $tag"
	docker tag "$IMAGE:$tag" "$IMAGE:live" || {
		err "cannot tag $IMAGE:$tag; nothing was switched"
		return 4
	}
	if git -C "$APP" checkout --quiet --force --detach "$sha" && bring_up; then
		return 0
	fi
	if [[ -z $prev ]] || [[ -z $prev_id ]]; then
		err "no previous version to roll back to: the site is down until a working commit is deployed"
		return 5
	fi
	warn "rolling back to $(short "$prev")"
	docker tag "$prev_id" "$IMAGE:live" || rc=5
	git -C "$APP" checkout --quiet --force --detach "$prev" || rc=5
	((rc != 0)) || bring_up || rc=5
	if ((rc == 0)); then
		warn "rolled back: $(short "$prev") is serving again"
		return 4
	fi
	err "ROLLBACK FAILED: the site is down — see docs/DEPLOY_LAB.md (troubleshooting)"
	return 5
}

keep_ref() { git -C "$APP" update-ref "refs/stl/deployed/$(short "$1")" "$1" 2>/dev/null || true; }

record_success() {
	local sha=$1 prev
	prev=$(state_get deployed)
	if [[ -n $prev ]] && [[ $prev != "$sha" ]]; then state_set previous "$prev"; fi
	state_set deployed "$sha"
	state_set deployed_at "$(ts)"
	rm -f "$STATE/failed"
	keep_ref "$sha"
	history_add deployed "$(short "$sha")" "\"$(subject "$sha")\""
}

record_failure() { # record_failure SHA CODE
	local sha=$1 code=$2 fsha="" fcode="" fattempts=0 fwhen=0 attempts=1
	read -r fsha fcode fattempts fwhen < <(state_get failed) || true
	[[ $fsha == "$sha" ]] && [[ $fattempts =~ ^[0-9]+$ ]] && attempts=$((fattempts + 1))
	state_set failed "$sha $code $attempts $(date +%s)"
	history_add failed "$(short "$sha")" "code=$code attempt=$attempts"
}

skip_reason() { # skip_reason SHA → prints why SHA is skipped now (exit 0), or exit 1 when it should be tried
	local fsha="" fcode="" fattempts=0 fwhen=0 now
	read -r fsha fcode fattempts fwhen < <(state_get failed) || return 1
	[[ $fsha == "$1" ]] || return 1
	now=$(date +%s)
	if [[ $fcode == rollback ]]; then
		printf 'it was rolled back'
		return 0
	fi
	if ((fattempts >= 2)); then
		printf 'it failed %s times (code %s)' "$fattempts" "$fcode"
		return 0
	fi
	if ((now - fwhen < 1800)); then
		printf 'it failed %s min ago (code %s; one retry 30 min after the failure)' "$(((now - fwhen) / 60))" "$fcode"
		return 0
	fi
	return 1
}

prune() {
	local keep=() ids=() t live_id id
	# keep the tags of the most recent deployments until STL_KEEP_IMAGES distinct images are kept (several commits,
	# e.g. documentation-only ones, share one image)
	while read -r t; do
		id=$(image_id "$IMAGE:$t")
		[[ -n $id ]] || continue
		if [[ " ${ids[*]} " != *" $id "* ]]; then
			((${#ids[@]} < KEEP_IMAGES)) || break
			ids+=("$id")
		fi
		keep+=("$t")
	done < <(awk '$2 == "deployed" { print $3 }' "$STATE/history" 2>/dev/null | tac | awk '!seen[$0]++')
	live_id=$(image_id "$IMAGE:live")
	while read -r t; do
		[[ -n $t ]] && [[ $t != live ]] && [[ $t != "<none>" ]] || continue
		[[ " ${keep[*]} " == *" $t "* ]] && continue
		id=$(image_id "$IMAGE:$t")
		if docker image rm "$IMAGE:$t" >/dev/null 2>&1; then
			[[ $id == "$live_id" ]] || info "removed old image $IMAGE:$t"
		fi
	done < <(docker image ls "$IMAGE" --format '{{.Tag}}' 2>/dev/null)
	# images of this project that lost all their tags (a rebuild of the same commit, --refresh): only dangling ones,
	# and never one a container still uses
	docker image prune -f --filter "label=com.docker.compose.project=$PROJECT" >/dev/null 2>&1 || true
	# git refs that keep deployed commits reachable (rollback after a force-push): only for kept images
	while read -r ref; do
		t=${ref##*/}
		[[ " ${keep[*]} " == *" $t "* ]] || git -C "$APP" update-ref -d "$ref" 2>/dev/null || true
	done < <(git -C "$APP" for-each-ref --format='%(refname)' refs/stl/deployed/ 2>/dev/null)
	find "$LOGS" -maxdepth 1 -name 'build-*.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | tail -n +21 |
		cut -d' ' -f2- | xargs -r rm -f -- || true
}

serving() { # which version keeps serving after a failed deploy
	local d
	d=$(state_get deployed)
	if [[ -n $d ]]; then printf '%s keeps serving' "$(short "$d")"; else printf 'nothing is deployed yet'; fi
}

drop_image() { # drop_image SHA: remove the candidate image that failed (unless something still uses it)
	local sha=$1 ref
	ref=$IMAGE:$(short "$sha")
	[[ $(image_id "$ref") == "$(image_id "$IMAGE:live")" ]] && return 0
	docker image rm "$ref" >/dev/null 2>&1 || true
	# a failed rebuild of the deployed commit: its tag goes back to the image that is serving
	if [[ $sha == "$(state_get deployed)" ]] && [[ -n $(image_id "$IMAGE:live") ]]; then
		docker tag "$IMAGE:live" "$ref" || true
	fi
}

# deploy SHA [pull]: build → canary → switch; exits with the result code on failure
deploy() {
	local sha=$1 pull=${2-} tag t0=$SECONDS rc new_id new_fp live_fp
	tag=$(short "$sha")
	info "deploying $tag \"$(subject "$sha")\" from $BRANCH"
	check_commit "$sha" || {
		record_failure "$sha" $E_BUILD
		exit $E_BUILD
	}
	if ! prepare_build_tree "$sha" || ! build_image "$sha" "$pull"; then
		record_failure "$sha" $E_BUILD
		drop_image "$sha"
		err "deploy of $tag FAILED at the build; $(serving)"
		exit $E_BUILD
	fi
	new_fp=$(image_fp "$IMAGE:$tag")
	live_fp=$(image_fp "$IMAGE:live")
	if [[ -n $live_fp ]] && [[ $new_fp == "$live_fp" ]] && [[ -n $(state_get deployed) ]]; then
		info "image content unchanged: reusing the running image, no canary, no restart"
		new_id=$(image_id "$IMAGE:$tag")
		docker tag "$IMAGE:live" "$IMAGE:$tag"
		[[ $new_id == "$(image_id "$IMAGE:live")" ]] || docker image rm "$new_id" >/dev/null 2>&1 || true
		git -C "$APP" checkout --quiet --force --detach "$sha" || true
	elif ! canary "$IMAGE:$tag"; then
		stop_canary
		record_failure "$sha" $E_HEALTH
		drop_image "$sha"
		err "deploy of $tag FAILED: the new version is unhealthy and was not switched in; $(serving)"
		exit $E_HEALTH
	else
		rc=0
		switch_to "$sha" || rc=$?
		if ((rc != 0)); then
			record_failure "$sha" "$rc"
			drop_image "$sha"
			exit "$rc"
		fi
	fi
	record_success "$sha"
	prune
	info "DEPLOYED $tag in $((SECONDS - t0)) s"
	kit_note "$sha"
}

# ------------------------------------------------------------------------------------------------ commands
update_locked() { # update_locked FORCE: the timer run (0) or --force (1), with the lock held
	local force=$1 head deployed why
	if [[ -f $STATE/paused ]] && ((!force)); then
		info "automatic updates are paused ($(cat "$STATE/paused")); resume with: stl-lab --resume"
		exit $E_OK
	fi
	sync_remote_url
	head=$(fetch_head) || exit $E_FETCH
	[[ -n $head ]] || die $E_FETCH "branch '$BRANCH' not found on the remote"
	state_set remote_head "$head"
	deployed=$(state_get deployed)
	if [[ $head == "$deployed" ]] && ((!force)); then
		info "up to date: $(short "$head") on $BRANCH"
		exit $E_OK
	fi
	if ((!force)) && why=$(skip_reason "$head"); then
		local fcode
		fcode=$(state_get failed | awk '{ print $2 }')
		info "not deploying $(short "$head"): $why; $(serving) (push a fix, or: stl-lab --force)"
		[[ $fcode == rollback ]] && exit $E_OK
		exit "${fcode:-$E_BUILD}"
	fi
	deploy "$head"
}

cmd_update() {
	load_settings
	preflight
	if (($1)); then lock wait; else lock nowait; fi
	update_locked "$1"
}

cmd_after_install() { # install.sh re-run: apply the (possibly new) kit to the running site, then a normal run
	local fcode
	load_settings
	preflight
	lock wait
	if [[ -n $(state_get deployed) ]]; then
		info "applying the installed kit and stl.env to the running site"
		bring_up || die $E_DOWN "the site did not come up with the installed kit (sudo stl-lab --status)"
	fi
	if [[ ${1-} == kit-changed ]]; then
		fcode=$(state_get failed | awk '{ print $2 }')
		if [[ -n $fcode ]] && [[ $fcode != rollback ]]; then
			info "the kit changed: the failed commit gets a new try"
			rm -f "$STATE/failed"
		fi
	fi
	update_locked 0
}

cmd_refresh() { # rebuild the deployed commit on fresh base images; pull the other images
	local deployed out
	load_settings
	preflight
	lock wait
	if [[ -f $STATE/paused ]]; then
		info "automatic updates are paused ($(cat "$STATE/paused")): no refresh; resume with: stl-lab --resume"
		exit $E_OK
	fi
	deployed=$(state_get deployed)
	[[ -n $deployed ]] || die $E_CONFIG "nothing is deployed yet"
	info "refresh: pulling the base, proxy and tunnel images, then rebuilding $(short "$deployed")"
	prepare_build_tree "$deployed" || die $E_BUILD "cannot check out $(short "$deployed")"
	# the Dockerfile's base images: `docker pull` updates the local tags (a later ordinary build, which does not pull,
	# would otherwise go back to the old local copy); `build --pull` below then uses exactly these
	local stages b
	stages=" $(awk 'toupper($1) == "FROM" { for (i = 2; i < NF; i++) if (toupper($i) == "AS") print $(i + 1) }' "$BUILD/Dockerfile" | tr '\n' ' ') "
	while read -r b; do
		[[ -n $b ]] && [[ $b != *'$'* ]] && [[ ${b,,} != scratch ]] && [[ $stages != *" $b "* ]] || continue
		if out=$(docker pull -q "$b" 2>&1); then info "pulled $b"; else
			warn "docker pull $b failed (the local copy is used):"
			printf '%s\n' "$out" | tail -n 3 | indent
		fi
	done < <(awk 'toupper($1) == "FROM" { for (i = 2; i <= NF; i++) if ($i !~ /^--/) { print $i; break } }' "$BUILD/Dockerfile" | sort -u)
	if ! out=$(compose "$APP" pull --ignore-buildable --quiet 2>&1); then
		warn "pulling the proxy/tunnel images failed (the running ones stay):"
		printf '%s\n' "$out" | scrub | tail -n 5 | indent
	fi
	deploy "$deployed" pull
	bring_up || die $E_DOWN "the site is not healthy after the refresh"
	info "refresh done"
}

cmd_rollback() {
	local ref=${1-} target from
	load_settings
	preflight
	lock wait
	from=$(state_get deployed)
	if [[ -n $ref ]]; then
		target=$(git -C "$APP" rev-parse --verify --quiet "$ref^{commit}") || die $E_CONFIG "unknown commit '$ref' (see: stl-lab --status, git -C $APP log)"
	else
		target=$(state_get previous)
		[[ -n $target ]] || die $E_CONFIG "no previous deployment recorded; name a commit: --rollback <sha>"
	fi
	[[ $target != "$from" ]] || die $E_CONFIG "$(short "$target") is already deployed"
	check_commit "$target" || exit $E_CONFIG
	info "rolling back from $(short "$from") to $(short "$target") \"$(subject "$target")\""
	if [[ -n $(image_id "$IMAGE:$(short "$target")") ]]; then
		local rc=0
		switch_to "$target" || rc=$?
		((rc == 0)) || exit "$rc"
		record_success "$target"
	else
		info "image for $(short "$target") is gone; rebuilding it"
		deploy "$target"
	fi
	[[ -z $from ]] || {
		state_set failed "$from rollback 99 $(date +%s)"
		history_add rollback "$(short "$from")" "to=$(short "$target")"
	}
	info "rolled back to $(short "$target"). Automatic updates skip $(short "$from") and resume with the next new commit on $BRANCH (stl-lab --force deploys the branch head now)"
}

cmd_restart() { # apply stl.env changes: recreate every container of the stack
	load_settings
	preflight
	lock wait
	policy_check "$APP" || exit $E_CONFIG
	info "recreating the containers with the current stl.env"
	compose "$APP" up -d --no-build --force-recreate --remove-orphans 2>&1 | scrub | indent || {
		err "docker compose up failed"
		exit $E_DOWN
	}
	bring_up || exit $E_DOWN
	info "restart done"
}

cmd_up() { # boot (stl-compose.service): start the stack as deployed
	load_settings
	preflight
	lock wait
	if [[ -z $(state_get deployed) ]]; then
		info "nothing is deployed yet"
		exit $E_OK
	fi
	bring_up || exit $E_DOWN
}

cmd_stop() {
	load_settings
	compose "$APP" stop
}

cmd_check_config() { # install.sh: is the (staged) kit usable with this stl.env?
	load_settings
	preflight
	policy_check "$APP" || exit $E_CONFIG
	info "kit $KIT: compose configuration valid and within the safety rules"
}

cmd_status() {
	local deployed head fsha fcode fattempts fwhen url site port mode rc=0 health d src
	load_settings
	deployed=$(state_get deployed)
	head=$(state_get remote_head)
	site=$(env_get STL_SITE_ADDRESS localhost)
	port=$(env_get STL_HTTPS_PORT 443)
	mode=$(env_get STL_TLS_MODE internal)
	printf 'STL lab server — %s\n' "$(ts)"
	printf '  home        %s (user %s, project %s)\n' "$STL_HOME" "$(stat -c %U "$STATE")" "$PROJECT"
	printf '  repository  %s  branch %s\n' "$(git -C "$APP" remote get-url origin 2>/dev/null | redact)" "$BRANCH"
	if [[ -n $deployed ]]; then
		printf '  deployed    %s "%s" (since %s)\n' "$(short "$deployed")" "$(subject "$deployed")" "$(state_get deployed_at)"
	else
		printf '  deployed    (nothing yet)\n'
	fi
	if [[ -n $head ]]; then
		local when
		when=$(state_get last_fetch)
		[[ -z $when ]] || when=$(date -d "@$when" '+%Y-%m-%d %H:%M:%S')
		if [[ $head == "$deployed" ]]; then
			printf '  branch head %s = deployed (checked %s)\n' "$(short "$head")" "$when"
		else
			printf '  branch head %s NOT deployed (checked %s)\n' "$(short "$head")" "$when"
		fi
	fi
	read -r fsha fcode fattempts fwhen < <(state_get failed) || true
	if [[ -n ${fsha:-} ]]; then
		if [[ $fcode == rollback ]]; then
			printf '  rolled back %s is skipped until a new commit arrives\n' "$(short "$fsha")"
		else
			local where="journalctl -u stl-update (or $LOGS/update.log with cron)"
			[[ $fcode != "$E_BUILD" ]] || where=$LOGS/build-$(short "$fsha").log
			printf '  last failure %s code %s (%s attempt(s), %s) — log: %s\n' "$(short "$fsha")" "$fcode" "$fattempts" \
				"$(date -d "@$fwhen" '+%Y-%m-%d %H:%M')" "$where"
		fi
	fi
	[[ ! -f $STATE/paused ]] || printf '  PAUSED      %s (stl-lab --resume)\n' "$(cat "$STATE/paused")"
	printf '  containers\n'
	compose "$APP" ps --all --format '{{.Service}}\t{{.Status}}' 2>/dev/null | sed -e 's/^/              /' || true
	health=$(compose "$APP" exec -T app python -c \
		"import json,urllib.request as u; d=json.load(u.urlopen('http://127.0.0.1:8000/api/health', timeout=4)); print('ok=%s access_gate=%s workers=%s engine=%s' % (d.get('ok'), d.get('access_gate'), d.get('workers'), d.get('version')))" 2>/dev/null || true)
	if [[ $health == *"ok=True access_gate=on"* ]]; then
		printf '  health      %s\n' "$health"
	else
		printf '  health      UNHEALTHY or down %s\n' "${health:+($health)}"
		rc=$E_DOWN
	fi
	url=https://${site%%,*}
	[[ $port == 443 ]] || url=$url:$port
	printf '  url         %s/   (TLS: %s)\n' "$url" "$mode"
	printf '  images      %s\n' "$(docker image ls "$IMAGE" --format '{{.Tag}}' 2>/dev/null | paste -sd ' ' -)"
	src=$(state_kit_source)
	printf '  kit         %s (installed from %s)\n' "$KIT" "${src:+$(short "$src")}"
	if [[ -n $deployed ]]; then
		d=$(kit_diff "$deployed")
		[[ -z $d ]] || printf '  NOTE        the deployed commit has a different deploy kit (%s): review, then sudo bash %s/bin/install.sh\n' "$d" "$STL_HOME"
	fi
	if [[ -n $head ]] && [[ $head != "$deployed" ]]; then
		d=$(kit_diff "$head")
		[[ -z $d ]] || printf '  NOTE        the branch head has a different deploy kit (%s)\n' "$d"
	fi
	if [[ -d /run/systemd/system ]] && command -v systemctl >/dev/null 2>&1; then
		printf '  timers      %s\n' "$(systemctl list-timers stl-update.timer stl-refresh.timer --no-pager --no-legend 2>/dev/null | tr -s ' ' | paste -sd ';' - || echo '?')"
	fi
	exit $rc
}

# ------------------------------------------------------------------------------------------------ main
case ${1-} in
"") cmd_update 0 ;;
--force) cmd_update 1 ;;
--after-install) cmd_after_install "${2-}" ;;
--refresh) cmd_refresh ;;
--status) cmd_status ;;
--rollback) cmd_rollback "${2-}" ;;
--restart) cmd_restart ;;
--up) cmd_up ;;
--stop) cmd_stop ;;
--check-config) cmd_check_config ;;
--pause)
	load_settings
	mkdir -p "$STATE"
	state_set paused "since $(ts)${2:+: $2}"
	info "automatic updates paused (the site keeps running); resume with: stl-lab --resume"
	;;
--resume)
	load_settings
	rm -f "$STATE/paused"
	info "automatic updates resumed"
	;;
--compose)
	shift
	load_settings
	compose "$APP" "$@"
	;;
-h | --help) usage ;;
*)
	usage >&2
	exit $E_CONFIG
	;;
esac
