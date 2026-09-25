# 연구실 서버에 올리기 — 자동 업데이트 · HTTPS · 비밀번호

연구실 리눅스 서버에 시뮬레이터를 **실시간 서버(live server)** 로 띄우고, GitHub에 push된 코드를 서버가
**자동으로** 받아서 다시 배포하게 만드는 방법입니다. GPT/Codex는 GitHub에 push만 할 수 있으므로, 서버가 5분마다
배포 브랜치(deploy branch)를 확인해 새 커밋이 있으면 빌드하고, 건강 검사(health check)를 통과한 경우에만 바꿔
끼웁니다. 링크와 비밀번호를 모두 아는 사람만 쓸 수 있습니다.

필요한 파일은 모두 `deploy/lab/`에 있습니다.

| 파일 | 역할 |
|---|---|
| `deploy/lab/install.sh` | 서버 설정(처음 한 번, 이후 키트 갱신 때 다시 실행 — 안전함) |
| `deploy/lab/update.sh` | 자동 업데이트(5분마다), 상태 확인, 되돌리기, 베이스 이미지 갱신. 서버에서는 `stl-lab` 명령 |
| `deploy/lab/docker-compose.yml` | 컨테이너 구성: `app`(시뮬레이터) + `caddy`(HTTPS 프록시) + 선택 프로필 |
| `deploy/lab/Caddyfile` | HTTPS, 보안 헤더, 압축, 시간 제한, 요청 크기 제한 |
| `deploy/lab/stl.env.example` | 설정 파일 견본. 서버에서 `/opt/stl-sim/stl.env`(비밀번호 포함, 커밋 금지)로 복사됨 |
| `deploy/lab/systemd/` | `stl-update.timer`(5분마다), `stl-refresh.timer`(매주 베이스 이미지 갱신), `stl-compose.service`(부팅 시 시작) |
| `deploy/lab/tailscale-funnel.json` | Tailscale Funnel 프로필 설정 |

---

## 0. 전체 흐름과 보안 모델

```
GPT/Codex ──push──▶ GitHub (private 저장소, 배포 브랜치)
                              │  읽기 전용 배포 키(deploy key)로 git fetch
                              ▼
연구실 서버  stl-update.timer (5분마다) → /opt/stl-sim/bin/update.sh (root 소유, 고정)
              1) 새 커밋? → 깨끗한 작업 폴더에 checkout → docker compose build (이미지 태그 = 커밋 해시)
              2) 카나리(canary): 새 이미지를 네트워크·권한 없이 따로 띄워 health check + 로그인 + 계산 2건(기준, L 400 nm) 확인
              3) 통과하면 교체(switch): docker compose up -d → app health check → 프록시 경유 확인
              4) 어디서든 실패하면 이전 버전을 그대로 유지하거나 되돌림(rollback)
                              │
사용자 브라우저 ──HTTPS──▶ caddy (80/443) ──내부 네트워크──▶ app:8000 (로그인 게이트)
```

- 빌드하는 동안에도 기존 버전이 계속 서비스합니다. 교체 순간에는 Caddy가 요청을 최대 30초까지 붙잡고 기다려
  주므로 사용자는 오류 없이 잠깐 느려질 뿐입니다(시험에서 교체 중 요청 215개 모두 성공). 로그인도 유지됩니다.
- `app` 컨테이너는 호스트 포트에 열리지 않습니다(not published). 외부 인터넷으로 나가는 길도 없는 내부
  네트워크(`internal: true`)에만 붙어 있고, 호스트도 그 네트워크에 주소가 없어서(`inhibit_ipv4`) 서버에서 앱으로
  직접 연결할 수 없습니다. 오직 Caddy(와 선택한 터널 컨테이너)만 접근합니다.
- 앱 프로세스는 서버의 어떤 계정과도 겹치지 않는 uid(기본 61000, `STL_APP_UID`)로 돕니다. 서버에 로그인한 다른
  사용자가 앱 프로세스의 환경 변수(비밀번호)를 읽거나 파일을 고칠 수 없습니다.
- 비밀번호와 세션 키는 서버의 `/opt/stl-sim/stl.env`(권한 600, git 작업 폴더 밖)에만 있습니다. 저장소에는 절대
  들어가지 않습니다.

### 0.1 push가 바꿀 수 있는 것과 없는 것 (중요)

| push로 자동 반영됨 | push로 바뀌지 않음(서버에 고정, 소유자가 검토 후 `install.sh` 재실행) |
|---|---|
| 앱 코드와 이미지(`engine/`, `server/`, `web/`, `Dockerfile`, 의존성) | `deploy/lab/docker-compose.yml`, `Caddyfile`, `tailscale-funnel.json` |
| | `deploy/lab/update.sh`, `install.sh`, `systemd/*` |
| | `/opt/stl-sim/stl.env`(비밀번호, 설정) |

이렇게 나눈 이유: 서비스 계정 `stl`은 docker 그룹에 있고, **docker 그룹 = 서버의 root 권한**입니다. compose
파일을 바꿀 수 있는 사람은 `privileged: true`나 `/:/host` 마운트 한 줄로 서버 전체를 가질 수 있습니다. 그래서
키트 파일은 `install.sh`가 root 소유의 `/opt/stl-sim/kit/`와 `/opt/stl-sim/bin/`에 복사해 두고, 업데이터는 그
복사본만 씁니다. 키트를 바꾸는 커밋이 오면 업데이터는 앱은 배포하되 로그와 `--status`에 **"kit changed"** 를
남깁니다. 소유자가 변경 내용을 보고 `sudo bash /opt/stl-sim/bin/install.sh`를 다시 실행하면, 설치 스크립트가
차이(diff)를 보여 주고 묻고, 검사(`bash -n`, compose 안전 규칙, `caddy validate`)를 통과한 경우에만 설치합니다.
업데이터는 빌드·시작 전마다 compose 설정을 다시 검사해 privileged, 호스트 네트워크·PID·IPC, 추가 capability,
장치, 키트 파일 외의 호스트 폴더 마운트, 빌드 네트워크·secret·ssh 같은 설정이 있으면 거부합니다.

그래도 남는 것: **배포 브랜치에 push할 수 있는 사람은 앱을 마음대로 바꿀 수 있습니다.** 앱은 접속 비밀번호를
알고 있으므로(로그인 확인용), 악의적인 코드는 비밀번호를 화면에 노출하거나 사용자에게 엉뚱한 결과를 보여 줄 수
있습니다(서버 자체의 root는 아님). 그래서:

- GitHub 계정에 **2단계 인증(2FA)** 을 켜고, 배포 브랜치에 **branch protection**(force-push 금지, 가능하면 PR과
  리뷰 필수)을 겁니다. GPT/Codex에 준 GitHub 권한(토큰)이 새면 곧 앱이 바뀐다는 뜻입니다.
- GPT가 이상한 지시(프롬프트 인젝션)를 따르지 않도록, 배포 브랜치에는 소유자가 요청했을 때만 push하게 합니다(5장).
- 가능하면 다른 사람과 같이 쓰지 않는 **전용 VM**에 설치하거나, Docker를 rootless 모드로 씁니다. 공용 서버라면
  docker 그룹 사용자는 모두 root와 같다는 점을 전산 담당자와 공유하세요.

---

## 1. 준비물

### 1.1 서버

| 항목 | 권장 | 최소 |
|---|---|---|
| OS | Ubuntu 22.04/24.04, Debian 12, Rocky Linux 9 (x86_64) | systemd가 있는 리눅스 |
| CPU·메모리 | 4코어 이상, 8 GB 이상 | 2코어, 2 GB (`STL_WORKERS=1`) |
| 디스크 여유 | 30 GB 이상(이미지 3개 + 빌드 캐시) | 15 GB |
| 권한 | `sudo` (없으면 3.4) | docker 그룹 |

계산 프로세스(worker) 하나가 약 200–300 MB를 씁니다. 설치 스크립트가 CPU 수 − 1(최대 8)과 메모리를 보고
`STL_WORKERS`를 정합니다. 다른 연구실 작업과 서버를 같이 쓴다면 `STL_WORKERS`를 2–3으로 낮추세요.

### 1.2 소프트웨어

- **Docker CE ≥ 27 + compose plugin ≥ 2.24 + buildx** — 배포판의 `docker.io`나 `podman-docker`가 아니라
  <https://docs.docker.com/engine/install/> 의 공식 저장소 방법으로 설치합니다.
  - Ubuntu: "Install using the apt repository" → `sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`
  - Rocky: RHEL 안내 → `sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`
  - 그다음 `sudo systemctl enable --now docker`
- `git`, `openssh-client`(Rocky: `openssh-clients`), `util-linux`(flock, runuser), `python3`(compose 안전 검사) —
  대부분 이미 있습니다. `tmux`도 권장합니다(3.2).

### 1.3 네트워크 — 먼저 학과·연구실 전산 담당자에게 물어볼 것

접속 방식(4장의 A/B/C)은 학교 방화벽 정책에 달려 있습니다. 설치 전에 아래를 확인하세요.

1. 서버에 **고정 IP**가 있는가? 교내 DNS 이름(예: `stl.xxx.kaist.ac.kr`)을 등록할 수 있는가?
2. **인터넷(교외)에서 이 서버의 80, 443 포트로 들어오는 연결**을 허용해 줄 수 있는가(방화벽 예외 신청)?
3. 서버에서 밖으로 나가는 연결이 허용되는가: GitHub **SSH**(`github.com:22`, 막혀 있으면 `ssh.github.com:443`),
   Docker Hub, PyPI, npm. (Cloudflare Tunnel/Tailscale을 쓴다면 그 서비스로 나가는 443도.) 비공개 저장소는
   SSH 배포 키로만 받으므로 HTTPS만 열려 있으면 `ssh.github.com:443`을 씁니다(7장).
4. 나가는 연결에 **HTTP 프록시**가 필요한가? 필요하면 7장의 프록시 항목대로 Docker에 설정합니다.
5. 외부 공개 서비스에 대한 학교 보안 규정(신고·승인 절차)이 있는가?

요청 예시: "연구용 웹 시뮬레이터(비밀번호로 보호, HTTPS)를 연구실 서버 <IP>에서 운영하려고 합니다. 외부에서
TCP 80/443 인바운드 허용과, 가능하면 DNS 이름 등록을 요청드립니다. 불가능하면 교내에서만 접속하도록 쓰겠습니다."

| 답 | 고를 방식 |
|---|---|
| 80/443 개방 + DNS 이름 가능 | **(A) 도메인 + Let's Encrypt** — 가장 깔끔하고 안전함 |
| 포트를 열 수 없지만 밖에서도 써야 함 | **(C) Cloudflare Tunnel 또는 Tailscale Funnel** — 들어오는 포트 없이 공개 |
| 교내에서만 접속하면 됨(또는 VPN) | **(B) 교내 전용, 자체 인증서(internal TLS)** — 4장 (B)의 주의 사항을 꼭 읽기 |

---

## 2. GitHub 저장소 준비

1. **저장소를 비공개(private)로 전환**: GitHub → 저장소 **Settings** → **General** → 맨 아래 **Danger Zone** →
   **Change visibility** → Private. 공개 저장소면 비밀번호와 상관없이 누구나 모델 코드(`engine/`)를 받을 수
   있습니다. 전환 전에 GPT/Codex의 GitHub 연결이 private 저장소를 읽고 push할 수 있는지 확인하세요
   (`docs/HANDOFF_GPT.md` §9).
2. **배포 브랜치(deploy branch) 정하기**: 서버는 이 브랜치의 최신 커밋을 배포합니다.
   - **권장: 먼저 PR #1을 병합(merge)하고 `main`을 배포합니다.** 그러면 GPT는 다른 브랜치에서 작업하고, 배포할
     때만 `main`에 병합(또는 소유자가 PR을 병합)합니다. 작업 중인 push가 곧바로 공개되지 않습니다.
   - 병합 전에 설치하려면 `claude/stl-simulator-web-j0yy9i`를 배포 브랜치로 둘 수 있습니다(지금 `main`에는 예전
     Streamlit 앱만 있어서 설치 스크립트가 멈춥니다). 이때는 **그 브랜치에 push하는 모든 작업이 5–20분 안에
     공개**되므로, GPT에게 새 작업은 다른 브랜치에서 하고 소유자가 배포를 요청할 때만 이 브랜치에 push하라고
     알려 줍니다. 나중에 병합하면 `stl.env`의 `DEPLOY_BRANCH=main`으로 바꾸고 `sudo stl-lab --force`.
3. **branch protection**: Settings → Branches → Add rule → 배포 브랜치 이름 → "Do not allow force pushes",
   가능하면 "Require a pull request before merging". 계정에는 2FA.
4. **배포 키(deploy key)**: 설치 스크립트가 서버 전용 SSH 키를 만들고 공개 키 한 줄과 등록할 페이지를 보여 줍니다.
   - 페이지: <https://github.com/JunhyoungPark-NOBEL/Single-Transistor-Latch-simulator-SOI-/settings/keys/new>
   - Title: `stl-lab-server (서버 이름)`, Key: 화면의 `ssh-ed25519 …` 한 줄 전체
   - **"Allow write access"는 체크하지 않습니다(read-only).** 서버는 읽기만 합니다. 서버가 해킹되더라도 저장소를
     고칠 수 없습니다.

---

## 3. 설치 (`install.sh`)

### 3.1 스크립트를 서버로 가져오기

저장소가 private이면 서버에서 바로 받을 수 없으므로(배포 키는 설치 중에 만들어짐), 스크립트 파일 하나만
먼저 옮깁니다.

- 브라우저: GitHub에서 `deploy/lab/install.sh` → **Download raw file** → `scp install.sh 사용자@서버:~/`
- 아직 public이면 서버에서: `curl -fsSLO https://raw.githubusercontent.com/JunhyoungPark-NOBEL/Single-Transistor-Latch-simulator-SOI-/<브랜치>/deploy/lab/install.sh`

스크립트를 실행하기 전에 한 번 읽어 보세요(`less install.sh`).

### 3.2 실행

첫 빌드가 5–15분 걸리므로 SSH가 끊겨도 계속되도록 `tmux` 안에서 실행하는 것을 권장합니다(`tmux new -s stl`,
끊기면 `tmux attach -t stl`). systemd가 있으면 첫 배포는 서비스로 돌고 타이머도 먼저 켜지므로, 끊겨도 서버에서
끝까지 진행됩니다.

```bash
sudo bash install.sh
```

묻는 것(Enter = 괄호 안 기본값):

1. 배포 브랜치 — `main`(PR #1 병합 후, 권장) 또는 `claude/stl-simulator-web-j0yy9i`
2. (배포 키 등록 안내 → GitHub에 붙여 넣고 Enter. 접속이 안 되면 git의 실제 오류가 함께 나옵니다)
3. 사이트 주소 — 도메인, 교내 호스트 이름, 또는 서버 IP(쉼표로 여러 개)
4. TLS 방식 — `acme`(A) 또는 `internal`(B, C)
5. 접속 비밀번호 — 두 번 입력, 화면에 안 보임. 16자 이상, 관계없는 단어 여러 개 권장. 작은따옴표(`'`)는 불가.

그다음 자동으로: 서비스 계정 `stl` 생성(docker 그룹) → `/opt/stl-sim/app`에 clone → `/opt/stl-sim/stl.env` 작성
(권한 600, 세션 키 `openssl rand -hex 32` 자동 생성, 앱 uid는 서버에 없는 번호로) → 배포 키트 검사 후 설치
(`/opt/stl-sim/bin`, `/opt/stl-sim/kit`, root 소유) → `stl-lab` 명령과 systemd 유닛 → 타이머 켜기 →
**첫 배포(5–15분)** → 주소 출력.

옵션(`sudo bash install.sh --help`): `--home`, `--user`(예: `--user "$SUDO_USER"`로 본인 계정 사용), `--branch`,
`--repo`, `--site`, `--tls`, `--http-port`/`--https-port`, `--workers`, `--app-uid`, `--kit-from`,
`--password-stdin`, `--non-interactive`, `--reset-password`, `--no-deploy` 등.

### 3.3 다시 실행하기(키트 갱신, 설정 추가)

```bash
sudo bash /opt/stl-sim/bin/install.sh       # 설치된(고정된) 사본으로 실행
```

- clone, `stl.env`(비밀번호·세션 키·설정), 배포 키는 그대로 두고 빠진 값만 채웁니다. 서비스 계정도 기억합니다.
- 배포 브랜치 최신 커밋의 키트가 설치된 것과 다르면 **차이를 보여 주고 설치할지 묻습니다.** 누가 왜 바꿨는지
  확인하고 설치하세요(0.1). `install.sh` 자신이 바뀌었으면 새 버전으로 이어서 실행합니다.
- 새 키트가 검사를 통과하지 못하면(깨진 Caddyfile, 안전 규칙 위반 등) 설치하지 않고 이전 키트로 계속 동작하며
  종료 코드 1로 알려 줍니다.
- 마지막에 설치된 키트와 `stl.env`를 사이트에 적용하고, 평소처럼 한 번 업데이트합니다. **되돌린(rollback) 커밋과
  `--pause`는 그대로 존중합니다**(브랜치 최신을 강제로 배포하지 않음). 키트가 바뀌었으면 실패했던 커밋은 다시
  시도합니다.

### 3.4 sudo 권한이 없을 때

systemd 유닛은 설치할 수 없고 cron으로 대신합니다.

1. 관리자에게 Docker CE 설치와 **docker 그룹에 추가**(`sudo usermod -aG docker <내 계정>`)를 부탁합니다. 그다음
   다시 로그인. (docker 그룹은 root와 같은 권한이라는 점을 관리자도 알아야 합니다.)
2. 설치:
   ```bash
   bash install.sh --no-systemd --user "$USER" --home ~/stl-sim
   ```
3. 마지막에 출력되는 cron 줄 세 개(5분마다 업데이트, 매주 베이스 이미지 갱신, 부팅 시 시작)를 `crontab -e`에
   붙여 넣습니다. 명령은 `STL_HOME=~/stl-sim ~/stl-sim/bin/update.sh --status` 식으로 씁니다(`sudo` 없이).

이 방식에서는 키트 파일도 내 계정 소유이므로 0.1의 root 고정이 없습니다(어차피 docker 그룹이므로 같은 권한).

### 3.5 설치 후 서버의 구조

```
/opt/stl-sim/                     root:stl 750 — stl은 읽기만(아래 root 파일을 바꾸거나 치울 수 없음)
├── stl.env                       ★ 설정·비밀번호 (stl 소유, 권한 600, git 밖)
├── bin/update.sh, install.sh     root 소유. 업데이터(/usr/local/bin/stl-lab), 설치 스크립트 사본
├── kit/                          root 소유. docker-compose.yml, Caddyfile, tailscale-funnel.json, systemd/, SOURCE
├── app/                          git 저장소 + 지금 배포된 커밋의 checkout(참고용, 여기 파일은 실행되지 않음)
├── build/                        빌드용 작업 폴더(git worktree, 빌드마다 깨끗이 정리)
├── ssh/deploy_key(.pub)          읽기 전용 배포 키, known_hosts(GitHub 호스트 키 고정)
├── state/                        deployed / previous / failed / history / update.lock
├── logs/build-<커밋>.log          빌드 로그(최근 20개)
└── home/                         서비스 계정 stl의 홈(~/.docker 등)
/etc/systemd/system/stl-update.{service,timer}, stl-refresh.{service,timer}, stl-compose.service
```

Docker 쪽: 프로젝트 `stl`, 이미지 `stl-lab/app:<커밋 12자리>`(서로 다른 이미지 최근 3개)와
`stl-lab/app:live`(지금 쓰는 것), 볼륨 `stl_results`(결과 캐시), `stl_caddy_data`(인증서·CA 키), `stl_caddy_config`.

### 3.6 공유 전 확인

```bash
sudo stl-lab --status                                          # health ok=True access_gate=on
curl -sk -o /dev/null -w '%{http_code}\n' https://<주소>/api/meta   # 401 (로그인 없이는 막힘)
curl -sk https://<주소>/api/health                               # "access_gate":"on"
```

브라우저로 `https://<주소>/` → 로그인 화면 → 비밀번호 → 시뮬레이터.

---

## 4. 접속 방식

`stl.env`를 고친 뒤에는 `sudo stl-lab --restart`로 적용합니다(`sudoedit /opt/stl-sim/stl.env`).

### (A) 도메인 + Let's Encrypt — 권장

조건: DNS A 레코드가 서버 IP를 가리키고, **인터넷에서** 80과 443 포트로 서버에 들어올 수 있음.

```ini
STL_SITE_ADDRESS=stl.example.ac.kr
STL_TLS_MODE=acme
```

Caddy가 인증서를 자동으로 받고 갱신합니다. 브라우저 경고가 없고, HSTS 헤더가 붙습니다. 인증서 발급이
실패하면 `sudo stl-lab --compose logs caddy`에 이유(DNS, 방화벽)가 나옵니다. 발급 시도를 너무 많이 하면
Let's Encrypt가 한동안 막으므로, 방화벽이 열렸는지 먼저 확인하세요.

### (B) 교내 전용 — 자체 인증서(internal TLS)

```ini
STL_SITE_ADDRESS=143.248.x.y, stl-server.xxx.kaist.ac.kr
STL_TLS_MODE=internal
```

Caddy가 자기 인증 기관(local CA)으로 인증서를 만듭니다. 통신은 암호화되지만 브라우저가 "안전하지 않음"
경고를 띄웁니다.

> **주의 — 루트 인증서를 PC에 설치하지 마세요(특히 "로컬 컴퓨터"나 macOS "시스템" 키체인).**
> Caddy의 루트 인증서는 이름 제한(name constraints)이 없는 10년짜리 인증 기관이고, 그 개인 키는 서버의
> `stl_caddy_data` 볼륨에 있습니다. 이 키를 손에 넣은 사람(서버의 root·docker 그룹 사용자, 서버를 해킹한 사람)은
> 그 루트를 설치한 PC에서 **은행·포털·구글을 포함한 모든 HTTPS 사이트**를 가짜로 만들 수 있습니다.

대신 이렇게 합니다.

1. **권장: (A)나 (C)** — 공인 인증서라 아무것도 설치할 필요가 없습니다. 교내 DNS 이름에 대해 학교 전산팀이
   인증서를 발급해 주는지도 물어볼 만합니다.
2. **(B)를 꼭 써야 하면: 사이트별 예외** — 처음 접속할 때 브라우저 경고에서 "고급 → 계속 진행"을 눌러 이 사이트만
   예외로 둡니다. 루트를 설치하지 않으므로 다른 사이트에는 영향이 없습니다. 주소창의 인증서 정보가 매번 같은지
   (지문 fingerprint) 가끔 확인하세요:
   `sudo stl-lab --compose exec -T caddy cat /data/caddy/pki/authorities/local/root.crt | openssl x509 -noout -fingerprint -sha256`
3. 그래도 루트를 설치해야 한다면 본인 PC의 **현재 사용자** 저장소에만, 필요한 기간에만 두고, 이 위험을 사용자에게
   알립니다. 지우는 법: Windows `certmgr.msc` → 신뢰할 수 있는 루트 인증 기관 → "Caddy Local Authority" 삭제,
   macOS 키체인 접근 → 해당 인증서 삭제, Firefox 설정 → 인증서 → 기관 → 삭제.
4. `stl_caddy_data` 볼륨의 백업은 **비밀번호처럼** 다룹니다(6.6).

교외에서는 학교 VPN을 켜야 접속됩니다.

### (C) 포트를 열지 않고 캠퍼스 밖에서 — 터널

서버가 터널 업체로 **나가는** 연결만 만들고, 사용자는 업체의 HTTPS 주소로 들어옵니다. 학교 방화벽에서 들어오는
포트를 열 필요가 없고, 공인 인증서라 브라우저 경고도 없습니다. 터널만 쓸 때는 호스트 포트도 막아 둡니다.

```ini
STL_BIND_ADDR=127.0.0.1     # 80/443을 서버 밖으로 열지 않음
STL_SITE_ADDRESS=localhost
STL_TLS_MODE=internal
```

터널 컨테이너는 내부 네트워크로 `caddy:8080`(평문 HTTP, 호스트에 열리지 않음)에 연결합니다. 방문자 주소는 두
터널 모두 `X-Forwarded-For`로 전달합니다(Tailscale Funnel은 방문자 주소로 설정, Cloudflare는 맨 오른쪽에 추가).
Caddy는 내부 네트워크에서 온 이 헤더만, 그것도 내부 네트워크가 아닌 맨 오른쪽 주소만 믿고 로그인 시도 제한에
씁니다. 방문자가 직접 넣은 `X-Forwarded-For`나 `CF-Connecting-IP` 같은 헤더로는 다른 사람을 잠그거나 제한을
피할 수 없습니다(시험함). 어느 쪽이든 학교 보안 규정에 맞는지 먼저 확인하세요.

**Cloudflare Tunnel** — Cloudflare에 도메인(DNS)이 있어야 합니다(무료 플랜 가능).

1. Cloudflare 대시보드 → **Zero Trust** → **Networks** → **Tunnels** → **Create a tunnel** → Cloudflared →
   이름 입력 → 설치 명령에 나오는 **토큰**(`eyJ…`)만 복사합니다(명령은 실행하지 않음).
2. 같은 터널의 **Public Hostname** 추가: Subdomain `stl`, Domain `내도메인`, Service **HTTP**, URL **`caddy:8080`**.
3. `stl.env`:
   ```ini
   COMPOSE_PROFILES=cloudflare
   CLOUDFLARE_TUNNEL_TOKEN=eyJ...
   ```
4. `sudo stl-lab --restart` → `https://stl.내도메인/`. 원하면 Cloudflare Access로 이메일 인증을 한 겹 더 둘 수
   있습니다(Zero Trust → Access → Applications).

**Tailscale Funnel** — 도메인 없이 `https://<이름>.<tailnet>.ts.net` 주소가 생깁니다(개인 무료 플랜 가능).

1. <https://login.tailscale.com/admin> → **DNS**에서 MagicDNS와 **HTTPS Certificates** 켜기 → **Access controls**에서
   Funnel 허용(`nodeAttrs`에 `"attr": ["funnel"]`; 콘솔 안내를 따름).
2. **Settings → Keys → Generate auth key**(Reusable 끔, 필요하면 Tag 지정) → `tskey-auth-…` 복사.
3. `stl.env`:
   ```ini
   COMPOSE_PROFILES=tailscale
   TS_AUTHKEY=tskey-auth-...
   TS_HOSTNAME=stl-sim
   ```
4. `sudo stl-lab --restart` → 관리 콘솔 Machines에 `stl-sim`이 보이면 `https://stl-sim.<tailnet>.ts.net/`.
   기기 등록 후에는 콘솔에서 이 기기의 **key expiry를 끄면** 키가 만료돼도 계속 동작합니다.

> **주의(방화벽)**: Docker가 여는 포트(`ports:`)는 ufw/firewalld 규칙을 거치지 않고 열립니다. 80/443을 막고
> 싶으면 방화벽이 아니라 `STL_BIND_ADDR=127.0.0.1`로 막으세요. debug 포트는 항상 127.0.0.1에만 열립니다.

---

## 5. GPT/Codex의 수정이 서버에 반영되는 과정

1. GPT/Codex가 배포 브랜치에 push(또는 PR 병합)합니다.
2. 5분 안에(타이머 5분 + 무작위 지연 최대 1분) 서버가 `git fetch`로 새 커밋을 발견합니다.
3. 빌드: 서버 코드만 바뀌면 보통 1–3분, 의존성(`server/requirements.txt`, `web/package-lock.json`)이나 베이스
   이미지가 바뀌면 5–15분. 이미지에 들어가지 않는 변경(문서, `deploy/`)은 이미지가 그대로라 카나리를 건너뛰고
   재시작 없이 기록만 바뀝니다.
4. 카나리 확인(약 30–90초) → 교체 → health check → 끝. 대략 **push 후 5–20분**이면 링크에 반영됩니다.
5. 커밋이 키트(`deploy/lab/`의 compose, Caddyfile, 스크립트, 유닛)를 바꿨다면 앱만 배포되고 키트는 그대로입니다.
   로그에 `changes the deploy kit`가, `--status`에 `NOTE`가 나옵니다 → 3.3대로 검토 후 `install.sh` 재실행.

진행 상황 보기:

```bash
journalctl -u stl-update -f          # 배포 로그(타임스탬프 포함). 예: "DEPLOYED 1a2b3c4d5e6f in 184 s"
sudo stl-lab --status                # 배포된 커밋, 브랜치 최신 커밋, 마지막 실패, 키트, 컨테이너, health
ls -t /opt/stl-sim/logs/             # 빌드 로그
sudo systemctl start stl-update      # 5분을 기다리지 않고 지금 확인
```

실패하면: 빌드 실패(code 3)나 새 버전 이상(code 4)이면 **이전 버전이 계속 서비스**합니다. 실패한 커밋은 30분 뒤
한 번 더 시도하고(네트워크 일시 오류 대비), 그래도 실패하면 새 커밋이 올 때까지 건너뜁니다. 고쳐서 push하면
자동으로 다시 배포됩니다. 로그의 `ERROR` 줄과 빌드 로그 끝부분을 GPT에 붙여 주면 원인을 찾기 쉽습니다
(비밀번호는 로그에 나오지 않습니다).

GPT/Codex에 지켜야 할 규칙(HANDOFF §11, GPT_PROMPT에도 있음):

- 배포 브랜치에 push하면 **바로 공개 서비스에 반영**됩니다. 소유자가 배포를 요청했을 때만, push 전에 반드시
  테스트를 돌린 커밋만 넣습니다(`python3 -m pytest server/tests -q -m "not slow"`,
  `cd web && npm run typecheck && npm test`). 평소 작업은 다른 브랜치에서 합니다.
- 비밀번호·세션 키·토큰·`stl.env`를 절대 커밋하지 않습니다. `deploy/lab/stl.env`나 `deploy/lab/.env`가 들어 있는
  커밋은 업데이터가 배포를 거부합니다.
- `deploy/lab/`의 키트를 고쳤다면 서버에 자동 적용되지 않는다는 것과, 무엇을 왜 바꿨는지 소유자에게 알립니다
  (소유자가 diff를 보고 `install.sh`를 다시 실행). compose에 privileged·호스트 마운트 같은 설정을 넣지 않습니다.
- `Dockerfile`의 `ARG APP_UID`와 `useradd -u "$APP_UID"`를 유지합니다(서버는 서버에 없는 uid로 빌드·실행).
- force-push로 이력을 지우지 않습니다. 되돌리기는 소유자가 서버에서 `sudo stl-lab --rollback`으로 합니다.

---

## 6. 운영

모든 명령은 `sudo stl-lab …`로 실행합니다(서비스 계정 `stl`로 자동 전환). `stl-lab --help`에 전체 목록이 있습니다.

### 6.1 상태

`sudo stl-lab --status` — 배포된 커밋과 시각, 브랜치 최신 커밋(마지막 확인 시각), 마지막 실패와 로그 위치,
일시 정지 여부, 컨테이너 상태, health(`ok=True access_gate=on workers=…`), 주소, 보관 중인 이미지, 설치된 키트와
키트 변경 알림(NOTE), 타이머. 사이트가 비정상이면 종료 코드 5.

### 6.2 비밀번호 바꾸기

```bash
sudo bash /opt/stl-sim/bin/install.sh --reset-password --no-deploy   # 새 비밀번호 입력(화면에 안 보임)
sudo stl-lab --restart                                                # 적용
```

또는 `sudoedit /opt/stl-sim/stl.env`에서 `STL_ACCESS_PASSWORD='새 비밀번호'`로 고치고(작은따옴표로 감쌈)
`sudo stl-lab --restart`. 비밀번호가 쿠키 서명 키에 묶여 있어서 **바꾸면 기존 로그인이 모두 끊깁니다.**
명령줄 인자나 `export`로 비밀번호를 쓰지 마세요(쉘 기록에 남음).

### 6.3 모든 사람 로그아웃

```bash
openssl rand -hex 32        # 새 값을 /opt/stl-sim/stl.env의 STL_SESSION_SECRET에 붙여 넣기
sudo stl-lab --restart
```

### 6.4 되돌리기(rollback)와 일시 정지

```bash
sudo stl-lab --rollback              # 바로 이전 배포로
sudo stl-lab --rollback 1a2b3c4d     # 특정 커밋으로(이미지가 없으면 다시 빌드)
sudo stl-lab --force                 # 브랜치 최신 커밋을 지금 다시 배포(실패 기록·되돌림·일시 정지 무시)
sudo stl-lab --pause "시연 중"       # 자동 업데이트만 멈춤(사이트는 계속 동작)
sudo stl-lab --resume
```

되돌린 커밋은 자동 업데이트가 건너뛰고(`install.sh` 재실행도 마찬가지), 브랜치에 **새 커밋이 올라오면** 다시
자동 배포가 시작됩니다. 커밋 목록: `sudo -u stl git -C /opt/stl-sim/app log --oneline -20 origin/<브랜치>`.
서로 다른 이미지 3개(`STL_KEEP_IMAGES`)는 남아 있어 즉시 되돌아가고, 그보다 오래된 커밋은 다시 빌드합니다.

### 6.5 로그

```bash
journalctl -u stl-update --since today          # 업데이터
journalctl -u stl-refresh                       # 주간 베이스 이미지 갱신
sudo stl-lab --compose logs -f --tail 100 app   # 시뮬레이터(로그인 실패, 계산 오류 …)
sudo stl-lab --compose logs --tail 100 caddy    # 프록시(인증서, 접속)
```

컨테이너 로그는 10 MB × 5개로 자동 회전합니다. `stl-lab --compose config`는 **비밀번호를 그대로 출력**하므로
결과를 어디에도 붙여 넣지 마세요.

프록시를 거치지 않고 앱을 직접 확인하려면(서버 안에서만, 로그인 게이트는 그대로):

```bash
sudo stl-lab --compose --profile debug up -d debug      # 127.0.0.1:8000 → app (STL_DEBUG_PORT)
curl -s http://127.0.0.1:8000/api/health
sudo stl-lab --compose --profile debug rm -sf debug     # 끝나면 제거
```

### 6.6 백업

| 대상 | 필요? | 방법 |
|---|---|---|
| `/opt/stl-sim/stl.env` | **필요**(비밀번호·세션 키·설정) | `sudo cp -p /opt/stl-sim/stl.env /안전한/곳/`(권한 600 유지) |
| `/opt/stl-sim/ssh/deploy_key` | 선택 | 잃으면 새로 만들어 GitHub에 다시 등록(install.sh 재실행) |
| 볼륨 `stl_caddy_data` | internal 모드면 선택 | **비밀**(CA 개인 키 포함). 백업하면 비밀번호처럼 보관: `docker run --rm -v stl_caddy_data:/d -v "$PWD":/b alpine tar czf /b/caddy_data.tgz -C /d .` |
| 볼륨 `stl_results` | 불필요 | 결과 캐시일 뿐, 지워도 다시 계산됨 |
| 코드, `/opt/stl-sim/kit` | 불필요 | GitHub에 있음(키트는 install.sh 재실행으로 다시 설치) |

### 6.7 OS·Docker·베이스 이미지 업데이트

```bash
sudo apt update && sudo apt upgrade        # Rocky: sudo dnf upgrade
sudo reboot                                 # 필요할 때
sudo stl-lab --status                       # 재부팅 후 확인(stl-compose.service가 자동으로 띄움)
```

- Docker가 업데이트되며 재시작돼도 컨테이너는 `restart: unless-stopped`로 다시 뜹니다.
- **베이스 이미지(Python, Node)와 Caddy 이미지의 보안 업데이트**: 보통 빌드는 서버에 있는 베이스 이미지를
  그대로 쓰므로, `stl-refresh.timer`가 매주(일요일 새벽) `sudo stl-lab --refresh`를 실행합니다: `Dockerfile`의
  베이스 이미지와 Caddy·터널 이미지를 `docker pull`하고, 배포된 커밋을 `--pull`로 다시 빌드해 카나리를 거쳐
  교체합니다. 바뀐 것이 없으면 아무것도 재시작하지 않습니다. 지금 바로: `sudo stl-lab --refresh`.
  cron 설치(3.4)면 cron 줄에 들어 있습니다.
- Docker Hub가 `429 Too Many Requests`를 주면(학교 NAT 뒤에서 흔함) 잠시 뒤 다시 되거나, 서비스 계정으로
  `sudo -u stl -H docker login`(무료 계정)을 하면 한도가 늘어납니다.
- 디스크 정리: `docker system df`로 확인, `docker builder prune --filter until=720h`(한 달 넘게 안 쓴 빌드 캐시).
  업데이터는 오래된 앱 이미지를 스스로 지웁니다.

### 6.8 키트 갱신과 제거

- 저장소의 `deploy/lab/`(compose, Caddyfile, `update.sh`, `install.sh`, `systemd/`)가 바뀌면 `--status`에 NOTE가
  나옵니다. 변경을 확인하고(`sudo -u stl git -C /opt/stl-sim/app diff <설치된 커밋> origin/<브랜치> -- deploy/lab`)
  `sudo bash /opt/stl-sim/bin/install.sh`를 실행합니다(3.3). 업데이터가 스스로를 바꾸지 않는 것은 push 하나로
  서버 전체나 자동 배포 자체가 넘어가지 않게 하려는 것입니다.
- 제거: `sudo systemctl disable --now stl-update.timer stl-refresh.timer stl-compose.service` →
  `sudo stl-lab --compose down -v` →
  `sudo rm -rf /opt/stl-sim /usr/local/bin/stl-lab /usr/bin/stl-lab /etc/systemd/system/stl-*` →
  `sudo userdel stl` → GitHub에서 배포 키 삭제. (internal 모드에서 PC에 루트 인증서를 설치했다면 그것도 삭제, 4장 (B).)

---

## 7. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `git fetch … failed: Permission denied (publickey)` (code 2) | 배포 키 미등록, 다른 저장소에 등록 | 2장 4번. `sudo cat /opt/stl-sim/ssh/deploy_key.pub`를 해당 저장소 Deploy keys에 추가 |
| `Connection timed out` / `Connection refused` (github.com:22) | 학교가 나가는 SSH를 막음 | `stl.env`: `STL_REPO_URL=ssh://git@ssh.github.com:443/JunhyoungPark-NOBEL/Single-Transistor-Latch-simulator-SOI-.git` → `sudo bash /opt/stl-sim/bin/install.sh` (호스트 키는 고정되어 있음) |
| 나가는 연결에 HTTP 프록시가 필요 | 학교 정책 | Docker 데몬: `sudo systemctl edit docker` → `[Service]` `Environment="HTTPS_PROXY=http://proxy:port" "NO_PROXY=localhost,127.0.0.1"` → 재시작. 빌드·컨테이너: 서비스 계정의 `/opt/stl-sim/home/.docker/config.json`에 `"proxies": {"default": {"httpsProxy": "…", "noProxy": "…"}}`. GitHub SSH는 HTTP 프록시로 통과하지 않으므로 `ssh.github.com:443` 직접 허용을 전산 담당자에게 요청 |
| `Host key verification failed` | known_hosts 손상 | `sudo bash /opt/stl-sim/bin/install.sh` 재실행(GitHub 키를 다시 고정) |
| `branch … does not contain deploy/lab/` | 배포 브랜치가 PR #1 병합 전 `main` | PR #1 병합, 또는 `--branch claude/stl-simulator-web-j0yy9i` |
| `build FAILED` (code 3) | 코드 오류, PyPI·npm·Docker Hub 접속 불가, 디스크 부족, Docker Hub 429 | 빌드 로그 끝부분 확인. 네트워크면 30분 뒤 자동 재시도 또는 `--force`. `df -h`, `docker system df`. 429면 6.7 |
| `canary … unhealthy` / `smoke test FAILED` (code 4) | 새 코드가 시작하지 못하거나 계산이 실패 | 로그에 컨테이너 마지막 줄이 나옴. 고쳐서 push(이전 버전이 계속 서비스 중) |
| `REFUSED: the compose configuration breaks the kit's safety rules` | 키트나 `STL_COMPOSE_OVERRIDE`에 위험한 설정 | 나열된 항목을 빼거나, 그 변경을 설치하지 않음(0.1) |
| `the kit of … was REFUSED` (install.sh) | 새 키트가 검사 실패(깨진 Caddyfile 등) | 출력된 오류를 GPT에 전달해 고친 뒤 다시 실행. 그동안 이전 키트로 동작 |
| `ROLLBACK FAILED` / 사이트 다운 (code 5) | 이전 버전도 뜨지 못함(디스크, 포트 충돌, Docker 문제) | `sudo stl-lab --status`, `sudo stl-lab --compose logs`, `sudo stl-lab --rollback <정상 커밋>` |
| 모든 요청이 503 | `STL_ACCESS_PASSWORD`가 비어 있음(fail closed) | 6.2로 비밀번호 설정 후 `--restart` |
| 브라우저 인증서 경고 | internal 모드 | 4장 (B): 사이트별 예외, 또는 (A)/(C) |
| Let's Encrypt 실패 | DNS가 서버를 가리키지 않음, 80/443이 인터넷에서 막힘 | `sudo stl-lab --compose logs caddy`. 전산 담당자 확인, 또는 (B)/(C) |
| `port is already allocated` | 서버에 nginx/apache가 80/443 사용 중 | 그 서비스를 끄거나 `STL_HTTP_PORT`/`STL_HTTPS_PORT` 변경 |
| `Pool overlaps with other one on this address space` | 내부 네트워크 대역 충돌 | `STL_NET_SUBNET`을 다른 /24로 |
| `uid/gid … belongs to an account on this server` | `STL_APP_UID`가 서버 계정과 겹침 | `sudo bash /opt/stl-sim/bin/install.sh --app-uid <비어 있는 번호> --no-deploy` → 아래 줄처럼 결과 캐시 볼륨을 지우고 `sudo stl-lab --force` |
| 결과 캐시 쓰기 오류(권한) | `STL_APP_UID`를 바꿈 | `sudo stl-lab --compose down` → `sudo docker volume rm stl_results` → `sudo stl-lab --up` |
| `permission denied … docker.sock` | 서비스 계정이 docker 그룹 밖 | `sudo bash /opt/stl-sim/bin/install.sh` 재실행 |
| `sudo: stl-lab: command not found` | sudo의 secure_path에 /usr/local/bin이 없음(Rocky) | install.sh 재실행(/usr/bin/stl-lab도 만듦) 또는 `sudo /usr/local/bin/stl-lab …` |
| 로그인이 "잠시 후 다시" | 5회 틀림 → 30초, 이후 두 배(최대 15분) | 기다리기. 공용 NAT 뒤에서는 여러 사람이 한 주소로 보일 수 있음 |
| 느림, 429(queue full) | 동시 계산이 많음 | `STL_WORKERS`↑(메모리 확인), `STL_MEM_LIMIT` |
| 컨테이너가 자꾸 재시작(OOM) | 메모리 상한 초과 | `STL_WORKERS`↓ 또는 `STL_MEM_LIMIT`↑ |
| 타이머가 안 도는 것 같음 | 유닛 비활성 | `systemctl list-timers 'stl-*'`, `systemctl status stl-update` |
| Rocky에서 Caddyfile 읽기 거부 | SELinux를 켠 Docker | Docker 기본값(SELinux off)을 쓰거나, root 소유 추가 compose 파일(`STL_COMPOSE_OVERRIDE`)에서 caddy의 Caddyfile 마운트를 `…/kit/Caddyfile:/etc/caddy/Caddyfile:ro,z`로 다시 선언 |
| 디스크 부족 | 빌드 캐시·이미지 | `docker system df`, `docker builder prune --filter until=720h` |

`update.sh` 종료 코드: 0 정상/변경 없음 · 1 설정 오류 · 2 git fetch 실패 · 3 빌드 실패(이전 버전 유지) ·
4 새 버전 이상(교체 안 함 또는 되돌림, 이전 버전 유지) · 5 사이트 비정상(사람이 봐야 함) · 75 다른 실행이 진행 중.

---

## 8. 참고

### 8.1 `stl.env` 주요 항목

전체 설명은 `deploy/lab/stl.env.example`(한국어·영어 주석)에 있습니다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `STL_ACCESS_PASSWORD` | (필수) | 접속 비밀번호 |
| `STL_SESSION_SECRET` | (필수, 자동 생성) | 로그인 쿠키 서명 키 |
| `STL_REPO_URL`, `DEPLOY_BRANCH` | GitHub SSH 주소, `main` | 배포할 저장소와 브랜치 |
| `STL_SITE_ADDRESS`, `STL_TLS_MODE` | `localhost`, `internal` | 주소와 인증서 방식(4장) |
| `STL_HTTP_PORT`, `STL_HTTPS_PORT`, `STL_BIND_ADDR` | 80, 443, 0.0.0.0 | 호스트에 여는 포트 |
| `STL_WORKERS`, `STL_MEM_LIMIT` | CPU−1(최대 8), 작업 수에 맞춤 | 성능과 메모리 상한 |
| `STL_APP_UID` | 61000부터 비어 있는 번호 | 앱 컨테이너의 uid/gid(서버 계정과 겹치면 안 됨) |
| `COMPOSE_PROFILES` | (없음) | `cloudflare`, `tailscale`, `debug`(127.0.0.1:`STL_DEBUG_PORT` → app) |
| `STL_NET_SUBNET` | 172.30.83.0/24 | 내부 네트워크(앱이 X-Forwarded-For를 믿는 범위) |
| `STL_HEALTH_TIMEOUT`, `STL_KEEP_IMAGES` | 300 s, 3 | 배포 대기 시간, 보관할 서로 다른 이미지 수 |
| `STL_MAX_BODY_KB` | 256 | 요청 본문 한도(KiB). 앱과 프록시가 같은 값을 씀(프록시가 먼저 같은 JSON 413으로 거절) |
| `STL_COMPOSE_OVERRIDE` | (없음) | 이 서버에만 필요한 추가 compose 파일(절대 경로, git 작업 폴더 밖; 안전 규칙 검사 받음) |

### 8.2 보안 설계 요약

- 링크 + 비밀번호: `server/auth.py` 게이트, `STL_REQUIRE_PASSWORD=1`(비밀번호가 없으면 열린 채로 뜨지 않고 503),
  업데이터와 health check가 `access_gate == "on"`을 확인하므로 게이트가 꺼진 버전은 교체되지 않습니다.
- push는 앱만 바꿉니다. compose 파일·Caddyfile·업데이터·유닛은 root 소유의 `kit/`·`bin/`에 고정되고, 소유자가
  diff를 확인한 뒤 `install.sh`로만 바뀝니다. 업데이터는 매번 compose 설정의 안전 규칙을 검사합니다(0.1).
- `app`은 호스트 포트 없음, 외부로 나가는 길이 없고 호스트 주소도 없는 내부 네트워크(`internal: true`,
  `inhibit_ipv4`), 서버에 없는 uid(`STL_APP_UID`), `cap_drop: ALL`, `no-new-privileges`, 메모리 상한. 카나리도
  같은 제한에 네트워크 없이(`--network none`) 일회용 임의 비밀번호로 실행됩니다.
- uvicorn은 내부 네트워크(`FORWARDED_ALLOW_IPS=STL_NET_SUBNET`)에서 온 전달 헤더만 믿고, Caddy는 내부 네트워크의
  터널이 보낸 `X-Forwarded-For`의 맨 오른쪽 외부 주소만 쓰며 그 한 주소만 앱에 넘깁니다(`STL_TRUST_PROXY=1`).
- Caddy: HTTPS, `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`, `Permissions-Policy`(카메라·마이크·위치 끔), `frame-ancestors 'none'` +
  `X-Frame-Options: DENY`, 실제 도메인(acme)에서만 HSTS, 요청 본문 256 KiB 제한(앱과 같음, 같은 JSON 413), 읽기 전용 파일 시스템.
- 비밀 값은 `stl.env`(600, git 밖)에만. 업데이터는 비밀 값을 출력하지 않고(오류 출력에서도 지움), docker compose를
  깨끗한 환경으로 실행하며, `stl.env`가 들어 있는 커밋은 배포를 거부합니다. 배포 키는 읽기 전용, GitHub 호스트
  키는 고정(`github.com`, `ssh.github.com:443`).

### 8.3 이 키트의 검증 범위

2026-09-25에 Claude 컨테이너(Ubuntu 24.04, Docker 29.3, Compose 5.1, Caddy 2.11, systemd 없음)에서 실제 Docker로
끝까지 시험했습니다. 원격 저장소는 file:// bare 저장소, 주소는 `localhost`, internal TLS, 더미 비밀번호.

- 설치: root 모드, 전용 서비스 계정(root 소유 `bin/`·`kit/`, 서비스 계정은 이 파일들을 바꾸거나 옮길 수 없음),
  sudo 없는 docker 그룹 사용자(cron 안내). 다시 실행해도 설정·비밀번호 유지, `--user` 없이도 서비스 계정을 기억.
- 접속: `/` → 로그인 화면, 로그인 없이 `/api/meta` 401, 틀린 비밀번호 401, 맞는 비밀번호 303 + 쿠키
  (`HttpOnly; Secure; SameSite=Lax`), 계산 `folds` V_LU 3.7037 V, 보안 헤더, zstd, 2 MB 본문 413, 위조한
  `X-Forwarded-For`·`CF-Connecting-IP` 무시(터널 대역 시험 포함), 앱은 uid 61000(서버의 uid 1000 계정이 환경
  변수를 읽거나 파일을 고치지 못함), 호스트 포트·인터넷·호스트 직접 접근 없음.
- 자동 배포: 문서만 바뀐 커밋(재시작 없음), 코드 커밋(교체 중 요청 모두 성공), 시작 못 하는 커밋(code 4),
  빌드 실패 커밋(code 3), 수정 커밋으로 자동 복구, `privileged`·`/:/host`를 넣은 compose 커밋(적용되지 않음,
  install.sh도 거부), 깨진 Caddyfile 커밋(적용되지 않음, install.sh가 `caddy validate`로 거부), 키트 갱신
  (diff → 새 install.sh로 이어서 실행), 되돌린 뒤 install.sh 재실행(되돌림 유지), `--refresh`(베이스 이미지 갱신
  반영, 이전 이미지 정리), 동시 실행(75), 호출한 사람의 작업 폴더를 서비스 계정이 못 읽는 경우.
- `systemd-analyze verify`(설치 경로로 치환한 유닛), `bash -n`, `shellcheck`.

실제 서버에서 처음 확인하게 되는 것: systemd 타이머의 실제 동작과 첫 배포의 서비스 실행, GitHub 배포 키(SSH,
`ssh.github.com:443` 포함)로 private 저장소 fetch, Let's Encrypt 발급, Cloudflare Tunnel·Tailscale Funnel(명령
형식과 헤더 처리만 확인), Rocky/SELinux.
