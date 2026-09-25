# 공개 URL로 배포하기 / Deploying to a public URL

시뮬레이터는 Python(numba) 계산 서버가 있어야 실시간으로 계산합니다. 저장소의 `Dockerfile` 하나로 프런트엔드와
서버가 함께 올라가므로, Docker를 지원하는 호스팅 중 하나를 고르면 됩니다. 어떤 방법이든 처음 빌드에 5–10분
(npm 빌드 + numba 컴파일)이 걸립니다.

The simulator computes live only with its Python (numba) server. The repository's single `Dockerfile` builds the
frontend and the server together, so any Docker host works. The first build takes 5–10 minutes (npm build +
numba compilation).

| 방법 / Option | 비용 / Cost | 메모리 / Memory | 비고 / Notes |
|---|---|---|---|
| Render (Blueprint) | Starter 유료 · paid Starter | 512 MB → `STL_WORKERS=1` | `render.yaml` 포함 · included |
| Hugging Face Spaces (Docker) | 무료 CPU basic · free | 16 GB, 2 vCPU → `STL_WORKERS=2` | 공개 Space는 코드가 공개됨 · public Spaces expose the code |
| 연구실 서버 · Lab server | — | 2 GB 이상 권장 · ≥ 2 GB | **자동 배포 키트 · auto-deploy kit: [DEPLOY_LAB.md](DEPLOY_LAB.md)** |

## 1. Render

1. <https://render.com> 에 GitHub 계정으로 로그인 → **New → Blueprint** → 이 저장소 선택 → **Apply**.
2. `render.yaml`이 Docker 웹 서비스 하나(`stl-simulator`)를 만듭니다. 배포가 끝나면 대시보드에
   `https://stl-simulator-xxxx.onrender.com` 형태의 주소가 나옵니다.
3. 메모리가 더 큰 플랜(Standard, 2 GB)에서는 환경변수 `STL_WORKERS`를 2–3으로 올리면 동시 계산이 빨라집니다.
4. Blueprint를 처음 적용할 때 `STL_ACCESS_PASSWORD` 값을 넣는 칸이 나옵니다. 비밀번호를 넣으면 링크와 비밀번호를
   모두 아는 사람만 쓸 수 있습니다(아래 [4. 비밀번호로 보호하기](#4-비밀번호로-보호하기--password-protection)).
   `render.yaml`에 `STL_REQUIRE_PASSWORD=1`이 들어 있어서, 비밀번호를 비워 두면 서버가 열린 채로 뜨지 않고 모든
   요청에 503을 돌려주며 배포가 실패합니다.

Sign in at <https://render.com> with GitHub → **New → Blueprint** → select this repository → **Apply**. The
blueprint creates one Docker web service (`stl-simulator`); its `https://…onrender.com` URL appears on the
dashboard when the deploy finishes. On a 2 GB plan raise `STL_WORKERS` to 2–3. When the blueprint is first
applied it asks for a value of `STL_ACCESS_PASSWORD`; set it to require a password (section 4). `render.yaml` sets
`STL_REQUIRE_PASSWORD=1`, so with an empty password the server does not start open: it answers 503 to everything
and the deploy fails.

## 2. Hugging Face Spaces

1. <https://huggingface.co/new-space> → SDK **Docker** → Hardware **CPU basic**. 비공개(Private)로 만들면 링크를
   아는 사람도 로그인·권한이 있어야 볼 수 있습니다.
2. Space 저장소의 `README.md` 맨 위에 다음 머리말을 두고, 이 저장소의 파일(`Dockerfile`, `engine/`, `server/`,
   `scripts/`, `web/`)을 push합니다.

   ```yaml
   ---
   title: STL Simulator
   sdk: docker
   app_port: 8000
   ---
   ```
3. Settings → Variables에 `STL_WORKERS=2`를 추가합니다. 주소는 `https://<user>-<space>.hf.space`입니다.

Create a Space at <https://huggingface.co/new-space> with SDK **Docker** (CPU basic). Put the front matter above at
the top of the Space's `README.md`, push this repository's `Dockerfile`, `engine/`, `server/`, `scripts/` and
`web/`, and set the variable `STL_WORKERS=2`. The app is served at `https://<user>-<space>.hf.space`. Make the
Space private if the unpublished model must not be public.

## 3. 연구실 서버 / Lab server

> **권장: [`docs/DEPLOY_LAB.md`](DEPLOY_LAB.md)** — `deploy/lab/`의 키트 하나로 HTTPS 프록시(Caddy), 비밀번호
> 게이트, GitHub push 자동 배포(5분마다 확인, 빌드 → 카나리 → 교체, 실패하면 이전 버전 유지·되돌리기)까지
> 설치합니다(`sudo bash install.sh`). GPT/Codex가 push만 해도 앱이 서버에 반영됩니다. 서버를 제어할 수 있는
> 배포 키트(compose 파일, Caddyfile, 업데이터)는 서버에 고정되어 push로 바뀌지 않고, 소유자가 차이를 확인한 뒤
> 적용합니다. 아래 `docker run` 방법은 자동 배포 없이 손으로 띄울 때만 씁니다.
>
> **Recommended: [`docs/DEPLOY_LAB.md`](DEPLOY_LAB.md)** — the `deploy/lab/` kit installs an HTTPS proxy (Caddy),
> the password gate and auto-deploy on GitHub push (checked every 5 min; build → canary → switch; on failure the
> previous version keeps serving or is rolled back) with `sudo bash install.sh`, so pushes from GPT/Codex go live
> on their own. The deploy kit that controls the server (compose file, Caddyfile, updater) is pinned on the server:
> pushes never change it; the owner reviews a diff and applies it. The `docker run` recipe below is for a manual
> setup without auto-deploy.

```bash
docker build -t stl-websim .
docker run -d --restart unless-stopped -p 80:8000 -e STL_WORKERS=3 -e FORWARDED_ALLOW_IPS=127.0.0.1 \
  -v stl-cache:/app/server/.cache --name stl stl-websim
```

컨테이너를 밖에 직접 열 때는 `FORWARDED_ALLOW_IPS=127.0.0.1`로 클라이언트가 보낸 `X-Forwarded-For`를 믿지 않게
합니다(기본값 `*`는 Render·HF처럼 플랫폼 프록시 뒤에 있을 때용). When the container is exposed directly, set
`FORWARDED_ALLOW_IPS=127.0.0.1` so client-sent `X-Forwarded-For` headers are ignored (the default `*` is for
platform proxies such as Render and HF Spaces).

HTTPS가 필요하면 앞단에 Caddy나 nginx를 둡니다. 이때는 컨테이너를 밖에 직접 열지 말고
`-p 127.0.0.1:8000:8000`으로 프록시에만 보이게 한 뒤 `-e STL_TRUST_PROXY=1`을 줍니다(4절). 결과 캐시는
`stl-cache` 볼륨에 남습니다.

Put Caddy or nginx in front for HTTPS. Then publish the container only to the proxy (`-p 127.0.0.1:8000:8000`
instead of `-p 80:8000`) and add `-e STL_TRUST_PROXY=1` (section 4). The result cache persists in the `stl-cache`
volume.

## 4. 비밀번호로 보호하기 / Password protection

환경변수 `STL_ACCESS_PASSWORD`를 설정하면 서버가 **링크 + 비밀번호**로 잠깁니다. 설정하지 않으면 지금처럼
누구나 쓸 수 있습니다(기본값). 잠긴 상태에서는 로그인 화면(`/login`), `/api/health`(호스팅 상태 확인용),
파비콘만 열려 있고, 화면·정적 파일·API·`/docs`·`/openapi.json`은 모두 로그인 세션이 있어야 합니다.
세션이 없으면 페이지 요청은 로그인 화면으로, API 요청은 `401 {"error": "login required"}`로 답합니다.

| 환경변수 / Variable | 필수 / Required | 설명 / Meaning |
|---|---|---|
| `STL_ACCESS_PASSWORD` | 잠글 때 / to lock | 공유할 비밀번호(앞뒤 공백은 지워집니다). 설정하면 게이트가 켜집니다. |
| `STL_REQUIRE_PASSWORD` | 권장 / recommended: `1` | 비밀이 아닌 안전장치. `1`인데 비밀번호가 비어 있으면 서버가 열린 채로 뜨지 않고 `/api/health`까지 모든 요청에 503을 돌려줍니다(배포가 실패해 바로 드러남). |
| `STL_SESSION_SECRET` | 권장 / recommended | 긴 임의 문자열(32자 이상). 없으면 프로세스마다 새로 만들어져 **서버가 재시작·재배포될 때마다 모두 다시 로그인**해야 합니다. |
| `STL_TRUST_PROXY` | Render·HF, 프록시 뒤 연구실 서버에서 `1` | 로그인 시도 제한에 쓸 접속 IP를 프록시가 붙인 `X-Forwarded-For`의 마지막 주소에서 읽습니다. **모든 요청이 프록시를 거칠 때만** 설정하세요. 직접 접속받는 서버에서 켜면 누구나 주소를 꾸며낼 수 있습니다. 설정하지 않으면 `X-Forwarded-For`가 붙은 요청은 모두 한 접속지로 셉니다. |

- **비밀번호는 저장소에 절대 넣지 않습니다.** `render.yaml`, `Dockerfile`, README, 커밋된 `.env` 어디에도 쓰지
  말고, 호스팅 대시보드의 비밀 값(secret)으로만 넣습니다.
  - **Render**: 서비스 → **Environment** → **Add Environment Variable** → `STL_ACCESS_PASSWORD` 입력 → **Save**
    (저장하면 다시 배포됩니다). Blueprint로 새로 만들면 `render.yaml`에 키만 있고(`sync: false`) 값은 Render가
    물어봅니다. **이미 있는 서비스에 Blueprint를 다시 동기화할 때는 물어보지 않으므로** Environment 탭에서 직접
    넣어야 합니다. 그 전까지는 `STL_REQUIRE_PASSWORD=1` 때문에 새 배포가 실패하고, Render는 이전 배포를 계속
    띄워 둡니다(이전 배포가 잠금 전 버전이면 여전히 열려 있음). `STL_SESSION_SECRET`은 Blueprint가 임의 값을
    자동으로 만들고(`generateValue`), `STL_TRUST_PROXY=1`, `STL_REQUIRE_PASSWORD=1`도 들어 있습니다.
  - **Hugging Face Spaces**: Space → **Settings** → **Variables and secrets** → **New secret**으로
    `STL_ACCESS_PASSWORD`, `STL_SESSION_SECRET`을 넣고, **New variable**로 `STL_TRUST_PROXY=1`,
    `STL_REQUIRE_PASSWORD=1`을 넣습니다.
    주의: 공개(Public) Space는 Files 탭에서 소스 코드(모델 포함)가 그대로 보이므로 비밀번호는 실행 중인 앱만
    보호합니다. 또 huggingface.co 페이지 안에 끼워진 화면에서는 로그인 쿠키가 유지되지 않으니
    `https://<user>-<space>.hf.space` 주소를 직접 열어 쓰세요.
  - **연구실 서버**: 저장소 밖의 파일(예: `/etc/stl/secrets.env`, 권한 600)에 값을 적고
    `docker run … --env-file /etc/stl/secrets.env -e STL_REQUIRE_PASSWORD=1 …`로 넘깁니다. `--env-file`은
    따옴표를 그대로 값에 넣으므로 `STL_ACCESS_PASSWORD=값`처럼 따옴표 없이 적습니다.
- 세션 비밀 값 만들기: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.
- 로그인 화면의 "이 기기에서 30일 동안 로그인 유지"를 켜면 30일, 끄면 브라우저를 닫을 때까지(최대 24시간)
  로그인이 유지됩니다. 로그아웃은 `/logout`입니다(다른 사이트의 링크로 열리면 확인 버튼을 한 번 더 누릅니다).
  세션은 서버에 저장되지 않으므로 로그아웃은 그 브라우저의 쿠키만 지웁니다. **모든 사람을 한꺼번에
  로그아웃시키려면 `STL_SESSION_SECRET`을 새 값으로 바꾸세요**(비밀번호를 바꿔도 기존 세션이 모두 끊깁니다).
  공용 PC에서는 "로그인 유지"를 켜지 마세요.
- 비밀번호를 5번 틀리면 30초를 기다려야 하고, 이후 틀릴 때마다 대기 시간이 두 배(최대 15분)가 됩니다. 한
  시간 동안 시도가 없으면 틀린 횟수는 잊힙니다. 서버 전체로 1분에 20번 틀리면 그 뒤로는 최근에 틀린 적이 없는
  접속지만 시도할 수 있고(1분에 10번까지), 따라서 1분에 확인하는 비밀번호는 최대 30개입니다.
- **비밀번호는 길고 추측하기 어렵게 정하세요**(예: 서로 관계없는 단어 4개 이상, 또는 16자 이상 무작위 문자).
  연구실·학교 이름에 숫자를 붙인 형태는 로그인 화면과 README에 나오는 이름으로 금방 추측됩니다. 위의 제한은
  추측 속도를 늦출 뿐 막지는 못합니다(하루 최대 약 4만 개).
- HTTPS는 Render와 Hugging Face가 자동으로 제공하므로 비밀번호와 쿠키가 암호화되어 오갑니다(쿠키에 `Secure`가
  붙습니다). 연구실 서버는 Caddy·nginx로 HTTPS를 붙이세요. HTTPS 없이 쓰면 비밀번호가 평문으로 전송됩니다.
- 서버 로그 첫머리에 `STL: access gate ON (…)`이 찍힙니다. `OFF`나 `MISCONFIGURED`면 설정을 확인하세요.
- **링크를 공유하기 전에 꼭 확인하세요**: `curl -s -o /dev/null -w "%{http_code}\n" https://<주소>/api/meta` →
  `401`, `curl -s https://<주소>/api/health` → `200`이고 `"access_gate":"on"`.

Set `STL_ACCESS_PASSWORD` to lock the server behind **link + password**; leave it unset for open access (the
default). When locked, only the login page (`/login`), `/api/health` (host health checks) and the favicon are
open; pages, static files, the API, `/docs` and `/openapi.json` need a session. Without one, page requests get
the login page and API requests get `401 {"error": "login required"}`.

- **Never put the password in the repository** (not in `render.yaml`, the `Dockerfile`, a README or a committed
  `.env`); set it only as a secret in the host's dashboard. Surrounding whitespace is removed from the value.
  **Render**: service → **Environment** → **Add Environment Variable** → `STL_ACCESS_PASSWORD` → **Save**
  (redeploys). A new blueprint declares the key with `sync: false`, so Render asks for the value; **re-syncing the
  blueprint of an existing service does not ask**, so add the value in the Environment tab. Until then
  `STL_REQUIRE_PASSWORD=1` makes the new deploy fail and Render keeps the previous deploy running (still open if
  that was a pre-gate version). The blueprint also generates a random `STL_SESSION_SECRET` (`generateValue`) and
  sets `STL_TRUST_PROXY=1` and `STL_REQUIRE_PASSWORD=1`. **Hugging Face Spaces**: **Settings** → **Variables and
  secrets** → **New secret** for `STL_ACCESS_PASSWORD` and `STL_SESSION_SECRET`, **New variable**
  `STL_TRUST_PROXY=1` and `STL_REQUIRE_PASSWORD=1`. A public Space still shows its source code (the model) in the
  Files tab, so the password protects only the running app; and the view embedded in huggingface.co cannot keep
  the login cookie, so open `https://<user>-<space>.hf.space` directly. **Lab server**: keep the values in a file
  outside the repository (mode 600), pass it with `--env-file` plus `-e STL_REQUIRE_PASSWORD=1`, and write the
  value without quotes (`--env-file` keeps quotes as part of the value).
- `STL_REQUIRE_PASSWORD=1` (not a secret) is a safety net: if the password is then missing or empty, the server
  does not run open — every request, `/api/health` included, answers 503, so the host's health check fails.
- `STL_SESSION_SECRET` (a long random string, e.g. `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)
  keeps sessions valid across restarts and redeploys; without it a random per-process secret is used and everyone
  has to sign in again after every restart. `STL_TRUST_PROXY=1` makes the login limiter use the address the host's
  proxy appended to `X-Forwarded-For`; set it **only when every request passes through that proxy** (Render, HF, a
  lab server behind Caddy/nginx published on 127.0.0.1) — on a directly reachable server anyone could forge the
  address. Without it, all requests that carry `X-Forwarded-For` count as one client.
- "Keep me signed in" = 30 days; otherwise until the browser closes (at most 24 h). `/logout` signs out (opened
  from another site's link, it asks for a click first). Sessions are not stored on the server, so logging out
  only deletes that browser's cookie; **to sign everyone out at once, set a new `STL_SESSION_SECRET`** (changing
  the password also signs everyone out). Do not use "keep me signed in" on shared PCs.
- Five wrong passwords → 30 s wait, doubling with each further failure (max 15 min); a client's failures are
  forgotten after an idle hour. After 20 failures server-wide within a minute, only clients without recent
  failures may try (10 per minute), so at most 30 passwords per minute are checked.
- **Choose a long password that is hard to guess** (4+ unrelated words, or 16+ random characters). A lab or school
  name with digits is guessed quickly from the names on the login page and in the README; the limits above only
  slow guessing down (about 40 000 per day at most).
- Render and Hugging Face serve HTTPS, so the password and the cookie are encrypted in transit (the cookie is
  marked `Secure`). Put Caddy or nginx with HTTPS in front of a lab server; without HTTPS the password is sent in
  clear text.
- The server log starts with `STL: access gate ON (…)`; `OFF` or `MISCONFIGURED` means the variables are wrong.
- **Check before sharing the link**: `curl -s -o /dev/null -w "%{http_code}\n" https://<host>/api/meta` → `401`;
  `curl -s https://<host>/api/health` → `200` with `"access_gate":"on"`.

## 모델 공개 범위 / Keeping the model private

아직 발표 전 모델이므로 공개 URL에는 접근 제한을 두는 것을 권합니다. 가장 간단한 방법은 위 4절의 비밀번호
보호이고, 그 밖에 HF Private Space, 연구실 서버의 VPN·방화벽도 쓸 수 있습니다. 브라우저에는 계산 결과만
전송되고 모델 코드는 서버에만 있지만, 물리 모델 탭에는 수식이 그대로 나옵니다.

**비밀번호는 실행 중인 앱만 보호합니다.** GitHub 저장소가 공개(Public)이면 모델 코드와 데이터(`engine/`)를 누구나
링크·비밀번호 없이 받을 수 있습니다. 저장소를 비공개로 바꾸세요: GitHub → 저장소 **Settings** → **General** →
**Danger Zone** → **Change visibility** → Private. Render와 HF는 GitHub 앱 권한으로 비공개 저장소에서도 빌드합니다.

The model is unpublished, so restrict access to a public URL — most simply with the password gate of section 4;
a private Space on HF or VPN/firewall rules on a lab server also work. Only results reach the browser and the
model code stays on the server, but the Physics tab shows the equations in full.

**The password protects only the running app.** If the GitHub repository is public, anyone can download the model
code and data (`engine/`) without the link or the password. Make the repository private: GitHub → repository
**Settings** → **General** → **Danger Zone** → **Change visibility** → Private. Render and HF can still build from a
private repository through their GitHub app.
