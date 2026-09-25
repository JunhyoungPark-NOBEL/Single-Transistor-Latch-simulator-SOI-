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
| 연구실 서버 · Lab server | — | 2 GB 이상 권장 · ≥ 2 GB | `docker run -p 80:8000 …` |

## 1. Render

1. <https://render.com> 에 GitHub 계정으로 로그인 → **New → Blueprint** → 이 저장소 선택 → **Apply**.
2. `render.yaml`이 Docker 웹 서비스 하나(`stl-simulator`)를 만듭니다. 배포가 끝나면 대시보드에
   `https://stl-simulator-xxxx.onrender.com` 형태의 주소가 나옵니다.
3. 메모리가 더 큰 플랜(Standard, 2 GB)에서는 환경변수 `STL_WORKERS`를 2–3으로 올리면 동시 계산이 빨라집니다.

Sign in at <https://render.com> with GitHub → **New → Blueprint** → select this repository → **Apply**. The
blueprint creates one Docker web service (`stl-simulator`); its `https://…onrender.com` URL appears on the
dashboard when the deploy finishes. On a 2 GB plan raise `STL_WORKERS` to 2–3.

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

```bash
docker build -t stl-websim .
docker run -d --restart unless-stopped -p 80:8000 -e STL_WORKERS=3 \
  -v stl-cache:/app/server/.cache --name stl stl-websim
```

HTTPS가 필요하면 앞단에 Caddy나 nginx를 둡니다. 결과 캐시는 `stl-cache` 볼륨에 남습니다.
Put Caddy or nginx in front for HTTPS; the result cache persists in the `stl-cache` volume.

## 모델 공개 범위 / Keeping the model private

아직 발표 전 모델이므로 공개 URL은 접근 제한(Render는 앞단 인증, HF는 Private Space, 연구실 서버는 VPN·방화벽)을
두는 것을 권합니다. 브라우저에는 계산 결과만 전송되고 모델 코드는 서버에만 있지만, 물리 모델 탭에는 수식이
그대로 나옵니다.

The model is unpublished, so restrict access (an auth proxy on Render, a private Space on HF, VPN/firewall on a
lab server). Only results reach the browser and the model code stays on the server, but the Physics tab shows
the equations in full.
