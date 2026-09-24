# Running the STL web simulator / STL 웹 시뮬레이터 실행

The service is one Python process (FastAPI + uvicorn) that owns a pool of compute worker processes
(numba engine) and serves the built React frontend. API details: `docs/API.md`.

서비스는 하나의 Python 프로세스(FastAPI + uvicorn)이며, 그 안에서 계산 작업 프로세스 풀(numba 엔진)을
운영하고 빌드된 React 프런트엔드를 함께 제공한다. API 세부 사항은 `docs/API.md` 참고.

---

## 한국어

### 1. 준비물
- Python ≥ 3.11, Node 22 + npm (프런트엔드 빌드/개발 시).
- Python 패키지: `pip install -r server/requirements.txt`
  (numpy, scipy, numba, fastapi, uvicorn[standard], orjson, pytest, httpx).
- 최초 1회 numba 컴파일(약 10–60 s)을 미리 해 두려면: `python3 scripts/warmup.py`
  (엔진 스모크 테스트: `python3 engine/stl_api.py`, 기대값은 `engine/docs/VALIDATION.md`).

### 2. 로컬 개발
```bash
scripts/dev.sh          # 백엔드 :8000 (server/ 수정 시 자동 재시작) + Vite :5173 (/api → :8000 프록시)
scripts/dev.sh api      # 백엔드만
scripts/dev.sh web      # 프런트엔드만 (API_PORT의 백엔드 사용)
```
브라우저에서 http://127.0.0.1:5173 을 연다. 백엔드만 직접 실행하려면
`uvicorn server.main:app --port 8000` (uvicorn `--workers`는 쓰지 말 것 — 계산 풀은 프로세스 안에 있으며,
병렬도는 `STL_WORKERS`로 조절한다).

### 3. 프로덕션 빌드 (Docker 없이)
```bash
cd web && npm ci && npm run build && cd ..        # web/dist 생성
pip install -r server/requirements.txt
python3 scripts/warmup.py                          # numba 캐시·FPT 노드 캐시 채우기
STL_WORKERS=3 uvicorn server.main:app --host 0.0.0.0 --port 8000
```
`web/dist`가 있으면 서버가 `/`에서 프런트엔드를 제공한다(SPA fallback). 없으면 안내 페이지가 나온다.

### 4. Docker
```bash
docker build -t stl-websim .
docker run --rm -p 8000:8000 -e STL_WORKERS=2 stl-websim     # http://localhost:8000
```
다단계 빌드: `node:22-slim`에서 `web/` 빌드 → `python:3.11-slim` 런타임. 빌드 중 `scripts/warmup.py`가
numba 커널을 컴파일하고 엔진 캐시를 채운다. 컨테이너는 uid 1000 사용자로 실행되며 `/app`은 쓰기
가능해야 한다(numba 캐시 `engine/**/__pycache__`, FPT 노드 `engine/photo_extension/photo_nodes/`,
결과 캐시 `server/.cache/`). 결과 캐시를 유지하려면 `-v stl-cache:/app/server/.cache`를 붙인다.

### 5. 배포 메모
- **메모리**: 작업 프로세스 하나당 약 150–250 MB(numba + 엔진). 512 MB 플랜에서는 `STL_WORKERS=1`.
- **Render**: Docker 서비스로 만들고 포트는 `PORT` 환경변수를 따른다(Dockerfile의 `CMD`가 처리).
  헬스 체크 경로 `/api/health`. 무료 플랜은 유휴 시 잠들기 때문에 첫 요청이 느리다.
- **Fly.io**: `fly launch`(Dockerfile 감지) 후 `fly.toml`에서 `internal_port = 8000`,
  `[[vm]] memory = "1gb"`, 환경변수 `STL_WORKERS`. 머신 자동 정지를 쓰면 캐시는 볼륨에 둔다.
- **Hugging Face Spaces**: SDK = Docker. README 머리말에 `app_port: 8000` (또는 `PORT=7860` 환경변수로
  맞춤). Space는 uid 1000으로 실행하므로 Dockerfile의 사용자 설정을 그대로 쓰면 된다. CPU basic
  (2 vCPU)에서는 `STL_WORKERS=1`–`2`.
- 역방향 프록시 뒤에서는 긴 요청 대신 폴링을 쓰므로 타임아웃 문제는 없다(요청당 최대 대기 60 s).

### 6. 환경변수
| 변수 | 기본값 | 의미 |
|---|---|---|
| `STL_WORKERS` | CPU − 1 (affinity/cgroup 반영), 최소 1 | 계산 작업 프로세스 수 |
| `STL_CACHE_DIR` | `server/.cache/results` | 디스크 결과 캐시 |
| `STL_DISK_CACHE_MB` | 1024 | 디스크 캐시 한도(시작 시 오래된 것부터 삭제) |
| `STL_PREWARM` | 1 | 시작 시 모든 작업 프로세스를 미리 띄움 |
| `STL_MP_CONTEXT` | `spawn` | multiprocessing 시작 방식 |
| `STL_CORS_ORIGINS` | localhost:5173, :4173 | 허용 origin(쉼표 구분) |
| `STL_WEB_DIST` | `web/dist` | 빌드된 프런트엔드 경로 |
| `PORT` | 8000 | Docker 실행 포트 |

### 7. 테스트
```bash
python3 -m pytest server/tests -q                  # 전체 (약 1 분, 작업 프로세스 2개)
python3 -m pytest server/tests -q -m "not slow"    # 빠른 테스트만
python3 scripts/warmup.py --validate               # VALIDATION.md 빠른 검증을 콘솔에 출력
```

---

## English

### 1. Requirements
- Python ≥ 3.11; Node 22 + npm for the frontend.
- `pip install -r server/requirements.txt` (numpy, scipy, numba, fastapi, uvicorn[standard], orjson, pytest, httpx).
- Pre-compile the numba kernels once (10–60 s): `python3 scripts/warmup.py`. Engine smoke test:
  `python3 engine/stl_api.py` (expected numbers in `engine/docs/VALIDATION.md`).

### 2. Local development
```bash
scripts/dev.sh          # backend :8000 (auto-reload on server/ edits) + Vite :5173 (proxies /api → :8000)
scripts/dev.sh api      # backend only
scripts/dev.sh web      # frontend only (uses the backend on API_PORT)
```
Open http://127.0.0.1:5173. Backend alone: `uvicorn server.main:app --port 8000`. Do not use uvicorn
`--workers`: the compute pool lives inside the process; scale with `STL_WORKERS`.

### 3. Production build without Docker
```bash
cd web && npm ci && npm run build && cd ..        # creates web/dist
pip install -r server/requirements.txt
python3 scripts/warmup.py                          # numba + FPT-node caches
STL_WORKERS=3 uvicorn server.main:app --host 0.0.0.0 --port 8000
```
When `web/dist` exists the server serves it at `/` with SPA fallback; otherwise a short help page.

### 4. Docker
```bash
docker build -t stl-websim .
docker run --rm -p 8000:8000 -e STL_WORKERS=2 stl-websim     # http://localhost:8000
```
Multi-stage: `node:22-slim` builds `web/`, `python:3.11-slim` runs the API; `scripts/warmup.py` runs at build
time so the numba kernels and the FPT node for the validation are cached in the image. The container runs as
uid 1000 and needs `/app` writable (numba caches in `engine/**/__pycache__`, FPT nodes in
`engine/photo_extension/photo_nodes/`, results in `server/.cache/`). Mount `-v stl-cache:/app/server/.cache`
to keep the result cache across restarts.

### 5. Hosting notes
- **Memory**: ~150–250 MB per worker process (numba + engine). Use `STL_WORKERS=1` on 512 MB plans.
- **Render**: Docker web service; the `CMD` honours `PORT`; health check path `/api/health`. Free instances
  sleep when idle (first request after a sleep is slow).
- **Fly.io**: `fly launch` detects the Dockerfile; set `internal_port = 8000`, `[[vm]] memory = "1gb"`,
  env `STL_WORKERS`. With auto-stop machines, put the result cache on a volume.
- **Hugging Face Spaces**: SDK Docker, `app_port: 8000` in the README front matter (or set `PORT=7860`).
  Spaces run as uid 1000, matching the Dockerfile user. CPU basic (2 vCPU): `STL_WORKERS=1`–`2`.
- The UI polls jobs (≤ 60 s per request), so proxy time-outs are not an issue.

### 6. Environment variables
See the table in the Korean section above (same variables) or `docs/API.md`.

### 7. Tests
```bash
python3 -m pytest server/tests -q                  # everything (~1 min with 2 workers)
python3 -m pytest server/tests -q -m "not slow"    # quick subset
python3 scripts/warmup.py --validate               # print the fast VALIDATION.md checks
```
