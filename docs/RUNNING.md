# STL simulator 실행과 개발

사용자 설치와 연구실 서버 연결은 [STUDIO_START_KO.md](../STUDIO_START_KO.md)를 먼저 확인합니다.

## 로컬 실행

```bash
python launch.py
python launch.py --check
```

첫 실행만 `.venv` 설치가 필요합니다. 브라우저는 서버의 실제 health 응답 후 열립니다. 기존 환경을 명시적으로 쓰려면 `--use-current-python`을 추가합니다. `--no-install`은 환경 생성과 패키지 설치를 생략합니다. `--host` 기본값은 `127.0.0.1`, `--port` 기본값은 `8000`입니다.

## 화면 개발

Python 3.11 이상, Node.js 22가 필요합니다.

```bash
python launch.py --setup-only
cd web
npm ci
npm run dev
```

별도 터미널에서 가상 환경 Python으로 `python -m uvicorn server.main:app --port 8000`을 실행합니다. Vite 화면은 `http://127.0.0.1:5173`에서 API를 프록시합니다. 배포 화면은 `npm run build`로 생성합니다.

API는 **하나의 uvicorn 프로세스**가 계산 작업 프로세스 풀을 관리합니다. `--workers`를 uvicorn에 주지 말고 `STL_WORKERS`를 사용합니다. API import는 numba/엔진을 불러오지 않습니다. 계산 프로세스의 관리 통신은 임의 키로 인증한 private loopback TCP를 사용합니다.

## 주요 환경변수

| 변수 | 의미 |
|---|---|
| `STL_WORKERS` | 계산 프로세스 수. 제공 launcher와 Docker의 기본값 2 |
| `STL_CACHE_DIR` | 결과 캐시 경로. 기본 `server/.cache/results` |
| `STL_DISK_CACHE_MB` / `STL_MEM_CACHE_MB` | 디스크/메모리 결과 캐시. 기본 1024/256 MB |
| `STL_NODE_CACHE_MB` / `STL_JOB_RESULTS_MB` | 확률 노드/완료 결과 캐시. 기본 1024/64 MB |
| `STL_MAX_PENDING` / `STL_MAX_PENDING_PER_CLIENT` | 전체/접속지별 대기+실행 작업 수. 기본 64/16 |
| `STL_ABANDON_S` | 조회가 끊긴 작업 자동 취소. 기본 600초 |
| `STL_MAX_BODY_KB` | 요청 본문 한도. 기본 256 KiB |
| `STL_PREWARM` | 시작 시 계산 프로세스 준비. 기본 1 |
| `STL_WEB_DIST` | 빌드된 프런트엔드 경로 |
| `STL_CORS_ORIGINS` | 원격 연결을 허용할 화면 origin의 쉼표 구분 목록 |
| `STL_ACCESS_PASSWORD` | 서버 로그인 비밀번호 |
| `STL_REQUIRE_PASSWORD` | 1이면 비밀번호 미설정 시 503으로 닫힘 |
| `STL_SESSION_SECRET` | 서버 재시작 시 같은 origin의 로그인 쿠키를 유지할 키 |
| `STL_TRUST_PROXY` | 신뢰하는 프록시 hop 수. 직접 노출은 0 |
| `FORWARDED_ALLOW_IPS` | uvicorn이 신뢰하는 프록시 주소 |
| `PORT` | launcher/Docker의 기본 실행 포트. 기본 8000 |

## 검사

```bash
python -m pytest tests/test_launcher.py -q
python -m pytest server/tests -m "not slow" -q
cd web
npm run typecheck
npm test
npm run e2e
```

최초 계산은 numba 컴파일이 추가됩니다. 선택적으로 `python scripts/warmup.py --quick`을 실행해 기준 소자 커널을 준비할 수 있습니다. Docker는 빌드 시 이 빠른 준비를 수행하며 회로 커널은 첫 회로 계산 시 준비합니다.
