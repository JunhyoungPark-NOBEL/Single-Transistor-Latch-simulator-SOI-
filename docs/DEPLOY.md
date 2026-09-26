# STL simulator 연구실 서버 배포

전체 명령은 [STUDIO_START_KO.md](../STUDIO_START_KO.md), 설정 예시는 [deploy/](../deploy/)에 있습니다.

## 권장 구성

브라우저 → HTTPS nginx → `127.0.0.1:8000`의 단일 API → 독립 계산 프로세스 풀.

`compose.yml`은 기본적으로 loopback 포트만 게시합니다. 외부 접속은 `deploy/nginx.conf.example`의 도메인과 인증서를 설정한 프록시로 연결합니다. 프록시 예시는 사용자가 보낸 전달 헤더를 덮어씁니다. 이 구성에서만 `STL_TRUST_PROXY=1`, `FORWARDED_ALLOW_IPS=*`를 사용합니다. 프록시를 거치지 않는 직접 LAN 구성은 `STL_TRUST_PROXY=0`, `FORWARDED_ALLOW_IPS=127.0.0.1`을 사용합니다.

- Docker: `docker compose up -d --build`
- Docker 없이: `launch.py --setup-only` 후 `deploy/biristor-studio.service.example`을 서버 경로에 맞춥니다.
- 단일 API를 유지하며 병렬 계산은 `STL_WORKERS`로 조절합니다.
- 서버 캐시는 Compose의 `studio-cache` 볼륨에 남습니다. 브라우저에 저장한 소자·회로는 그 브라우저에 남습니다.

## 로그인·원격 연결

비밀번호와 세션 키는 프로젝트 밖의 환경 파일로 주입합니다. 제공 예시는 `STL_REQUIRE_PASSWORD=1`이므로 비밀번호를 채우기 전에는 health를 포함해 503을 돌려줍니다. 서버가 열린 뒤에는 `/api/health`의 `ok` 및 `access_gate`를 확인할 수 있습니다.

같은 서버 주소에서 화면을 열면 기존 HttpOnly 로그인 쿠키를 사용합니다. 다른 주소의 로컬/연구실 홈페이지 화면에서 접속할 때는 `STL_CORS_ORIGINS`에 그 화면의 정확한 origin을 등록해야 합니다. 와일드카드와 경로가 붙은 주소는 지원하지 않습니다.

```text
STL_CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

다른 포트나 배포 홈페이지를 쓰면 해당 origin을 추가합니다. HTTPS 화면에서 HTTP API로 연결하지 않습니다. API 기본 URL은 `https://실제-서버-주소`이며, 화면의 **연결 → 연구실 서버**에서 지정합니다.

원격 인증은 `POST /api/session`에 비밀번호를 보내 발급받은 Bearer 토큰을 사용합니다. 토큰은 화면의 메모리에만 존재하며 새로고침하면 다시 인증합니다. 서버는 토큰 원문이 아닌 digest를 보관하고, 세션은 24시간 또는 서버 재시작 시 종료됩니다. `DELETE /api/session`은 해당 토큰을 폐기합니다. `STL_SESSION_SECRET`을 고정해도 원격 Bearer 세션은 서버 재시작을 넘기지 않습니다.

## 배포 상태

이 패키지는 실행 코드와 설치 설정을 제공합니다. 실제 연구실 서버에 설치하거나 DNS·인증서를 연결한 상태는 아닙니다. 연구실의 서버 주소·운영체제·관리자 접근 정보에 맞게 설치해야 합니다. Docker 없는 로컬 실행 경로와 Docker Compose를 이용한 연구실 설치 경로가 같은 앱과 물리 엔진을 사용합니다.
