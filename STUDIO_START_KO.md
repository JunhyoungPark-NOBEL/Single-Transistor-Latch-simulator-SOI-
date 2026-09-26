# STL simulator

측정 데이터와 기존 STL 수치 엔진을 바탕으로 만든 소자·회로 시뮬레이터입니다.
같은 앱을 개인 PC에서 계산하거나 연구실 서버에 연결해 사용합니다.

## 내 PC에서 실행

**Windows 10/11 x64에서는 Python을 따로 설치하지 않아도 됩니다.** 아래 실행기가 필요한 경우 함께 제공된 공식 Python 설치 파일을 실행합니다. macOS/Linux는 Python 3.11 이상이 필요합니다. 압축파일의 `web/dist/`에 완성된 화면이 포함되어 있어 **Node.js는 실행에 필요하지 않습니다**.

| 운영체제 | 실행 |
|---|---|
| Windows 10/11 · x64 | `start-local.bat` 더블클릭 · Python 자동 준비 |
| macOS / Linux | 폴더에서 터미널을 열고 `sh start-local.sh` |
| 공통 | `python launch.py` (`python3`도 가능) |

처음 실행하면 Python을 확인하고, 이 폴더의 `.venv`에 필요한 계산 패키지를 설치합니다. **계산 패키지를 받기 위해 첫 설치에는 인터넷이 필요합니다.** 이후 요구사항이 바뀌지 않으면 다시 설치하지 않습니다. 서버가 준비되면 브라우저가 `http://127.0.0.1:8000`을 엽니다. 첫 계산은 수치 커널 컴파일 때문에 시간이 더 걸릴 수 있습니다. 터미널을 유지하고, 종료는 **Ctrl+C**로 합니다.

Windows에서는 ZIP을 **전체 압축 해제**하고 실행하세요. `start-local.bat`만 꺼내거나 압축파일 내부에서 실행하면 안 됩니다. 사용 가능한 기존 64비트 Python을 먼저 확인하고, 없으면 `vendor/`에 포함된 공식 **CPython 3.13.15**를 현재 사용자용으로 설치합니다. 설치 파일이 없는 소스 패키지에서는 같은 공식 파일을 다운로드합니다. 두 경로 모두 고정 SHA-256 및 Windows 서명 검사를 거쳐 실행합니다.

자동 설치 위치는 `%LOCALAPPDATA%\BiristorStudio\Python313`입니다. 시스템 PATH·파일 연결·공용 Python 런처를 추가하지 않는 현재 사용자용 설치로 구성했습니다. 설치된 Python은 Windows 앱 목록에 등록됩니다. 일반적인 최신 Windows에서는 관리자 실행이 필요하지 않지만, OS의 필수 런타임 업데이트나 조직 정책 때문에 설치가 차단되면 오류를 표시합니다. Windows ARM64·32비트 자동 설치는 제공하지 않습니다.

진행 단계는 명령창에 표시하고 설치·서버 출력은 같은 폴더의 `studio-startup.log`에 남깁니다. Python 설치 프로그램의 상세 기록은 `studio-python-install.log`에 남습니다. 첫 설치는 수 분 걸릴 수 있습니다. 준비되면 브라우저가 자동으로 열립니다.

브라우저 대신 “Press any key”가 나오면 이미 시작 오류가 발생한 것입니다. 오류 로그가 메모장으로 열리며, 창을 닫아도 파일은 남습니다. **`studio-startup.log` 마지막 오류 부분**으로 자동 설치 실패, 패키지 다운로드 실패, 포트 충돌 등을 구분할 수 있습니다.

Windows에서 실행 옵션을 지정하려면 명령 프롬프트에서 다음처럼 실행합니다. `--check`와 `--no-install`은 Python 자동 설치도 하지 않습니다.

```bat
start-local.bat --check
start-local.bat --port 8100
start-local.bat --setup-only
```

이미 Python 명령을 사용할 수 있는 환경에서는 다음 명령도 사용할 수 있습니다.

```bash
python launch.py --check                # 설치/서버 실행 없이 의존성과 화면 확인
python launch.py --port 8100            # 8000 포트를 다른 프로그램이 사용 중일 때
python launch.py --no-browser           # 브라우저를 자동으로 열지 않음
```

한 폴더에서 두 번째 실행을 하면 안내와 함께 종료됩니다. 이미 사용하는 포트를 발견했을 때도 새 서버를 조용히 띄우거나 다른 프로그램의 화면을 열지 않습니다.

소자 저장 슬롯은 해당 브라우저에 저장됩니다. 다른 PC로 이동하려면 **소자 관리 → JSON 내보내기/가져오기**를 사용합니다. 서버를 바꿔도 현재 브라우저에 저장한 소자와 회로는 유지됩니다.

## 성능과 예상 시간

**성능** 탭에서 Detailed·Simple의 종류별 계산 시간을 비교합니다. **환경 측정**을 실행하면 현재 연결된 계산 환경에서 짧은 실제 모델 계산을 수행합니다. 로컬 실행이면 내 PC, 원격 연결이면 연구실 서버를 측정합니다. 첫 측정은 수치 커널 준비 때문에 더 오래 걸릴 수 있습니다.

소자·회로의 실행 버튼 근처에는 현재 설정의 예상 시간이 작은 숫자로 표시됩니다. 환경 측정 전의 기준 추정과 측정 후의 보정값을 구분하며, 이후 실제 계산 기록으로 계속 보정합니다. 최초 준비·서버 대기·결과 캐시는 계산 시간과 구분합니다.

표의 조건과 예측 범위는 **[PERFORMANCE_KO.md](PERFORMANCE_KO.md)**를 참고하세요.

## Simple Model

Geometry 아래 **Model → Simple Model**을 선택합니다. 바디 정공 손실을 1차 decay로 근사하고, β는 같은 Geometry에서 바이어스에 대해 상수로 둡니다. 기준 β는 L·Nbody에 따라, 주입 전류는 W·Tsi를 포함하여 함께 환산합니다.

처음 선택할 때 기존 기본 VG=−2 V를 사용 중이면 시작 바이어스를 −3 V로 바꿉니다. 초기값은 논문 수치를 기준 폭으로 환산한 출발값이며, Device 1의 새 보정 결과가 아닙니다. **HRS 보정**에서 현재 Geometry·바이어스의 래치 이전 데이터 VD(V), ID(A)를 붙여넣고 IS 또는 수명을 맞출 수 있습니다. 두 값을 함께 맞추려면 구분 가능한 전압의 HRS 점이 3개 이상 필요합니다. 계산 결과를 확인한 뒤 **적용**하면 됩니다.

Simple은 결정론적 ID–VD·CSVM·5단자 BE 회로를 지원합니다. 확률적 모델과 상용 시뮬레이터 내보내기는 아직 제공하지 않습니다. `X1.vb`는 집중 전하 저장부 전위 w이며, `X1.vbody`는 소스 기준 B 접점 전압 u입니다. Detailed의 내부 정전기 전위와 같은 정의가 아닙니다.

모델 수식, 기하 의존성과 적용 범위: **[SIMPLE_MODEL_KO.md](SIMPLE_MODEL_KO.md)**. 실행시간 비교와 확인 조건: **[docs/SIMPLE_MODEL_QA_KO.md](docs/SIMPLE_MODEL_QA_KO.md)**.

## Body RC와 게이트 변조

회로의 STL 소자는 D·S·G·BG·B의 5단자를 제공합니다. G와 BG에 전압원을 연결하고 해당 전압원에서 Pulse·Sine·PWL을 선택하면 시간에 따라 바이어스를 바꿀 수 있습니다. B에 저항·커패시터를 연결하면 접점 전류가 실제 바디 전하 방정식에 반영됩니다. 연결하지 않은 BG는 저장된 VBG, B는 floating body로 동작합니다. BG·B 외부 연결은 현재 **결정론적 · BE**에서 지원합니다.

기존 3단자 회로는 배선을 유지합니다. 기존 소자를 선택한 뒤 **5단자 확장**을 누르면 BG·B 핀이 생깁니다. Detailed Model의 `X1.vb`는 내부 정전기적 바디 전위, `X1.vbody`는 소스 기준 B 접점 전압입니다. CSVM에서는 드레인 전압과 내부 바디 전위를 함께 표시하고 첫 피크 오버슈트는 그대로 보존합니다. Vtop·Vbottom·f는 초기 구간을 제외한 완전한 반복 주기에서 측정합니다.

완성된 연결 예제는 `examples/body-rc.stl-circuit.json`입니다. 회로 화면의 **파일 → 가져오기**에서 불러온 뒤 R·C·전압원 파형을 자유롭게 바꿀 수 있습니다.

새 VBG 결합은 미보정 BJT 바디 바이어스 근사입니다. 범위와 전하식은 `GEOMETRY_MODEL_KO.md`에 있습니다.

## 연구실 서버에서 실행

### A. Docker Compose

서버에 Docker Engine과 Compose 플러그인이 필요합니다. 이 폴더 전체를 서버로 복사한 뒤 환경 파일을 프로젝트 밖에 준비합니다.

```bash
sudo install -d -m 700 /etc/biristor-studio
sudo install -m 600 deploy/server.env.example /etc/biristor-studio/server.env
sudoedit /etc/biristor-studio/server.env
```

`STL_ACCESS_PASSWORD`에 사용할 비밀번호를, `STL_SESSION_SECRET`에 아래 명령으로 만든 값을 입력합니다. 비밀번호가 비어 있으면 서버가 열리지 않도록 `STL_REQUIRE_PASSWORD=1`이 기본 설정되어 있습니다.

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
sudo docker compose up -d --build
sudo docker compose logs -f --tail 80 studio
```

Compose 기본 설정은 **서버 내부의 `127.0.0.1:8000`에만 포트를 엽니다**. `deploy/nginx.conf.example`의 도메인·인증서 경로를 서버에 맞추고 HTTPS 프록시를 연결합니다. 그 후 연구실 구성원은 `https://실제-서버-주소`를 바로 열어 로그인하면 됩니다. 이 방법은 브라우저와 계산 API가 같은 주소여서 별도의 연결 설정이 필요 없습니다.

```bash
curl http://127.0.0.1:8000/api/health
sudo docker compose down               # 종료; 계산 캐시 볼륨은 유지
sudo docker compose up -d --build      # 업데이트 후 재시작
```

상태 확인은 `ok: true`, `access_gate: "on"`을 확인합니다. Compose는 API 프로세스 하나와 계산 작업 프로세스 2개를 사용합니다. `STUDIO_WORKERS=1` 등으로 작업 수를 조절할 수 있습니다. uvicorn의 `--workers`를 추가하지 않습니다.

서버마다 다른 경로·포트는 `STUDIO_ENV_FILE`, `STUDIO_PORT`, `STUDIO_BIND_IP`로 설정합니다. 예를 들어 전용 LAN에서 직접 접속받으려면 다음처럼 주소를 명시할 수 있습니다. 이때 환경 파일에서 `STL_TRUST_PROXY=0`, `FORWARDED_ALLOW_IPS=127.0.0.1`로 바꿉니다. HTTP를 직접 쓰는 구성은 VPN 또는 신뢰할 수 있는 내부망에 한정하고, 일반 접속은 HTTPS 프록시를 사용합니다.

```bash
sudo env STUDIO_BIND_IP=0.0.0.0 STUDIO_PORT=8000 docker compose up -d
```

비밀번호·세션 키를 소스에 넣지 않습니다. `server.env.example`에는 빈 자리만 있습니다. Compose 환경 파일에서 `$`가 포함된 값은 단일 따옴표로 감싸 변수 치환을 막습니다. 환경 파일 내용을 터미널 로그나 공유 문서에 출력할 필요가 없습니다.

### B. Docker 없이 Python으로

```bash
python3 launch.py --setup-only
python3 launch.py --no-install --no-browser --workers 2
```

위 명령은 서버 자신의 `127.0.0.1:8000`에 엽니다. 지속 실행은 `deploy/biristor-studio.service.example`에서 사용자·설치 경로를 바꿔 systemd에 등록합니다. 서비스 사용자가 프로젝트와 캐시 디렉터리에 쓸 수 있어야 합니다. 비밀번호 환경 파일은 `EnvironmentFile`로 읽으며 nginx 예시는 동일하게 사용할 수 있습니다. 직접 LAN 접속을 의도할 때만 `--host 0.0.0.0`을 지정합니다.

## 내 PC의 화면을 연구실 서버에 연결

1. 로컬 앱을 실행하고 화면 위 **연결**을 엽니다.
2. **연구실 서버**를 선택하고 `https://실제-서버-주소`를 입력합니다. 끝에 `/api`는 붙이지 않습니다.
3. 필요한 경우 **서버 비밀번호**를 입력하고 **연결 확인**을 누릅니다.

연구실 서버의 `STL_CORS_ORIGINS`에 **로컬 화면을 실제로 연 주소**를 등록해야 합니다. 예를 들어 아래 두 주소는 서로 다른 origin입니다.

```text
STL_CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

`--port 8100`으로 실행했다면 `http://127.0.0.1:8100`도 추가하고 서버를 재시작합니다. 별도 홈페이지에서 화면을 제공하면 해당 `https://...` origin을 등록합니다. 와일드카드 대신 허용할 주소를 정확히 지정합니다.

원격 인증은 API 세션을 발급하고 메모리에만 둡니다. 비밀번호와 세션 토큰은 브라우저 저장소에 저장하지 않으며 새로고침 후 다시 로그인이 필요합니다. URL만 기억합니다. HTTPS 홈페이지에서 HTTP API로 연결하는 것은 브라우저가 차단하므로 두 주소를 HTTPS로 맞춥니다. 계산 서버가 연결되지 않으면 그 상태를 표시하며, 실제 계산처럼 보이는 임의 곡선으로 대체하지 않습니다.

## 소스 수정·진단

화면을 수정할 때만 Node.js 22와 npm이 필요합니다.

```bash
cd web
npm ci
npm run build
```

이미 의존성이 설치된 별도 환경에서는 `python launch.py --use-current-python --check`로 검사하고 `python launch.py --use-current-python --no-install`로 실행할 수 있습니다. API 프로세스는 수치 엔진을 직접 불러오지 않고, 별도 작업 프로세스에서 계산합니다.

macOS/Linux에 Python이 없으면 [python.org](https://www.python.org/downloads/) 또는 운영체제의 패키지 관리자로 설치합니다. Linux에서 `.venv` 생성 오류가 나면 배포판의 `python3-venv` 패키지를 설치합니다. 시작 오류는 터미널 로그를 확인합니다. `--check`는 의존성·API import·화면 빌드 검사이며 물리 모델 검증을 대신하지 않습니다. 모델 식·가정·적용 범위는 `GEOMETRY_MODEL_KO.md`와 앱의 모델 문서를 확인합니다.

## Quick start (English)

- Windows 10/11 x64: extract the entire ZIP and run `start-local.bat`. If Python is missing, the bundled official CPython 3.13.15 installer runs per user; SHA-256 and Authenticode are checked first. Internet is needed for Python packages. Other platforms: install Python 3.11+ and run `sh start-local.sh`. The UI needs no Node.js. Stop with Ctrl+C.
- Lab: configure `/etc/biristor-studio/server.env`, run `docker compose up -d --build`, and put HTTPS nginx in front of loopback port 8000. Templates are in `deploy/`.
- Remote compute: open **연결 → 연구실 서버**, enter the server base URL/password, then **연결 확인**. Add the local frontend origin to the server's `STL_CORS_ORIGINS`. Passwords and bearer tokens are kept in memory only.
- This package contains launch and deployment configurations. It has not been installed on your lab server; the real server address, domain, certificate and administrator access must be supplied by your lab.
