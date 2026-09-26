# STL simulator

FDSOI single-transistor latch 소자와 회로를 위한 웹 시뮬레이터입니다. 측정 데이터와 원본 수치 엔진을 보존하고, 독립적으로 실행할 수 있는 화면과 서버 구성을 제공합니다.

**KAIST · NOBEL Lab**

## 시작

- **Windows 10/11 · x64:** 압축을 전부 풀고 `start-local.bat`을 실행합니다. Python이 없으면 포함된 공식 설치 파일로 자동 설치합니다.
- **macOS / Linux:** 터미널에서 `sh start-local.sh`를 실행합니다.
- **연구실 서버:** 환경 파일을 준비하고 `docker compose up -d --build`로 실행합니다.

Windows에서는 Python을 따로 설치할 필요가 없습니다. 최초 실행은 Python 확인·필요 시 설치 → `.venv`에 계산 패키지 설치 → 브라우저 열기 순서로 진행합니다. 계산 패키지를 받기 위해 첫 실행에는 인터넷이 필요합니다. macOS/Linux는 Python 3.11 이상을 먼저 설치하세요. 완성된 `web/dist/`가 포함되어 있어 Node.js 없이 사용할 수 있습니다. 종료는 Ctrl+C입니다.

실행·서버 설치·원격 연결: **[STUDIO_START_KO.md](STUDIO_START_KO.md)**

성능 측정과 예상 시간: **[PERFORMANCE_KO.md](PERFORMANCE_KO.md)**

이번 실행판의 확인 결과와 한계: [docs/RELEASE_CHECKS_KO.md](docs/RELEASE_CHECKS_KO.md)

## 사용할 수 있는 기능

현재 FDSOI만 지원합니다. PDSOI·Bulk는 미지원으로 표시하며 기존 기록을 임의로 FDSOI로 변환하지 않습니다. Geometry 그림의 치수를 누르면 해당 입력으로 이동하고, 변수는 이탤릭·아래첨자는 정자로 표시합니다. 주요 입력·탭·버튼 글씨는 14px 이상이며 결정론적·확률적 모드는 화면 상단에서 전환합니다.

- 모델·종류별 성능 표, 계산 환경 측정, 실행 전 예상 시간
- Detailed / Simple Model 선택 · Simple의 1차 바디 재결합 및 HRS 보정
- VSCM의 ID–VD, CSVM의 VD(t)·Vtop·Vbottom·주파수
- L·W·Tsi·EOT·Tbox·Nbody 및 VG·VBG 설정
- 사용자 소자 5개 저장, 자유로운 회로 배치와 배선, 기본 MOSFET·diode·BJT
- STL D·S·G·BG·B의 5단자, 시간 가변 G/BG, 외부 Body R·C (결정론적 BE)
- CSVM 드레인·내부 바디 전위 동시 표시, 첫 피크 오버슈트 보존 및 이후 반복 주기 측정
- 측정 ID–VD와 기준 보정 모델 비교
- 고정 조건의 LTspice·Verilog-A 준정적 모델 내보내기
- 로컬 계산과 연구실 서버 연결 전환

Simple Model의 β는 바이어스에 대해서만 상수이며, 동일 emitter 공정 가정에서 L·Nbody에 따라 변합니다. 초기값은 기존 Device 1 측정에 맞춘 값이 아닙니다. 수식·기하 가정·사용 범위는 [SIMPLE_MODEL_KO.md](SIMPLE_MODEL_KO.md)에 있습니다. Simple은 현재 결정론적 BE를 지원합니다.

Detailed Model의 VBG는 BJT source injection의 바디 바이어스 근사로 적용합니다. 결합 계수는 정전용량 divider이며 VBG sweep으로 새로 보정한 값은 아닙니다. 내부 정전기 전위와 외부 B 접점 전압을 구분합니다. Geometry 확장은 기준 보정을 벗어나는 모델 가정을 포함합니다. 지원 범위와 확인된 조건을 화면 및 문서에서 구분하며, 없는 물리 현상이나 측정하지 않은 정확도를 주장하지 않습니다. 실제 계산 서버 연결이 끊기면 계산을 안내 없이 데모 곡선으로 바꾸지 않습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `engine/` | 변경하지 않은 원본 모델·측정 데이터 |
| `server/` | API, 독립 계산 프로세스, geometry 확장, 회로 해석 |
| `web/` | React·TypeScript 화면, 빌드된 `dist/`, 상세 안내 |
| `launch.py` | 로컬 실행·의존성 검사·종료 관리 |
| `scripts/bootstrap-windows.ps1`, `vendor/` | Windows 자동 준비 및 공식 Python 설치 파일 |
| `deploy/` | 환경 파일·nginx·systemd 예시 |
| `SIMPLE_MODEL_KO.md` | 1차 재결합 모델, 확산 β, HRS 보정과 한계 |
| `GEOMETRY_MODEL_KO.md` | Geometry의 물리 관계, 가정, 기준값 |
| `docs/` | API·모델·회로·검증 문서 |
| `app.py` | 이전 Streamlit 앱, 변경하지 않음 |

화면 소스 수정 시에만 Node.js 22와 `cd web`, `npm ci`, `npm run build`가 필요합니다. 서버·프런트엔드 개발 상세는 [docs/RUNNING.md](docs/RUNNING.md)를 참고합니다.

## English

On Windows 10/11 x64, extract the full ZIP and run `start-local.bat`; Python is installed automatically if needed using the bundled official installer. First setup needs Internet for Python packages. On macOS/Linux, install Python 3.11+ and run `sh start-local.sh`. No Node.js is needed for the bundled UI. Lab deployment and remote-compute setup are in [STUDIO_START_KO.md](STUDIO_START_KO.md).

## Copyright and usage

Copyright © Junhyoung Park. All rights reserved.
This repository and its source code are provided for viewing and deployment by the owner only.
No reproduction, redistribution, modification, or commercial use is permitted without explicit prior permission from the author.
