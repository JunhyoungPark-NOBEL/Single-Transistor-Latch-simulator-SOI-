# Single-Transistor-Latch-simulator-SOI-
Web simulator for single-transistor latch (STL) behavior in SOI MOSFETs.
## Copyright and usage
Copyright © Junhyoung Park. All rights reserved.
This repository and its source code are provided for viewing and deployment by the owner only.
No reproduction, redistribution, modification, or commercial use is permitted without explicit prior permission from the author.

## About Me
- B.S. in Electrical Engineering, KAIST
- M.S. student in Electrical Engineering, KAIST
- Email: jhpark@nobelab.kaist.ac.kr

## STL Web Simulator (deterministic + stochastic) — `engine/`, `server/`, `web/`

**한국어** — SOI 단일 트랜지스터 래치(STL)의 결정론·확률 모델(전하 보존 + Kirchhoff 평균 모델, compound
first-passage, 국소 상태)을 그대로 계산하는 웹 시뮬레이터입니다. 현재 소자는 **FDSOI · L_g 500 nm · W 200 nm ·
T_Si 50 nm · EOT 14.1 nm**이며(기준 보정: 암조건 V_G −2 V, 0.4 V/s / 광조사 보정: V_G −1.8 V, 1200 V/s),
PDSOI·Bulk 모델은 준비 중입니다. 아직 발표되지 않은 모델이므로 연구용으로만 사용하세요. 상단에서
**Deterministic / Stochastic**을 고르고, 왼쪽 맨 위 **Geometry**(L·W·T_Si·EOT·T_BOX·N_body, V_BG)와 파라미터
카드에서 조건을 정한 뒤 실행합니다. 기본 소자는 `Device 1` 하나이며, 바꾼 소자는 5칸짜리 소자 선반에 저장합니다.
모든 그룹·패널의 **상세** 버튼은 같은 화면에 작은 창을 띄워 코드와 정확히 일치하는 수식·변수표·가정을 보여 줍니다.

- **소자 (Device)** — 기본 화면은 순방향·역방향 ID–VD만 보여 주고, `?view=all`이나 “모두 보기”로 나머지 분석을
  엽니다. 구동 방식은 VSCM(전압 스윕)과 CSVM(전류 강제 발진) 두 가지입니다. Deterministic: 정상상태 I–V branch(HRS/불안정/LRS, fold 전압 V_LU·V_LD), 전류 성분, 고정
  V_D에서의 바디 전하 균형(안정/불안정 근, 준퍼텐셜), V_G 의존성(래치 창). Stochastic: MC 스윕(보정 lookup
  엔진 / 임의 조건용 일반 엔진), V_LU·V_LD 히스토그램·CDF(측정값 비교), hazard·생존 확률, 확률 V_G 곡선,
  사이클 시계열, 설계 지도.
- **회로 (Circuit)** — 바디 전하 Q_B를 상태변수로 갖는 STL 소자를 넣은 MNA 과도해석(후진 오일러/사다리꼴 +
  Newton, 적응 Δt). 확률 모드는 매 스텝 Eq. 2 캐리어 잡음 증분(Poisson 단위 사건 + II 클러스터)을 더합니다.
  빈 캔버스에서 시작하며 STL 소자 외에 기본 MOSFET·다이오드·BJT를 넣을 수 있습니다. 예제(부하선 스윕, 펄스 열,
  p-bit, 결합 쌍)는 보조 메뉴에 있습니다. LTspice·Verilog-A 내보내기(보정값은 기본 제외, 선택 시 포함).
- **레퍼런스 (Reference)** — 실제 측정 중앙값 ID–VD와 보정 조건 계산 곡선 비교(바닥 전류 제외 RMSE, ΔV_LU·ΔV_LD).
  엔진 수치 검증은 자동 테스트에 남아 있습니다.
- **문서 (Docs)** — 전체 수식 문서(목차·검색).

만든 곳: KAIST · NOBEL 연구실. 모드 막대 오른쪽 끝의 **정보**를 누르면 소개 창(버전 포함)이 열립니다.

**English** — A web simulator that runs the deterministic and stochastic single-transistor latch (STL) model for
SOI unchanged (charge-conservation + Kirchhoff mean model, compound first passage, local states). The current
device is **FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm** (reference calibration: dark, V_G −2 V,
0.4 V/s; illumination calibration: V_G −1.8 V, 1200 V/s); PDSOI and bulk models are coming later. The model is
unpublished — for research use only. Pick **Deterministic / Stochastic** at the top, set **Geometry** (L, W, T_Si,
EOT, T_BOX, N_body, V_BG; top of the sidebar) and the grouped parameters, and run. The only built-in device is
`Device 1`; edited devices go on a five-slot device shelf. The Device tab shows forward/reverse ID–VD by default
(`?view=all` opens the other analyses) and forces the device by VSCM (voltage sweep) or CSVM (current-forced
oscillation). The Circuit tab starts on an empty canvas and adds basic MOSFET/diode/BJT elements; LTspice and
Verilog-A exports leave the calibration out unless you tick the option. The Reference tab compares the measured
median ID–VD with the calibrated sweep. Every group and panel has a **Details** button that opens a compact
floating window with the exact code-level equations, variable tables and assumptions.

Made at KAIST · NOBEL Lab. **About** at the right end of the mode strip opens an About card (with the app/engine
version).

### Quick start / 빠른 시작
```bash
pip install -r server/requirements.txt          # numpy, scipy, numba, fastapi, uvicorn, orjson
cd web && npm ci && npm run build && cd ..      # builds web/dist
python3 scripts/warmup.py                        # optional: pre-compile numba kernels
uvicorn server.main:app --port 8000             # open http://127.0.0.1:8000
```
Development (hot reload): `scripts/dev.sh` → http://127.0.0.1:5173. Docker: `docker build -t stl-websim . &&
docker run -p 8000:8000 stl-websim`. Details: [`docs/RUNNING.md`](docs/RUNNING.md).
Each person's own computer (password-unlocked installer kit): [`docs/LOCAL_INSTALL.md`](docs/LOCAL_INSTALL.md).
Lab server (HTTPS, password gate, auto-update on push): [`docs/DEPLOY_LAB.md`](docs/DEPLOY_LAB.md).

### Layout / 구성
| Path | Contents |
|---|---|
| `engine/` | Model handoff package, verbatim (`python3 engine/stl_api.py` = smoke test; docs in `engine/docs/`) |
| `server/` | FastAPI service, job pool + cache, compute modules (deterministic, stochastic, circuit, validation), geometry model |
| `web/` | React + TypeScript frontend; physics content in `web/src/content/physics/topics/` |
| `docs/` | `WEB_CONTRACT.md` (API/UI contract), `API.md`, `RUNNING.md`, `CIRCUIT_SIMULATOR.md`, `GEOMETRY_MODEL_KO.md`, `SIMULATOR_EXPORTS_KO.md`, `LOCAL_INSTALL.md`, `DEPLOY_LAB.md`, `DECISIONS.md` |
| `deploy/` | `local/` installer kit sources, `lab/` Docker Compose + Caddy + auto-update kit |
| `app.py` | Legacy Streamlit simulator (below), unchanged |

Tests: `python3 -m pytest server/tests -m "not slow"`, `cd web && npm run typecheck && npx vitest run && npx playwright test`.

## Legacy Streamlit app (`app.py`)
- Branch-preserving ID-VD double sweep
- Separate BTBT / II / Out-diffusion / Recombination controls
- TSi-only unified silicon-thickness parameter
- Snap-based oscillation branch extraction
- Plotly internal-quantity graph with clickable legend
- Live SOI MOSFET cross-section that updates with geometry
