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

**한국어** — 논문 모델(전하 보존 + Kirchhoff 평균 모델, compound first-passage, local state)을 그대로 쓰는 웹
시뮬레이터입니다. 상단에서 **Deterministic / Stochastic**을 고르고, 왼쪽의 그룹별 파라미터 카드로 조건을
정한 뒤 실행합니다. 모든 그룹·패널의 **상세** 버튼은 같은 화면 위에 작은 창을 띄워, 코드와 정확히 일치하는
수식·변수표·가정을 보여 줍니다.

- **소자 (Device)** — Deterministic: 정상상태 I–V branch(HRS/불안정/LRS, fold V_LU·V_LD), 전류 성분, 고정
  V_D의 body 전하 균형(안정/불안정 근, 준퍼텐셜), V_G 곡선(래치 창). Stochastic: MC 스윕(보정 lookup 엔진 /
  임의 조건 일반 엔진), V_LU·V_LD 히스토그램·CDF(측정값 비교), hazard·생존확률, 확률 V_G 곡선, 사이클
  시계열, 설계 지도.
- **회로 (Circuit)** — STL을 body 전하 Q_B를 상태변수로 갖는 소자로 넣은 MNA 과도해석(후진 오일러/사다리꼴 +
  Newton, 적응 Δt). 확률 모드는 Eq. 2 사건 증분(Poisson 단위 사건 + II 클러스터). 테스트벤치: 부하선 스윕,
  펄스 열, p-bit, 결합 쌍.
- **검증 (Validation)** — `engine/docs/VALIDATION.md` 수치 재현(빠른/전체 검사)과 논문 그림 재현.
- **물리 모델 (Physics)** — 18개 주제의 전체 수식 문서(목차·검색).

**English** — A web simulator running the paper's STL model unchanged (charge-conservation + Kirchhoff mean
model, compound first-passage, local states). Pick **Deterministic / Stochastic** at the top, set grouped
parameters in the sidebar and run. Every group and panel has a **Details (상세)** button that opens a compact
floating window on the same screen with the exact code-level equations, variable tables and assumptions.

### Quick start / 빠른 시작
```bash
pip install -r server/requirements.txt          # numpy, scipy, numba, fastapi, uvicorn, orjson
cd web && npm ci && npm run build && cd ..      # builds web/dist
python3 scripts/warmup.py                        # optional: pre-compile numba kernels
uvicorn server.main:app --port 8000             # open http://127.0.0.1:8000
```
Development (hot reload): `scripts/dev.sh` → http://127.0.0.1:5173. Docker: `docker build -t stl-websim . &&
docker run -p 8000:8000 stl-websim`. Details: [`docs/RUNNING.md`](docs/RUNNING.md).

### Layout / 구성
| Path | Contents |
|---|---|
| `engine/` | Model handoff package, verbatim (`python3 engine/stl_api.py` = smoke test; docs in `engine/docs/`) |
| `server/` | FastAPI service, job pool + cache, compute modules (deterministic, stochastic, circuit, validation) |
| `web/` | React + TypeScript frontend; physics content in `web/src/content/physics/topics/` |
| `docs/` | `WEB_CONTRACT.md` (API/UI contract), `API.md`, `RUNNING.md`, `CIRCUIT_SIMULATOR.md`, `DECISIONS.md` |
| `app.py` | Legacy Streamlit simulator (below), unchanged |

Tests: `python3 -m pytest server/tests -m "not slow"`, `cd web && npm run typecheck && npx vitest run && npx playwright test`.

## Legacy Streamlit app (`app.py`)
- Branch-preserving ID-VD double sweep
- Separate BTBT / II / Out-diffusion / Recombination controls
- TSi-only unified silicon-thickness parameter
- Snap-based oscillation branch extraction
- Plotly internal-quantity graph with clickable legend
- Live SOI MOSFET cross-section that updates with geometry
