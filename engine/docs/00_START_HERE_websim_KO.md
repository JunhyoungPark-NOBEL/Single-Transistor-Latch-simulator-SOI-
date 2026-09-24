# STL 시뮬레이터 웹 구현 — Claude Code 인수인계 (2026-09-24)

## 목표
Deterministic + stochastic STL(single transistor latch) 모델을 웹에서 돌리는 시뮬레이터. 두 단계로 나눈다.
1. **소자 시뮬레이터**: V_G, 빛(I_PH), 램프 속도, cycle 수, seed를 넣으면 I–V branch, MC 스윕, V_LU/V_LD 분포, V_G 곡선, 설계 지도가 나온다.
2. **회로 시뮬레이터**: STL을 상태변수(body 전하 Q_B)를 가진 비선형 2단자(+게이트) 소자로 넣고 R, C, 전원과 함께 과도해석한다. 확률 항은 사건 증분으로 넣는다.

## 패키지 구성
- `stl_api.py` — Python 진입점. `python3 stl_api.py`가 스모크 테스트(기대값 `docs/VALIDATION.md`).
- `model/` — 논문(Humantech 초록) 모델 원본 코드와 표. 상대경로 그대로 두어야 import가 된다.
  - `janus_calibration_20260920/reader_fig3_20260921/gate_model/gate_mean.py` — 평균 모델(전하 보존 + Kirchhoff), `FastModel.classify`가 branch와 fold(V_LU, V_LD)를 준다.
  - `.../gate_model/gate_fpt.py` — compound first-passage(II 클러스터 + unit event) hazard.
  - `.../hypothesis_study_20260920/{compound_fpt.py, conditional_table.py, coupled_mean.py, avalanche/cluster_pmf.npz}` — 격자 FPT와 II 클러스터 분포.
  - `.../claude_crosscheck_20260920/joint_model/{gate_dynamic_compare.py, gate_state_lookup.npz, gate_dynamic_calibration.json, refit_3.json, check_escape.py}` — local state(OU)로 진화하는 스윕 MC, 보정값.
  - `.../idvd_model/mean_model.py, idvd_model_v3/srh_transport.py, high_injection_review/current_carrying_transport.py` — 물리 상수, SRH/전송 해법.
  - `MODEL_PARAMETERS.json` — 보정 파라미터와 소자 치수.
- `photo_extension/` — 논문 모델의 사본 `photo_mean.py`에 확장 항(p[13]~p[25])을 넣은 것. 모두 0이면 논문 모델과 동일(검증됨). `setup_photo.py`가 배선, `photo_fpt.py`가 임의 램프 속도 FPT, 나머지는 광조사 데이터 검토 스크립트.
- `data/` — 측정: `raw_VLU.npy`(400 cycle × 8조건, 1200 V/s), `idvd_light.npy`(빛 6단계 IDVD), `idvg_dark.npy`, `measured_stats.json`; 논문 소자 기록은 `model/.../outputs/measured_idvd_parsed.npz`. `data/tables/`는 JS 이식용 (u, r) 격자표(`export_tables.py`로 재생성).
- `docs/MODEL_SPEC.md` — 방정식, 상태변수, 파라미터 인덱스, 잡음 과정, 검증 수치.
- `docs/CIRCUIT_ELEMENT_DESIGN.md` — 회로 소자로 넣는 설계.

## 실행 환경
`pip install -r requirements.txt` (numpy, scipy, numba). numba 첫 컴파일 20–60 s. FPT 노드 하나 3–5 s, 스윕 MC 100 cycle 1 s 미만, 격자표 생성 조건당 30 s.

## 권장 아키텍처
- **1단계(빠른 길)**: Python을 그대로 서비스로 둔다. FastAPI + uvicorn, 엔드포인트 `branches`, `folds`, `hazard`, `sweeps`, `vg_curve`, `design_map`. 프런트는 React(Vite) + Plotly. 계산이 수 초인 FPT는 작업 큐(백그라운드) + 캐시(`photo_extension/photo_nodes/`와 같은 키).
- **2단계(브라우저 단독)**: `data/tables/`의 (u, r) 격자를 JSON으로 넘겨 JS에서 보간한다. 결정론 부분(dQ_B/dt = F, I_D)은 격자만으로 되고, 확률 부분은 같은 격자의 unit-event/II 전류에 Poisson과 클러스터 표본을 붙이면 된다(`MODEL_SPEC.md` 4절). Python 없이 회로 시뮬레이션이 가능한 경로.

## 작업 순서 제안
1. `python3 stl_api.py` 통과 확인.
2. API 서비스화(위 엔드포인트, JSON 스키마는 `stl_api.py` 함수 시그니처).
3. 프런트: 패널 4개(I–V branch + MC 스윕, V_LU/V_LD 히스토그램·CDF, V_G 곡선, 설계 지도) + 파라미터 폼(V_G, I_PH 또는 mW, 램프, cycle, seed, local state 진폭).
4. 회로: `CIRCUIT_ELEMENT_DESIGN.md`의 상태공간 소자 + MNA 과도해석(후진 오일러/사다리꼴 + Newton). 먼저 결정론, 다음 사건 증분.
5. 검증: `docs/VALIDATION.md`의 수치, 논문 Fig. 1(d)/3(b)/3(c) 재현.

## 열려 있는 문제 (코드가 답을 강요하지 않게 옵션으로 둘 것)
- 이 소자(광조사 소자)의 −1.1 V는 채널 seed가 필요하다. 확장 항 p[17], p[18](고 V_D 채널 seed) 또는 p[15](body 결합)로 넣었고 값은 미확정.
- local state의 작용점: 논문은 GIDL(p[9] 요동). 빛 아래 σ 유지에는 국소 avalanche 경로(p[21]~p[25])가 필요했다. UI에서 선택 가능하게.
- 잔류 정공(body 기억)은 스윕 사이에 body를 비운다는 가정으로 빠져 있다. 회로 시뮬레이터에서는 Q_B를 연속으로 적분하므로 자동으로 포함된다.
