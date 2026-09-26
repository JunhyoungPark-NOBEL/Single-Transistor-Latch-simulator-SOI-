# STL 웹 시뮬레이터 — GPT 인수인계 문서

> 작성일 2026-09-25 · 기준 커밋 `16dd9ad` (브랜치 `claude/stl-simulator-web-j0yy9i`)
> 이 문서에는 Claude Code 세션에서 코드와 문서를 직접 읽거나 실행해 확인한 사실만 담았다. "미확인"으로 표시한 항목은 이번 점검에서 실행하지 않았다.
> 비밀번호와 개인 연락처는 이 문서에 없다. 잠긴 페이지의 비밀번호는 소유자가 따로 전달한다.
> 이 문서와 `docs/GPT_PROMPT.md`는 기준 커밋 바로 다음 커밋으로 저장소 `docs/`에 들어 있다(GitHub zip에도 포함).

---

## 0. 한눈에 보기

**무엇인가.** STL(single-transistor latch)의 결정론·확률 모델을 웹에서 돌려 보는 시뮬레이터다.
- 대상 소자: floating-body FDSOI n-MOSFET. UI 표기는 "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm".
- 소유자: 박준형(Junhyoung Park, 석사과정), KAIST 전기및전자공학부 NOBEL 연구실(지도교수 최양규, Prof. Yang-Kyu Choi).
- 기능:
  - 소자 시뮬레이터: I–V branch와 fold(V_LU, V_LD), V_G 곡선, 전하 균형, 확률 MC(분포, CDF, hazard, 설계 지도)
  - 회로 시뮬레이터: STL을 MNA 과도해석 소자로 쓰는 LTspice형 회로도 편집기와 빠른 벤치
  - 검증 탭: 원본 모델의 기준 수치 재현
  - 물리 탭: 코드 수준 수식 18개 주제와 파라미터 가이드

**현재 상태 (2026-09-25 확인)**
1. 저장소 `JunhyoungPark-NOBEL/Single-Transistor-Latch-simulator-SOI-`, 브랜치 `claude/stl-simulator-web-j0yy9i`, 기준 커밋 `16dd9ad` + 인수인계 문서 커밋. 작업 트리는 origin과 같다(커밋되지 않은 변경 없음).
2. PR #1(→ `main`)은 열려 있고 병합되지 않았다(mergeable_state clean). 커밋 45개, 파일 362개, +74,042/−1. `main`(`2068d95`)에는 예전 Streamlit 앱 커밋 6개만 있다.
3. 구성: `engine/`(원본 numba 모델, 수정 금지) → `server/`(FastAPI + spawn 프로세스 풀) → `web/`(Vite 8 + React 19 + TypeScript strict + Plotly + KaTeX + Zustand, 한국어/영어).
4. 오늘 실행해 확인한 것:
   - `python3 engine/stl_api.py`(2.9 s)와 실행 중인 백엔드의 `/api/folds`가 검증 수치를 재현한다: fold 3.7037/2.5979 V, FPT 노드 3.6442 V / 8.03 mV, 동적 MC 3.6344 V / 119.0 mV.
   - `npm run typecheck`(`tsc -b --noEmit`, 약 7.5 s) 통과, `npm test`(vitest) 24개 파일 448개 테스트 통과(약 4 s).
5. 목록만 확인하고 실행하지 않은 것: pytest 327개(slow 9개), Playwright 72개(7개 파일). Docker 빌드, `npm run build`, 전체 e2e도 이번에는 실행하지 않았다.
6. UI 기본 레이아웃은 "간단히"(답 막대, 대표 그래프 1개, 탭 카드 1개)다. `?view=all`을 붙이면 "모두 보기"가 된다.
7. 공개 링크 **https://claude.ai/artifact/FkxbAC39Pfe3fMvEPffC49** 는 **비밀번호로 잠긴 정적 스냅샷**이다.
   - 파일 79개 중 56개(앱 번들, CSS, 스냅샷 전부)를 AES-256-GCM으로 암호화했다. 나머지는 비밀번호 카드와 KaTeX 폰트다(§8, §9).
   - 서버 없이 미리 녹화한 결과만 보여 준다.
   - 비밀번호는 소유자가 따로 준다.
   - claude.ai에 다시 게시하는 작업은 Claude만 할 수 있다.
8. **GitHub 저장소가 현재 PUBLIC이다**(GitHub API: `"private": false`). 공개 상태에서는 잠금 페이지도, 서버 로그인도 모델을 보호하지 못한다.
9. `web/snapshot/`(녹화본, 약 25 MB)과 `web/dist-artifact/`(잠긴 빌드, 약 26 MB)는 gitignore 대상이다. 지금은 Claude 컨테이너에만 있고, 새 clone에는 없다.
10. CI(`.github/`)와 Makefile이 없다. 테스트와 빌드는 모두 손으로 돌린다. 실서버 배포 흔적도 없다(미확인).

---

## 1. 소유자 요구사항과 원칙

### 1.1 명시적 요구사항 (요청 순서)
출처는 `docs/WEB_CONTRACT.md`의 'User requirements'와 'Phase 2 addendum (owner requests, 2026-09-24)', 그리고 커밋 기록이다.

1. 결정론 + 확률 STL 모델을 다루는 쓰기 쉬운 웹 UI.
2. 전역 **Deterministic | Stochastic** 전환. 커밋 `e7a4632` 이후에는 Device와 Circuit 탭에만 보인다.
3. 파라미터는 그룹 카드로 묶고 접을 수 있게 한다. 고급 그룹은 "고급 설정" 아래에 접어 둔다.
4. 물리는 코드와 정확히 같은 수식으로 쓴다.
   - "상세 (Details)" 버튼은 같은 화면에 작은 창을 띄운다. 모달이 아니고, 끌어 옮기고 크기를 바꿀 수 있다(약 560×620 px, Esc로 닫음).
   - 수식은 80%로 줄이거나 스크롤한다. 잘리면 안 된다.
5. 원본 인수인계 계획(`engine/docs/00_START_HERE_websim_KO.md`)을 따른다.
   - 소자 시뮬레이터, 회로 시뮬레이터, 검증 수치를 모두 갖춘다.
   - **미해결 문제는 강제 답이 아니라 선택지로 남긴다.**
6. 로고는 biristor 기호다.
   - 원 안의 NPN: 컬렉터가 위, 이미터 화살표가 아래, 베이스는 왼쪽에 열린(floating) 채로 둔다.
   - 청록→남색→보라 그라데이션 배지에 흰 선. 크기 40 px 이상에서만 C/B 글자를 표시한다.
   - 파일: `web/public/favicon.svg`, `web/src/components/Logo.tsx`.
7. 크레딧: "NOBEL Lab · Prof. Yang-Kyu Choi · School of Electrical Engineering, KAIST · Developed by Junhyoung Park".
   - KO: "KAIST 전기및전자공학부 · NOBEL 연구실 (지도교수 최양규) · 개발 박준형".
   - 칩을 누르면 About 카드가 열린다. 카드 문구는 "미발표 모델 — 연구용으로만 사용하세요." / "Unpublished model — for research use only."(`web/src/i18n/strings.brand.ts`의 `brand.about.scope`).
   - **e-mail과 URL은 넣지 않고, NOBEL의 뜻을 풀어 쓰지 않는다.**
8. 한국어와 영어 모두 자연스럽게 쓴다.
9. **"paper / 논문 / paper parameters / Fig. 3(b)"라는 말을 UI에 절대 쓰지 않는다**(모델이 미발표 상태).
   - 소자는 "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm"로 설명한다.
   - 내부 preset id `paper`/`photo`/`custom`은 그대로 둔다. 사용자에게 보이는 이름은 **기준 보정 / 광조사 보정 / 사용자 정의**(Reference calibration / Illumination calibration / Custom)다.
   - 공정은 FDSOI만 사용할 수 있다. PDSOI와 Bulk는 "준비 중 / coming later"로 표시한다.
10. LTspice형 회로 편집기.
    - 소자 라이브러리(localStorage)에 저장한 소자를 회로에 배치한다.
    - 소자: R, C, 자유 V/I 전원(DC/PULSE/PWL/SINE).
    - 사용자가 `.tran` 값을 정한다: 종료 시간, 저장 시작, 최대/최소 시간 간격, BE/TRAP, reltol.
    - 시간 커서에서 노드 전압과 단자 전류를 보여 준다. 부품을 클릭하면 probe가 걸린다.
    - 정확도, 실행 시간, 페이지 무게를 균형 있게 맞춘다.
11. 확률 결과 옆에 통계(describe, KS, 히스토그램, ECDF, StatsTable)를 둔다. 회로에서도 확률 소자를 쓸 수 있다(STL별 `local_state`, override 스위치).
12. 축 이름: 단일 축은 "이름 기호 (단위)", 겹친 subplot의 y축은 "기호 (단위)". `axis.*` 문자열을 쓴다.
13. 전류 구동 STL은 발진해야 한다.
    - `tran.initial` auto|op|zero로 해결했다.
    - 예: 1 nA, 1 pF, V_G −2 V, 15 ms → 스파이크 10개, 860 Hz.
14. p-bit.
    - 구성: 드레인에 주기 펄스, 소스 저항 R_S를 접지로, V(R_S)에 사용자 V_ref 비교기.
    - 결과는 무작위로 발화한다. 예: 3.69 V 펄스 20개, R_S 100 kΩ, V_ref 0.1 V → P(fire) ≈ 0.5, lag-1 ≈ 0.
15. 그래프 수를 줄인 간결한 UI.
    - 기본 "간단히": 답 막대(V_LU, V_LD, 창), 대표 그래프 1개, 탭 카드 1개.
    - "모두 보기" 또는 `?view=all`: 예전 전체 격자.
16. 파라미터 가이드를 **먼저** 보여 준다.
    - 직관 한 문장: 바디 = 정공 양동이, GIDL·빛·충돌 이온화 = 수도꼭지, 재결합·확산 = 새는 구멍.
    - 값을 올렸을 때 V_LU와 V_LD가 어떻게 바뀌는지 3줄.
    - 사이드바 인라인, ⓘ 팝오버, Details 창의 "한눈에"에 먼저 나오고, 물리 탭도 가이드부터 연다.
17. 링크는 비밀번호로 보호한다: 암호화된 정적 아티팩트와 선택형 서버 로그인 게이트.

### 1.2 문구 규칙 (DECISIONS `[physics-lang]`, `[brand]`, `web/README.md` 'Wording')
- **한국어**
  - UI 라벨은 명사구로 쓴다.
  - UI 문장은 합니다체, 물리 설명 본문은 한다체로 쓴다.
  - 단어 단위로 줄을 바꾼다(`:root[lang=ko]`에 `word-break: keep-all`).
- **영어**
  - sentence case, 미국식 철자(center, ionization, color).
  - 물리 본문에는 serial comma를 쓰지 않는다.
- **라틴 문자로 남기는 용어:** branch, fold, hazard, latch-up/latch-down(한국어 문장에서는 래치업/래치다운). hazard는 처음 한 번 "단위 시간당 탈출률"로 풀어 쓴다.
- **β:** 확산비(diffusion ratio)가 아니라 **주입 비율(injection ratio)**, 즉 빠져나간 정공 하나당 주입되는 전자 수라고 부른다.
- **축 제목:** `axis.*` 키를 쓴다.
- **BRAND 문자열:** e-mail·URL 금지, NOBEL을 풀어 쓰지 않는다.
- **README.md의 'About Me':** 연락처가 들어 있다. UI로 옮기지 않는다.

### 1.3 용어집
| English | 한국어 | 비고 |
|---|---|---|
| latch-up / latch-down | 래치업 / 래치다운 | |
| latch-up voltage V_LU / latch-down voltage V_LD | 래치업 전압 / 래치다운 전압 | V_LU는 --hrs(파랑), V_LD는 --lrs(빨강) 색 |
| hysteresis window ΔV | 히스테리시스 창 ΔV | V_LU − V_LD |
| latch window | 래치 창 | fold 쌍이 존재하는 V_G 범위 |
| body | 바디 | |
| branch, fold, hazard | branch, fold, hazard | 라틴 문자 유지 |
| censored | 중도절단 | |
| calibrated lookup engine | 보정 조회표 | `calibrated_lookup` |
| local state | 국소 상태 | frozen/evolving/none = 고정/진화/없음 |
| carrier noise | 캐리어 잡음 | |
| first passage | 첫 통과 | FPT |
| reference record / illumination records | 기준 측정 기록 / 광조사 측정 기록 | |
| reference calibration / illumination calibration / custom | 기준 보정 / 광조사 보정 / 사용자 정의 | preset id `paper` / `photo` / `custom` |
| base model | 기본 모델 | 확장 파라미터가 모두 0 |
| load line | 부하선 | |
| noise band | 잡음 대역 | |
| look-ahead | 선행 감지 | |
| tier | 계층 | |
| acquisition trend | 획득 추세 | |
| injection ratio β | 주입 비율 β | |

---

## 2. 저장소 구조

```
.
├── engine/                 원본 모델 패키지(그대로 복사, 수정 금지, 추적 파일 66개)
│   ├── stl_api.py          엔진 진입점 + 스모크 테스트(python3 engine/stl_api.py). export_tables.py는 조회표 생성용
│   ├── photo_extension/    광조사 확장 평균 모델(photo_mean.py), 배선(setup_photo.py), hazard(photo_fpt.py)
│   │                       photo_nodes/ = FPT 노드 캐시(gitignore, 새 clone에는 없어서 첫 계산 때 다시 만듦)
│   ├── model/janus_calibration_20260920/  원본 모델 코드와 보정 데이터(하위 폴더에 gate_mean, gate_fpt, gate_dynamic_compare, compound_fpt, conditional_table, *.npz, *.json)
│   ├── model/MODEL_PARAMETERS.json, model/stl_stochastic_research/  원본 파라미터 요약과 연구 코드
│   ├── data/               측정 데이터(raw_VLU.npy 400 cycle × 8 조건 등), 설계 지도, tables/*.npz
│   ├── docs/               00_START_HERE_websim_KO.md, MODEL_SPEC.md, VALIDATION.md, CIRCUIT_ELEMENT_DESIGN.md
│   └── requirements.txt    엔진 단독 의존성
├── server/                 FastAPI 백엔드(API 프로세스는 numba를 import하지 않음)
│   ├── main.py             라우트, 미들웨어, web/dist 서빙
│   ├── jobs.py             프로세스 풀, 진행/취소, 중복 제거, 결과 캐시, 입장 제한
│   ├── payloads.py         payload 검증·클램프(순수 Python)
│   ├── params.py           프리셋, build_p() → 26개 p 벡터(보정값의 단일 출처)
│   ├── engine_bridge.py    엔진 import 유일 지점(워커 전용)
│   ├── auth.py             선택형 비밀번호 게이트
│   ├── progress.py, jsonutil.py
│   ├── compute/            deterministic, stochastic, stoch_core, stoch_mc, validation, data, circuit/
│   │   └── circuit/        element, mna, sim, netlist, benches, stochastic, runner, custom, oscillator, validate
│   ├── tests/              pytest(conftest, test_api, test_auth, test_deterministic, test_stochastic, test_circuit, test_circuit_custom)
│   └── requirements.txt    서버 런타임 + 테스트 의존성(이것을 사용)
├── web/                    프런트엔드(Vite + React + TS)
│   ├── src/                App.tsx, main.tsx, api/, state/, components/, sidebar/, device/, circuit/, schematic/, devices/,
│   │                       stats/, validation/, physics/, params/, content/{params,physics}/, i18n/, plots/, styles/,
│   │                       types/, utils/
│   ├── e2e/                Playwright spec 7개 + screenshots/(기본), screenshots/all/(?view=all), 둘 다 커밋돼 있음
│   ├── scripts/            record-snapshot, build-artifact, artifact-common, lock-build, lock-crypto, lock/, verify-artifact, verify-lock (.mjs)
│   ├── snapshot/           [생성물, gitignore] 녹화된 결과
│   └── dist-artifact/      [생성물, gitignore] 정적(잠긴) 빌드
├── scripts/                dev.sh, warmup.py, param_sensitivity.py
├── docs/                   WEB_CONTRACT, API, RUNNING, DEPLOY, CIRCUIT_SIMULATOR, DECISIONS, HANDOFF_GPT(이 문서), GPT_PROMPT
├── Dockerfile, render.yaml 배포
└── app.py, config.toml, requirements.txt   예전 Streamlit 앱(건드리지 않음, Docker 이미지에서 제외)
```

주의할 점:
- 루트 `.gitignore`는 Python 템플릿이라 `lib/`와 `*.spec`을 무시한다. 소스 폴더 이름을 `lib`로 짓지 않는다. `*.spec.ts`는 영향이 없다.
- `.gitignore`는 `web/dist`, `web/dist-artifact`, `web/snapshot`, `web/test-results`, `web/playwright-report`, `server/.cache/`, `engine/photo_extension/photo_nodes/`, 모든 `__pycache__`도 무시한다.
- 린터·포매터 설정(ESLint, Prettier, ruff, pyproject)과 `AGENTS.md`/`CLAUDE.md`는 저장소에 없다. 품질 검사는 타입 검사와 테스트가 전부다.

---

## 3. 개발 환경과 실행

### 3.1 필요한 것
| 항목 | 버전 | 비고 |
|---|---|---|
| Python | 3.11 | 개발 환경 3.11.15, Docker `python:3.11-slim` |
| Python 패키지 | `server/requirements.txt` | numpy≥1.26, scipy≥1.11, numba≥0.59, fastapi≥0.110, uvicorn[standard]≥0.29, orjson≥3.9, pytest≥8, httpx≥0.27 |
| Node / npm | Node 22 | 개발 환경 v22.22.2 / npm 10.9.7, Docker `node:22-slim`, 루트에는 package.json 없음 |
| Playwright | `@playwright/test` 1.56.1(고정) | Chromium만 사용 |

- 개발 환경에 설치된 버전: numpy 2.4.6, scipy 1.17.1, numba 0.67.0, fastapi 0.141.1, uvicorn 0.53.0, orjson 3.12.0, pytest 9.1.1, httpx 0.28.1.
- **numpy는 2.0 이상을 써야 한다.** `server/compute/circuit/oscillator.py:167-168`이 NumPy 2.0에 새로 생긴 `np.trapezoid`를 쓰는데, requirements는 `>=1.26`을 허용한다. 이 불일치는 §10에서 수정 후보로 다룬다.
- 루트 `requirements.txt`는 예전 Streamlit 앱용이다. 쓰지 않는다.

### 3.2 설치
```bash
# 저장소 루트에서
pip install -r server/requirements.txt
cd web && npm ci && cd ..
# Claude 컨테이너 밖(예: Codex 샌드박스)에서는 한 번 필요
cd web && npx playwright install chromium && cd ..
#   리눅스에 시스템 라이브러리가 없어 Chromium이 뜨지 않으면(root 필요): npx playwright install --with-deps chromium
# numba 커널 미리 컴파일 + 캐시 채우기(권장)
python3 scripts/warmup.py            # --quick | --validate
```

첫 실행과 캐시:
- numba 첫 컴파일은 엔진 20–60 s, 회로 커널 약 30–40 s 걸린다(문서 기준).
- JIT 캐시는 `engine/**/__pycache__`, `server/compute/circuit/__pycache__`에 생긴다(환경 변수 `NUMBA_CACHE_DIR`가 있으면 그곳). gitignore 대상이라 새 clone에서는 다시 컴파일한다.
- 런타임에 쓰기 가능해야 하는 폴더: `engine/**/__pycache__`, `engine/photo_extension/photo_nodes/`, `server/.cache/results/`, `server/.cache/stochastic/`.
- `warmup.py` 옵션: 인자 없음 = 컴파일 + 결정론 kind·FPT 노드·MC·회로(벤치와 custom)를 한 번씩 실행, `--quick` = import와 branch 한 번만, `--validate` = 여기에 FAST 검증 9개를 더해 결과를 출력.
- `warmup.py`는 V_LU fold가 3.7037 V에서 1e-3 V 넘게 벗어나거나 `--validate` 검사가 실패하면 exit 1로 끝난다. 나머지 단계는 실패해도 `FAILED`만 찍고 계속한다.

원래 Claude 컨테이너와의 차이: 그곳에서는 `:8000`에 백엔드가 이미 떠 있었고, Chromium이 `/opt/pw-browsers`에 있어서 "playwright install 금지" 규칙이 있었다. GPT 환경에는 둘 다 없다.

### 3.3 실행
```bash
# 엔진 스모크 테스트(따뜻한 캐시에서 약 2.9 s)
python3 engine/stl_api.py

# 백엔드만(저장소 루트에서 실행, --workers는 절대 쓰지 않음, 병렬도는 STL_WORKERS로)
uvicorn server.main:app --port 8000
curl -s http://127.0.0.1:8000/api/health            # ok:true, workers, access_gate
curl -s 'http://127.0.0.1:8000/api/folds?vg=-2&wait=30'   # V_LU 3.70369, V_LD 2.59787

# 개발 모드: API :8000(server/ 자동 리로드, STL_WORKERS=2) + Vite :5173(/api 프록시)
scripts/dev.sh            # = all
scripts/dev.sh api        # 백엔드만
scripts/dev.sh web        # 프런트만(API_PORT의 백엔드를 가정)
# 다른 포트의 백엔드에 붙이기
cd web && STL_API=http://127.0.0.1:8011 npm run dev

# 운영(도커 없이): 빌드 → 워밍업 → 서버(web/dist를 / 에서 서빙)
cd web && npm ci && npm run build && cd ..
python3 scripts/warmup.py
STL_WORKERS=3 uvicorn server.main:app --host 0.0.0.0 --port 8000

# Docker(문서 기준, 이번 점검에서는 미실행)
docker build -t stl-websim . && docker run --rm -p 8000:8000 -e STL_WORKERS=2 -v stl-cache:/app/server/.cache stl-websim
```
- 포트: 백엔드 8000, Vite 개발 5173, e2e 5174(`E2E_PORT`).
- `scripts/dev.sh`의 환경 변수: `API_PORT`(8000), `WEB_PORT`(5173), `STL_WORKERS`(2), `PYTHON`(python3). zip으로 받아 실행 권한이 빠졌으면 `bash scripts/dev.sh`로 돌린다.
- `web/dist`가 없으면 백엔드는 `/`에서 UI 대신 안내 페이지를 보여 준다. UI는 Vite(:5173)로 보거나 `npm run build`를 먼저 한다.
- 워커 하나가 메모리를 약 150–250 MB 쓴다. 512 MB 플랜에서는 `STL_WORKERS=1`로 둔다.

### 3.4 백엔드 없이 보기 (데모·스냅샷)
- `http://127.0.0.1:5173/?mock=1`(또는 `mock=true`)은 강제 데모 모드다.
- `mock`이 없을 때의 선택 순서:
  1. `/api/health`에 성공하면 live.
  2. 실패하면 `./snapshot/index.json`을 읽는다(snapshot 모드).
  3. 그것도 없으면 offline이 되고 데모 데이터를 쓴다.
- 데모 fold 값은 기준 보정 V_G −2 V에서 3.70 / 2.60 V다(`web/src/api/mock.ts`).
- `?view=all` 또는 `?view=simple`로 레이아웃을 강제한다. 선택값은 localStorage `stl-websim:layout`에 저장된다.

### 3.5 환경 변수
| 변수 | 기본값 | 의미 |
|---|---|---|
| `STL_WORKERS` | CPU − 1(cgroup 반영, 최소 1) | 계산 프로세스 수 |
| `STL_CACHE_DIR`, `STL_STOCH_CACHE_DIR` | `server/.cache/...` | 결과 캐시, 확률 노드 캐시 위치 |
| `STL_DISK_CACHE_MB` / `STL_MEM_CACHE_MB` / `STL_JOB_RESULTS_MB` / `STL_NODE_CACHE_MB` | 1024 / 256 / 64 / 1024 | 캐시 크기 |
| `STL_MAX_PENDING` / `STL_MAX_PENDING_PER_CLIENT` | 64 / 16 | 대기열 한도(넘으면 429) |
| `STL_ABANDON_S` | 600 | 아무도 폴링하지 않는 job을 취소하는 시간(0이면 끔) |
| `STL_MAX_BODY_KB` | 256 | 요청 본문 한도(넘으면 413) |
| `STL_PREWARM`, `STL_MP_CONTEXT` | 1, spawn | 워커 예열(0이면 끔), multiprocessing 방식 |
| `STL_CORS_ORIGINS` | `localhost`·`127.0.0.1`의 5173, 4173 | 허용 origin(쉼표 구분) |
| `STL_WEB_DIST` | `web/dist` | 서버가 `/`에서 서빙할 빌드 폴더 |
| `PORT` | 8000 | Docker `CMD`가 쓰는 포트(Render, HF가 설정) |
| `NUMBA_CACHE_DIR` | 없음(소스 옆 `__pycache__`) | numba 캐시 위치. 소스 폴더에 쓸 수 없을 때 지정 |
| `PYTHON` (`scripts/dev.sh`) | python3 | dev.sh가 쓸 파이썬 실행 파일 |
| `STL_ACCESS_PASSWORD` | 없음 | 값이 있으면 로그인 게이트가 켜짐 |
| `STL_REQUIRE_PASSWORD` | – | 1/true/yes/on이면 비밀번호가 없을 때 모든 요청에 503 |
| `STL_SESSION_SECRET` | 프로세스마다 무작위 | 세션 서명 키(없으면 재시작 때 세션이 끊김) |
| `STL_TRUST_PROXY` | 0 | 프록시 뒤에서만 1 또는 N(X-Forwarded-For hop) |
| `FORWARDED_ALLOW_IPS` | `*`(Dockerfile) | 직접 노출하는 컨테이너는 127.0.0.1 |
| `STL_API` (web) | http://127.0.0.1:8000 | Vite 프록시 대상, 스냅샷 녹화기의 백엔드 주소 |
| `E2E_PORT` (web) | 5174 | Playwright용 Vite 포트 |
| `STL_ARTIFACT_PASSWORD` (web) | – | 잠긴 빌드와 검증용 비밀번호(환경 변수로만 받음) |

---

## 4. 아키텍처

### 4.1 흐름
```
engine/ (numba, 수정 금지)
   │  server/engine_bridge.py  ← 유일한 import 지점, 워커 프로세스에서만
   ▼
server/compute/*  계산 종류(kind): run(payload, progress) -> dict
   │  server/jobs.py  spawn 프로세스 풀, 진행/취소, 중복 제거, 메모리 LRU + gzip 디스크 캐시
   ▼
server/main.py  FastAPI: /api/compute/{kind}, /api/jobs/{id}, GET 별칭, /api/data/*, web/dist 서빙
   ▼
web/src/api/client.ts (live) | snapshot.ts (정적 녹화본) | mock.ts (데모)
   ▼
web/src/state/runner.ts → Zustand store → Panel / FocusLayout / MoreCard
```

### 4.2 계산 함수 규약 (WEB_CONTRACT §1)
- 시그니처는 `run(payload: dict, progress) -> dict`다(`server/progress.py`).
  - `progress(fraction 0..1, message)`는 취소되면 `JobCancelled`를 던진다.
  - 계산 중 약 0.5 s마다 한 번은 부른다(`server/progress.py` 머리 주석).
- kind 목록(`server/compute/__init__.py` KINDS): `branches, charge_balance, vg_curve, hazard, sweep_mc, vg_curve_stochastic, circuit, validation`. 추가 kind `folds`는 `jobs.EXTRA_KINDS`에 있다.
- 모든 결과에 `runtime_s`(float)와 `warnings`(list[str])를 넣는다. `payloads.normalize`의 클램프 경고는 앞에 붙는다.
- 잘못된 입력에는 `ValueError('사람이 읽을 메시지')`를 던진다.
  - 제출 전에 걸리면 HTTP 422가 된다.
  - 실행 중에 걸리면 job status가 `error`가 되고 `error` 문자열이 붙는다.
- numpy 배열을 그대로 돌려줘도 된다. NaN과 ±inf는 `null`로 직렬화된다.
- **API 프로세스는 numba와 engine을 import하면 안 된다.**
  - `test_api.py::test_api_process_never_imports_numba`가 이를 강제한다.
  - `data.py`, `params.py`, `payloads.py`, `jsonutil.py`는 순수 Python/numpy로 유지한다.

### 4.3 payload 블록 (WEB_CONTRACT §1 'Shared payload blocks')
모든 필드는 선택이다. `params.resolve_device`가 preset 값과 deep merge하며, `None`은 기본값을 덮어쓰지 않는다.

- **device**
  - `preset`: paper | photo | custom
  - `vg` (V)
  - `light`: `{mode iph|power, iph_pA, power_mW, responsivity_pA_per_mW 0.75}`
  - `calib`: `{beta, tau_bulk_s, tau_junction_s, r_contact_ohm, l_gidl_nm, t_access_nm, na_access_cm3, l_access_nm, tau_ratio, phi_gidl0_V, phi_emitter0_V, channel_ii_scale}` → p[0..10], p[12]
  - `ext`: `{dibl, gamma, kappa, seed_ip_pA, seed_S, dj, dm, aloc, isat_pA, dloc, loc_carriers 0|1|2, kappaF}` → p[14..25]
  - `state`: `{delta_phi_G0_V, delta_phi_E0_V}`, 각각 p[9], p[10]에 더해진다
  - `numerics`: `{grid 601}`, 201..2001로 클램프
- **sweep**: `{vd_max_V (≤ 8), rate_V_per_s, dv_V}`
- **stochastic**:
  - `n_cycles` (≤ 2000), `seed`, `carrier_noise`, `ld_carrier_noise`
  - `local_state`: `{mode none|frozen|evolving, action gidl|local_avalanche|junction|multiplication, sigma, tau_s, sigma_E_V, tau_E_s, acquisition_trend}`
  - `engine`: auto|general|calibrated_lookup
  - `n_traces` (≤ 50), `fold_nodes` (≤ 61), `hazard_nodes` (≤ 9)
- 그 밖의 상한(`payloads.CAPS`, 넘으면 클램프하고 경고): vg_curve 점 61개, 회로 `max_steps` 2e6, `n_runs` 200.
- 구조 제한(`payloads.check_tree`):
  - 깊이 12
  - 값 5000개(custom 회로는 40000개)
  - 문자열 1000자 이하
  - 숫자는 유한값만
- 프런트는 모든 요청에 device/sweep/stochastic 블록 전체와 `device.preset`을 보낸다.

### 4.4 p 벡터 (26개 float, `params.build_p`)
| idx | 의미 | idx | 의미 |
|---|---|---|---|
| 0 | β 주입 비율(loss ∝ 1/β) | 13 | I_PH (A) |
| 1 | τ_bulk | 14 | DIBL |
| 2 | τ_junction | 15 | γ 바디 결합 |
| 3 | R_contact | 16 | κ |
| 4 | l_GIDL (nm) | 17 | 고 V_D seed I_p (A) |
| 5 | t_access (nm) | 18 | seed 기울기 S (V/dec) |
| 6 | N_A,access | 19 | junction offset dj |
| 7 | L_access (nm) | 20 | log(M−1) scale dm |
| 8 | τ_p/τ_n | 21 | 국소 avalanche 세기 |
| 9 | φ_GIDL offset (V) | 22 | 국소 경로 포화 (A) |
| 10 | φ_emitter offset (V) | 23 | 국소 log 요동 |
| 11 | V_G | 24 | 국소 캐리어 정의 |
| 12 | channel-II scale | 25 | κF |

- 확장 파라미터를 모두 중립값으로 두면 `setup_photo.BASE`, 곧 기본 모델이 된다. 이때 `gate_mean`과 1e-12 이내로 같다.

### 4.5 확률 엔진 (`sweep_mc`)
- **`calibrated_lookup`**: 원본 `gate_dynamic_compare.simulate`를 감싸고, 그 난수열을 그대로 재생한다.
  - `auto`는 아래 조건을 **모두** 만족할 때만 이 엔진을 고른다.
    - `is_paper_reference(device)`: 값이 기준 보정과 같음(V_G −2 V, 어두움, ext 중립, state 중심 0). preset 이름은 보지 않는다.
    - `action == gidl`
    - `vd_max == 4.0`
    - `carrier_noise`가 켜짐
    - ΔV가 10 mV/k(`calibrated_dv_ok`)
  - `engine: "calibrated_lookup"`을 명시했는데 기준 보정·gidl·0→4 V·ΔV 조건을 어기면 `ValueError`(→ 422)가 난다. `carrier_noise`는 auto 선택에서만 본다(`stochastic._choose_engine`).
- **`general`**: `stoch_mc.simulate_general`.
  - 삼각 sweep 탈출 MC.
  - 상태는 OU 또는 고정이고, hazard 하위 간격은 2 mV 이하.
  - `FoldTable`(PCHIP)과 `HazardField`를 쓴다.
- **hazard**: `stoch_core`의 compound FPT를 전체 p로 계산한다.
  - 노드 디스크 캐시: `server/.cache/stochastic/{fold,hazard}/<sha256>.json`.
  - `CACHE_VERSION = 'stoch-3'`. hazard나 fold 알고리즘을 바꾸면 이 값을 올린다.

### 4.6 HTTP API (WEB_CONTRACT §3, docs/API.md)
- **실행과 폴링**
  - `POST /api/compute/{kind}?wait=2.0`(wait ≤ 60 s)는 JobStatus를 돌려준다: `{job_id, kind, status queued|running|done|error|cancelled, progress, message, result?, error?, cached, elapsed_s}`.
  - `GET /api/jobs/{id}?wait=`로 폴링하고, `DELETE /api/jobs/{id}`로 취소한다. `GET /api/jobs`는 요청한 클라이언트의 job만 보여 준다.
- **정보와 데이터**: `GET /api/health`, `/api/meta`(프리셋, 상수, kind 목록, caps), `/api/data/measured`, `/api/data/design_map`. FastAPI 문서 `/docs`, `/redoc`, `/openapi.json`도 켜져 있다.
- **GET/POST 별칭**
  - 경로: `/api/branches`, `/api/folds`, `/api/hazard`, `/api/sweeps`(= sweep_mc), `/api/vg_curve`. `/api/design_map`은 GET만 있고 `/api/data/design_map`과 같다.
  - 쿼리: `preset, vg, iph_pA | power_mW, grid, dg, de, vd_max, rate, dv, n, seed, vg_min, vg_max, wait`.
- **오류**
  - 404: 알 수 없는 kind/job/path
  - 413: 본문 > 256 KiB
  - 422: 잘못된 입력
  - 429: 대기열 가득 참(클라이언트당 16, 전체 64), `Retry-After: 10` 포함
  - 500: 내부 정보 없는 JSON
  - 오류 본문은 항상 `{detail: string}`이다.
- **프런트의 job 처리**
  - `?wait=1.5`로 제출하고, 400 ms마다 폴링한다.
  - 같은 key로 새 실행이 들어오면 이전 실행을 취소한다.
  - 429를 받으면 `Retry-After`(0.2–60 s로 제한)만큼 기다렸다가 한 번 다시 시도한다.

### 4.7 결과 형식 (위치)
- 소자 kind의 결과 형식은 WEB_CONTRACT §2에 있다. 통계는 `{n, mean, sd(ddof=1), median, p05, p95, min, max, censored, lag1}`이다.
- 회로 결과는 WEB_CONTRACT §4(CircuitResult)와 §6.2, 그리고 `docs/CIRCUIT_SIMULATOR.md` §10, §12.5에 있다.
- WEB_CONTRACT §2에 빠진 키가 있다. `docs/API.md`가 일부를 다룬다.
  - validation: `level`, `summary`
  - branches: `grid`, `vd_max_V`, `sweep_dv_V`
  - hazard: `window_V`, `step_V`, `skipped`, `n_voltages`, `kernel_skipped`, `fold_atom`, `I_at_fold_A`, `dg`, `de`
  - sweep_mc: `n_cycles`, `seed`, `rate_V_per_s`, `vd_max_V`, `V_LU_continuous`, `state_axis` 등
  - vg_curve_stochastic: `beyond_sweep_weight`, `censored_weight`, `vd_max_V`, `window`
  - health: `app_version`, `engine_version`, `jobs`, `access_gate`
- 프런트 타입은 `web/src/api/types.ts`에, 필수 키 검사는 `web/src/api/guards.ts`(SPECS)에 있다.

### 4.8 결과 캐시 키
- 키는 `sha256(kind, 정규화된 payload, 클램프 경고, ENGINE_VERSION)`이다.
- `ENGINE_VERSION`에 들어가는 파일:
  - `server/compute/**/*.py`, `params.py`, `engine_bridge.py`, `jsonutil.py`, `jobs.py`
  - `engine/**`의 코드와 데이터(런타임 캐시 폴더 제외)
- 이 파일들을 고치면 캐시된 결과가 모두 무효가 된다.

### 4.9 회로 시뮬레이터 (`server/compute/circuit/`, 명세: `docs/CIRCUIT_SIMULATOR.md`)
- **소자 모델**: STL은 D, G, S 단자 3개를 가진 소자다.
  - 내부 Newton 미지수: u(소스-바디 quasi-Fermi 분리), r(드레인 접합 역바이어스).
  - 상태: 바디 전하 Q_B.
  - 소자 식은 `element.py`의 `stl_eval`에 있다. 엔진 `photo_mean.components`를 감싸고 u<0, r<0 확장을 더했다.
- **해석기**: `mna.py`의 numba MNA 커널.
  - 결정론 모드: BE(기본) 또는 TRAP. TRAP은 stiff한 스텝에서 BE로 돌아간다.
  - damped Newton과 적응 스텝을 쓴다.
- **확률 모드**: Eq. 2의 사건 증분을 명시적 tau-leap으로 적용한다.
  - Poisson 단위 사건, 현재 r의 pmf에서 뽑는 충돌 이온화 cluster, Poisson 손실.
  - tier: 1 사건 수준, 2 Gaussian, 3/4/5 drift.
  - 캐리어 잡음은 z<12 잡음 대역 안에서만 풀고, 4 τ_rel 앞을 미리 본다.
- **두 입구** (`runner.run_circuit`)
  - (a) 벤치 `load_line / pulse / pbit / coupled`: 기본값은 `benches.py`, 계약은 WEB_CONTRACT §4.
  - (b) `bench: "custom"` 넷리스트: `custom.py`, WEB_CONTRACT §6, CIRCUIT_SIMULATOR §12.
    - 소자: R, C, V, I(dc/pulse/pwl/sine), STL, CMP(행동 모델 비교기).
    - 추가 기능: ERC, 선형 DC 추정, 스트리밍 축약, `tran.initial` auto|op|zero.
    - 제한: 소자 40개, STL 8개, CMP 8개, 노드 30개, 사용자 PWL 2000점.
- **실행 가능성 사전 추정**
  - 거부: run당 `max_steps`의 2배 초과, 또는 요청 전체 4e7 스텝 초과.
  - 경고: `max_steps`의 0.5배 초과.
  - 고임피던스 셀(>100 MΩ + 드레인 C)은 `oscillator.py`의 준정적 부하선 보행으로 발진 여부와 주기를 예측한다.
- **래치 판정과 불변식**
  - 래치 상태는 전류 문턱이 아니라 바디 branch로 판정한다: u가 u_j에 닿으면 래치, u_i로 떨어지면 해제.
  - I_D 문턱(10 nA / 1 nA)은 사건 시각을 재는 데만 쓴다.
  - 벤치와 custom 템플릿은 비트 단위로 같아야 한다(테스트로 강제). 새 커널 기능은 벤치에서 꺼 두어야 한다.
  - numba 함수 시그니처(인자 dtype, 차원)를 바꾸면 요청마다 재컴파일된다.
- **부호 규약**(§6.1): `I(V1)`은 +에서 −로 흐르는 전류다. 전력을 내는 전원은 음수다(2 V / 1 kΩ → −2 mA).
  - `I(X1.d)=I_D`, `I(X1.s)=−I_D`, `I(X1.g)=0`
  - `X1.q_b = Q(t)−Q(0)`
  - `CMP1.bit`는 0 또는 1

### 4.10 프런트의 백엔드 모드와 스냅샷
- `BackendState = checking | online | offline | mock | snapshot`(타입은 `web/src/state/store.ts`, 전환 로직은 `web/src/state/runner.ts`).
- health는 20 s마다 폴링한다. 강제 mock(`?mock=1`)과 snapshot 모드에서는 하지 않고, offline(자동 데모)에서는 계속 폴링하다가 백엔드가 살아나면 다시 붙는다.
- URL hash는 `#tab=device|circuit|validation|physics&mode=deterministic|stochastic` 형식이다(앱 store가 관리).
- 스냅샷 조회(`web/src/api/snapshot.ts`)
  - 키: `{kind, payload}`의 정렬된 canonical JSON을 SHA-256으로 해시한 소문자 hex.
  - 정확히 맞는 키가 없으면:
    - `branches / charge_balance / vg_curve`만 가장 가까운 녹화점을 쓴다(V_G 0.25 V, 광 파워 0.8 mW 이내, 거리 = ΔV_G/0.1 + ΔP/1).
    - 나머지는 데모 데이터를 쓰고 UI에 표시한다.
  - payload 빌더(`utils/payload.ts`), 프리셋, 예제를 바꾸면 키가 달라져 조용히 데모로 떨어진다. **이런 변경 뒤에는 다시 녹화해야 한다.**
- 자동 실행
  - Device 탭의 결정론 모드에서만 돈다.
  - 파라미터가 바뀌면 700 ms debounce 뒤 다시 실행한다.
  - 확률 계산은 자동으로 실행하지 않는다.

### 4.11 회로도 편집기와 소자 라이브러리 (프런트, `web/src/schematic/`, `web/src/devices/`)
- `model.ts` 문서 모델과 핀 좌표, `edit.ts` 순수 편집 연산, `nets.ts` 넷 연결(union-find, 라벨 "0"/"gnd" = 접지), `store.ts` Zustand 상태(undo/redo, localStorage `stl-websim:schematic`), `persist.ts` 저장본 검증.
- `netlist.ts`: 회로도 → WEB_CONTRACT §6 요청(netlist, `.tran`, stochastic, detect, probes)과 읽기 전용 SPICE 텍스트. 순수 함수이고 `schematic.test.ts`가 검사한다.
- `erc.ts`: 제출 전 전기 규칙 검사(접지, 떠 있는 노드, 전압원 루프, 값 범위 등). `feasibility.ts`: 실행 비용 휴리스틱(최종 판단은 서버).
- `si.ts`: SPICE 숫자 표기("1k", "10u", "1meg"; SPICE처럼 "m"/"M"은 밀리, 단독 "F"는 펨토). `waves.ts`: DC/PULSE/PWL/SINE 파형.
- `templates.ts`: 예제 회로 5개 `load_line, pulse, pbit, coupled, oscillator`. 앞의 네 개는 `benches.py` 빠른 벤치와 같은 회로다. 템플릿이나 기본값을 바꾸면 스냅샷 키도 바뀐다.
- `run.ts`: ERC → 실행 가능성 확인 → 공통 job runner로 제출. 결과 슬롯은 `results["schematic"]`이다.
- `web/src/devices/`: 소자 라이브러리. 내장 FDSOI 소자(`/api/meta` 프리셋에서 파생, 읽기 전용)와 Device 탭에서 저장한 사용자 소자(localStorage `stl-websim:devices`)를 다루고, JSON 가져오기·내보내기를 검증한다(`library.ts`, `store.ts`, `library.test.ts`).

---

## 5. 물리 모델 요약과 기준 수치

### 5.1 모델 (engine/docs/MODEL_SPEC.md)
- 상태 변수는 u(소스-바디 분리)와 r(드레인 접합 역바이어스)다.
- Eq. 1 결정론 핵심: 바디 전하 보존 `dQ_B/dt = I_II + I_BTBT + I_GIDL + I_PH − I_REC − I_DIFF ≡ F(u, r; V_G, I_PH)`와 Kirchhoff 전류 법칙.
  - 단자 전압은 V_D = u + r + hole drop + (R_c + R_acc)·I_D다. MODEL_SPEC에는 R_c가 빠져 있다.
  - 정상 상태 F = 0의 locus가 접힌다: 저전류 해는 V_LU(첫 V_D 극대)까지, 고전류 해는 V_LD(마지막 극소)까지 존재한다.
  - 엔진 함수는 `components(u, r, p, …)`(19개 값)와 `FastModel.classify`다.
- Eq. 2 확률 부분: 단위 사건(Poisson), 충돌 이온화 cluster, 손실 사건. 결과 분포는 compound first-passage(FPT) hazard로 계산한다.
- 국소 상태: GIDL 위치 φ_G와 이미터 φ_E가 사이클마다 흔들린다(OU / frozen / evolving).
- 광조사 확장은 p[13..25]를 쓴다. I_PH = 0.75 pA/mW × P.

### 5.2 프리셋 (`server/params.py`, UI 이름은 §1.3)
| id | V_G | 빛 | sweep | 확률 설정 |
|---|---|---|---|---|
| `paper`(기준 보정) | −2 V | 어두움 | 0→4 V, 0.4 V/s, dv 2 mV | n 100, seed 2026092920, carrier_noise·ld_carrier_noise on, evolving gidl(σ 0.1534 V, τ 5 s, σ_E 0.437 mV, τ_E 1.62 s, acquisition_trend on), engine auto, n_traces 12, fold_nodes 25, hazard_nodes 5 |
| `photo`(광조사 보정) | −1.8 V | mode power, power_mW 0, γ 0.2794, δφ_G0 +0.0744 V | 0→5 V, 1200 V/s | frozen, σ 0.2154 V, σ_E 0, n 400, seed 20260922 |
| `custom`(사용자 정의) | `paper`의 deep copy | | | |

- 측정 광조사 조건(`MEASURED_PHOTO_CONDITIONS`)은 V_G −1.8 V와 −1.1 V, 각각 0 / 1.15 / 2.55 / 3.51 mW다. `raw_VLU.npy`의 0..7열에 해당한다.
- 보정값은 엔진 JSON에서 읽는다.
  - N_A 2.2958e17 cm⁻³
  - β 7.1665, τ_bulk 0.927 µs, τ_j 5.36 ns, R_c 1 Ω
  - l_GIDL 28.75 nm, t_acc 3.31 nm, N_A,acc 1.0e17, L_acc 70 nm, τ_ratio 117.4
  - p9 0.272 mV, p10 0.0591 mV
  - σ_φG 0.15339 V, σ_φE 0.437 mV, τ_G 5.0 s, τ_E 1.6206 s
  - photo: δφ_G0 +0.07443 V, σ 0.21536 V, γ 0.27942

### 5.3 기준 수치 (허용오차와 검사 위치)
검사 위치의 약어:
- VAL = `engine/docs/VALIDATION.md` + `server/compute/validation.py`(FAST 9개, FULL = FAST + 5개)
- T-det = `server/tests/test_deterministic.py`
- T-sto = `server/tests/test_stochastic.py`
- WU = `scripts/warmup.py`

| 항목 | 값 | 허용오차 | 검사 위치 |
|---|---|---|---|
| fold, 기준, V_G −2 V 어두움 | V_LU 3.7037 V, V_LD 2.5979 V (live 3.703689 / 2.597867) | ±1 mV | VAL, T-det, WU |
| fold, 기준, V_G −1.8 V 어두움 | 3.8644 / 2.5979 V | ±1 mV | VAL, T-det |
| fold, 기준 + 빛, V_G −1.8 V, I_PH 2.63 pA | 3.2913 / 2.5962 V (VALIDATION.md 2.596) | ±1 mV | VAL, T-det |
| FPT 노드, −2 V, 0.4 V/s, 상태 중심 | 평균 3.6442 V, SD 8.03 mV | VAL 평균 ±3 mV, SD ±1.5 mV / T-sto 3.6442 ±0.3 mV, 8.03 ±0.1 mV | VAL, T-sto |
| 동적 MC 100 sweep, seed 2026092920(보정 조회표, 10 mV 판독 중점) | V_LU 3.6344 V / 119.0 mV, V_LD 2.6999 V / 21.4 mV | VAL 3.63 ±10 mV, 120 ±10 mV, 2.70 ±10 mV, 20 ±3 mV / T-sto 3.634 ±2 mV, 119.0 ±1.5 mV, 2.700 ±2 mV, 21.4 ±1.0 mV | VAL, T-sto |
| 기준 기록 10 × 100 sweep(seed 2026093000..09) | σ_LU 125.8 mV, σ_LD 19.6 mV(측정 123.1 / 19.5) | ±1 / ±0.5 mV | VAL FULL(이번에 재실행 안 함) |
| 캐리어 잡음만, V_G −2 V | 재유도 구현 II 5.1, BTBT 3.0, REC 4.2, DIFF 1.9, 전체 8.0 mV(VALIDATION 목표 4.6/2.7/4.3/1.8/7.8) | ±1 mV(slow 테스트는 전체 8.03 ±0.1) | VAL FULL |
| 래치 창(기준, 고정 상태) | −3.906 / −0.810 V(VALIDATION −3.90 / −0.815) | ±10 mV(T-det는 위쪽 끝 −0.810 ±2 mV) | VAL, T-det |
| 확률 V_G 곡선(기준, frozen, 0–6 V, V_G −1.6…−0.9, 15점) | σ_LU 최대 129.8 mV @ −1.25 V, 평균 최대 4.354 V @ −1.10 V | σ ±10 mV, 평균 ±20 mV, 위치 ±0.1 V | VAL FULL |
| 빛 환산 | 1.15 / 2.55 / 3.51 mW → 0.86 / 1.91 / 2.63 pA | ±0.005 pA | VAL |
| 광조사 기록(−1.8 V 어두움, 1200 V/s, 400 cycle) | 평균 3.806 V, SD 173.2 mV | ±20 mV / ±20 mV | VAL FULL(`sweep_mc_photo_dark`) |
| 확장 항등성 | 확장을 모두 0으로 둔 photo_mean = gate_mean | ≤ 1e-12 V | VAL |
| 회로 부하선 | 래치업 ≈ 3.7037 V, 래치다운 ≈ 2.5979 V | ±30 mV | VAL FULL |
| general 엔진 V_LD | ld_carrier_noise off 2.598 V(fold), on 2.6996–2.7007 V(측정 2.700) | – | DECISIONS |

- 기타 참고값:
  - fold 민감도(−2 V): dV_LU/dφ_G ≈ −0.80 V/V. dV_LD/dφ_E는 fold 기준 −39.8 V/V, 조회표 기준 −41.2 V/V.
  - −2 V branch(`engine/stl_api.py` 출력): HRS 719점(0…3.704 V), LRS 428점(2.598…12.67 V), 4 V에서 I_D 2.65e-5 A. 서버 `branches` 결과는 격자가 달라 점 수가 다르고(HRS 421점), LRS는 vd_max + 1 V에서 자른다.
- 시간(문서 기준): FPT 노드 하나 3–5 s, 100 cycle MC 1 s 미만.

### 5.4 회로 기준 수치 (`docs/CIRCUIT_SIMULATOR.md` §7, §13, §14)
- **V1 결정론 부하선**(R_s 100 Ω, C_d 1 fF)
  - 0.4 V/s: V_LU 3.70393 V(+0.24 mV 지연), V_LD 2.597855 V.
  - 지연은 40 V/s에서 +5.0 mV, 1200 V/s에서 +44.0 mV다.
  - 테스트는 |V_LU − 3.7037| < 1 mV를 확인한다.
- **V2 확률 vs FPT**(광조사 조건)
  - 120 V/s: 3.2049 V / 35.5 mV vs FPT 3.2077 / 34.5 mV, KS 0.066.
  - 1200 V/s: fold를 넘는 비율 0.580 vs FPT 0.529.
- **V6 기준 소자**: 0.4 V/s 사건 수준에서 3.6453 V / 7.6 mV vs FPT 3.6442 / 8.0 mV.
- **발진기**(1 nA → 1 pF, V_G −2 V, 15 ms)
  - `initial_used` zero, 첫 래치업 3.744 ms, 주기 1.1628 ms(860 Hz).
  - 테스트 기준값: `OSC_T_REF` 1.1637 ms, 허용 ±1%.
  - 창 밖 조건: 5 pA는 HRS 3.6365 V에 머물고, 30 nA는 LRS 2.6037 V에 머문다.
- **p-bit**(V_G −2 V, 200 µs 평탄, 20 µs 에지, 1 ms 주기, R_S 100 kΩ, V_ref 0.1 V)
  - 확률 모드 P(fire): 3.66 V 0.075, 3.68 V 0.34, 3.69 V 0.495, 3.70 V 0.75. |lag-1| ≤ 0.08.
  - 결정론 모드: 3.70 V까지 발화하지 않고, 3.72 V부터 매번 발화한다.
  - 테스트는 0.15 < p_fire < 0.85를 확인한다.
- **성능**: 벤치 1셀 약 35–40 µs/step. custom은 약 3 µs/step에 STL당 60–90 µs가 더해진다. 전체 검증 V1–V6은 약 12 min, `--quick`은 약 3 min(문서 기준).

### 5.5 엔진 특이점 (엔진은 그대로 두고 server/에서 우회)
1. `p[24] loc_carriers = 2`는 1과 똑같이 동작한다.
   - 원인: `engine/photo_extension/photo_mean.py` 152행 `bulk = p[24] > 0.5`가 먼저 걸린다.
   - UI에서는 2를 비활성으로 표시한다.
2. `MODEL.classify`가 가짜 fold를 낼 수 있다.
   - locus 틈을 가로질러 fold 포물선을 맞추는 경우와, V_LD ≥ V_LU 또는 V_LU > 8 V인 쌍을 내는 경우다.
   - **항상 `deterministic.classify_checked`를 쓴다**(`LOCUS_GAP_U` 0.05 V, `V_FOLD_CAP` 8 V).
3. `S.state`는 r 구간 끝에서 NaN이면 ValueError를 던진다. 대신 `deterministic.state_row`를 쓴다.
4. `compound_fpt.backward`의 avalanche kernel 범위는 r ∈ [0.7, 5.0] V다.
   - 약 5.1 V보다 높은 fold는 `kernel_skipped`로 처리하고, 탈출을 fold에 둔다.
5. `gate_dynamic_compare.simulate`는 ΔV = 10 mV/k와 0→4 V에서만 동작한다.
6. V_D ≈ 0 근처의 전류는 약 1e-58 A다. 로그 축에는 하한이 필요하다(프런트는 1e-17 A).
7. import 부작용이 있다.
   - sys.path를 바꾸고, `conditional_table`을 변형하고, `photo_nodes/`를 만든다.
   - 그래서 엔진은 `server.engine_bridge`로만 import하고, `engine/`은 쓰기 가능해야 한다.
8. 엔진 문서끼리 맞지 않는 곳이 있다.
   - `stl_api.py` 출력의 'SD 6.8 mV'와 '2.5959'는 낡은 값이다. 실제는 8.03 mV와 2.5962다.
   - MODEL_SPEC의 V_D 식에는 R_c가 빠져 있다.
   - 광전자에 p[12]가 곱해지지 않는다.
   - 전하 정의가 세 가지다.
   - N_A가 MODEL_SPEC에서는 2.30e17, `refit_3.json`에서는 2.2958e17이다.
   - 모두 `docs/DECISIONS.md`에 기록돼 있다.

---

## 6. 프런트엔드 규칙

### 6.1 레이아웃 (`web/src/state/layout.ts`, `components/MoreCard.tsx`)
- `LayoutMode = "simple" | "all"`, 기본값은 `"simple"`이다.
  - 초기값 결정 순서: `?view=` → localStorage `stl-websim:layout` → `"simple"`.
  - `<html data-layout>`에 반영된다.
- 레이아웃은 별도 store인 `useLayout`에 있다. **앱 store(`useStore`)로 옮기지 않는다.**
  - URL hash `#tab=…&mode=…`는 앱 store가 관리한다. 레이아웃은 query(`?view=`)에만 둔다.
- `FocusLayout` props: `testId, scope, hero?, tabs: MoreTab[], defaultTab, side?, allOrder?`.
  - simple 모드: hero와 MoreCard(활성 탭 하나만 렌더링)를 보여 준다.
  - all 모드: 전체 격자를 보여 준다.
- MoreCard
  - 활성 탭을 localStorage `stl-websim:more:<scope>`에 저장한다. 코드에서 탭을 바꿀 때는 `selectMoreTab(scope, id)`를 쓴다.
  - 760 px 이하에서는 탭 줄이 `<select>`로 바뀐다.
- scope 목록:

| scope | hero | 탭 | 기본 탭 |
|---|---|---|---|
| `device-det` | IvPanel | vg, components, charge-balance | vg |
| `device-sto` | McIvPanel | dist, hazard, cycles, vg-sto, design-map | dist |
| `circuit`(빠른 벤치) | WaveformPanel | trajectory, c-dist, c-sweeps, c-events | trajectory |
| `schematic`(`SCH_MORE_SCOPE`, testId `sch-panels`) | WaveViewer | `sch-cmp-<이름>`, sch-dist, sch-traj | 첫 CMP 탭 → sch-dist → sch-traj |
| `validation` | 없음 | val-iv, val-photo, val-vg | val-iv |

- `LayoutToggle`은 KpiStrip, CircuitTab, ValidationTab에 있다.
- **`?view=all`은 계속 동작해야 한다.** 예전 e2e spec이 이것에 의존한다.

### 6.2 Panel과 새 패널 추가
- 모든 결과는 `<Panel id title desc topic entry hasData currentKey plot toolbar primary menu foot onPlotClick …/>`으로 감싼다.
  - 로딩, 빈 상태, 오류, stale("변경됨"), 데모 배지, CSV/PNG 내보내기를 Panel이 처리한다.
- 큰 빈 상태 Run 버튼은 `primary` 패널에만 있다.
- 눈에 보이는 toolbar 컨트롤은 하나까지만 둔다. 나머지는 ⋯ `menu`에 넣는다.
- 새 패널을 추가하는 순서:
  1. `api/types.ts`에 타입을 넣고, `api/guards.ts` SPECS에 필수 키를 넣는다. 새 Kind라면 `api/mock.ts`에 데모 생성기도 만든다.
  2. `useEntry<T>(key)`로 결과를 읽고, `usePalette()`로 색을 가져온다(`device/common.tsx`).
  3. `runKey(key, kind, payload)`로 실행한다. payload 빌더는 `utils/payload.ts`의 순수 함수다. Run 버튼에 묶으려면 `runDeterministic` 또는 `runStochastic`에 추가한다.
  4. 해당 탭의 `FocusLayout`에 MoreTab으로 등록하고, 필요하면 `allOrder`에도 넣는다.

### 6.3 i18n (`web/src/i18n/`)
- `const t = useT(); t("key", {var})`는 STRINGS 키에 쓴다. StrKey로 타입 검사가 된다.
  - STRINGS에는 SCHEMATIC(`schematic.*`), STATS(`stats.*`), BRAND(`brand.*`, `axis.*`)가 합쳐져 있다.
- **UX(`strings.ux.ts`), DEV(`strings.device.ts`), GUIDE(`strings.guide.ts`)는 STRINGS에 합쳐져 있지 않다.**
  - `t.l(UX["…"])`로 쓰고, 값 삽입은 `fill(t.l(UX["…"]), {n})`으로 한다.
  - `t("more.label")`처럼 쓰면 타입 검사에서 실패한다.
- 모든 사전은 `satisfies Record<string, L10n>`으로 끝난다. 자리표시자는 `{name}`이다. 기본 언어는 ko다.
- `GroupDef.title`과 `desc`는 STRINGS 키여야 한다.

### 6.4 파라미터 가이드
- 내용은 `web/src/content/params/guide.ts`(`PARAM_GUIDE`)에 있다: `{key, intuitive, effect: [V_LU 줄, V_LD 줄, 이유 줄], caveat?, basis?}`.
  - effect 줄은 파라미터를 **올렸을 때**를 기준으로 쓴다.
- 숫자와 화살표는 `sensitivity.json`과 맞아야 한다(`guide.test.ts`가 강제).
  - |Δ| < 3 SE이면 "→"로 표시한다.
  - 크기는 6% 이내로 맞아야 한다.
  - UI의 모든 파라미터 키에 항목이 있어야 한다.
- 가이드 숫자를 바꾸면 `python3 scripts/param_sensitivity.py`로 `sensitivity.json`을 다시 만든다.
  - `--det-only`는 약 1 min, 전체는 5–10 min 걸린다.
- UI는 가이드를 수정하지 않고 `params/guideUi.ts`로 읽기만 한다.
- 가이드를 보여 주는 컴포넌트:
  - `GuideInline`: 사이드바. testid `guide-inline-<key>`.
  - `GuidePopover`(ⓘ): 가리키면 tooltip, 클릭하면 고정 dialog. 760 px 이하에서는 bottom sheet.
  - `GuideBlock`: Details 창의 "한눈에".
  - `ParamGuideList`: 물리 탭.
- `web/src/content/physics/*`(18개 주제)는 physics-content 소유다. `physics.test.ts`가 모든 수식을 KaTeX `throwOnError: true`로 렌더링해 본다.

### 6.5 테마와 디자인 토큰 (`web/src/styles/app.css`)
- 색은 `:root` 토큰으로 정의하고, 다크 모드는 `:root[data-theme="dark"]`에서 덮어쓴다.
- 강조색은 모드를 따른다.
  - 결정론: 청록, `--det` #0d9488 / #2dd4bf.
  - 확률: 보라, `--sto` #7c3aed / #a78bfa, `:root[data-mode="stochastic"]`.
- 의미 색:
  - `--hrs`(파랑): HRS와 V_LU.
  - `--lrs`(빨강): LRS와 V_LD.
  - `--unstable`, `--meas`, `--ok/--warn/--err`, 그리고 각각의 `-soft`.
- 간격은 8 px 격자(`--s1..--s6` = 4/8/12/16/24/32 px), 반경은 `--r1..3` = 6/10/14 px다.
- 크롬 높이: header 52 px + context strip 32 px. 사이드바 폭은 348 px다.
- breakpoint:
  - 760 px: 폰
  - 1100 px: 사이드바가 drawer로
  - 1280 px: FocusLayout 좌우 배치
- **플롯 색은 `plots/theme.ts`의 `palette(theme)` 또는 `usePalette()`로만 가져온다.** hex를 하드코딩하면 다크 모드가 깨진다.
- 전폭 배너는 쓰지 않는다. 백엔드 상태는 ContextStrip의 상태 pill로 알린다.

### 6.6 testid와 저장 상태
- **testid**
  - Panel: `panel-<id>`, `empty-run-<id>`, `panel-caption-<id>`, `panel-menu-<id>`
  - MoreCard: `more-<scope>`, `more-tab-<id>`, `more-select-<scope>`
  - LayoutToggle: `layout-toggle`, `layout-simple`, `layout-all`
  - 가이드: `guide-inline-<key>`
- **localStorage**
  - 앱 상태: `stl-websim:v1`(schema v=2, 필드별 검증)
  - 레이아웃과 탭: `:layout`, `:more:<scope>`
  - 사이드바: `:groups`, `:adv-open`, `:field-adv`
  - 기타: `:hint-dismissed`, `:circuit-view`, `:schematic`, `:devices`, `:record`(개발 전용)
  - 모든 접근은 try/catch로 감싼다.
- 잠긴 페이지에서는 `stl-websim:*` 값을 암호화해 저장한다.

### 6.7 깨뜨리면 안 되는 것
- `?view=all`, hash `#tab=…&mode=…`, testid, 스냅샷 키(payload 빌더, 프리셋, 예제). 스냅샷 키가 바뀌면 다시 녹화해야 한다.
- `web/src/main.tsx` 맨 위(import 바로 다음)의 `globalThis.__STL_BOOTED__ = true`. 잠금 로더가 부팅 성공을 이것으로 판단한다.
- 문구 규칙: "paper/논문" 금지, e-mail 금지.

---

## 7. 테스트

| 명령 (저장소 루트 또는 `web/`) | 대상 | 개수 | 소요 시간 | 백엔드(:8000) 필요 |
|---|---|---|---|---|
| `python3 -m pytest server/tests -q -m "not slow"` | 서버 빠른 묶음 | 318 / 327 | 미측정 | 아니오(테스트가 자체 풀을 띄움) |
| `python3 -m pytest server/tests -q` | 서버 전체 | 327(slow 9) | RUNNING.md 기준 약 1 min(2 workers), 미측정 | 아니오 |
| `python3 -m pytest server/tests/test_circuit.py server/tests/test_circuit_custom.py -m "not slow" -q` | 회로 | 103 / 108 | 문서 기준 test_circuit 차가울 때 ~1 min, custom 따뜻할 때 30–40 s | 아니오 |
| `python3 -m pytest server/tests/test_auth.py -q` | 로그인 게이트 | 122 | 미측정 | 아니오 |
| `python3 -m pytest server/tests --collect-only -q` | 목록 | – | 1.3 s | 아니오 |
| `python3 engine/stl_api.py` | 엔진 스모크 | – | 약 2.9 s(따뜻할 때, 2026-09-25) | 아니오 |
| `python3 scripts/warmup.py --validate` | 컴파일 + FAST 검증 | 9 checks | 미측정 | 아니오 |
| `python3 -m server.compute.circuit.validate --quick` (`--out <json>`, `--only v1,v6`) | 회로 V1–V6 | – | ~3 min(전체 ~12 min), 문서 기준 | 아니오 |
| `cd web && npm run typecheck` | `tsc -b --noEmit`(`src/`와 vite·playwright 설정만. `e2e/*.spec.ts`와 `scripts/*.mjs`는 검사 대상이 아님) | – | 약 7.5 s, 통과(2026-09-25) | 아니오 |
| `cd web && npm test` (= `vitest run`) | 단위 테스트(기본 node 환경, `MoreCard.test.tsx`와 `layout.test.ts`만 jsdom) | 24 파일, 448 통과(`vitest list`에는 199개 항목) | 약 4 s(2026-09-25) | 아니오 |
| `cd web && npx vitest run src/api/lock.test.ts src/api/snapshot.test.ts` | 잠금·스냅샷 | 17 + 13(목록 기준) | 미측정 | 아니오 |
| `cd web && npm run e2e` (= `playwright test`) | E2E(Chromium) | 72 / 7 파일(app 21, ux-device 11, ux-shell 11, schematic 10, ux-guide 9, live 5, stats 5) | 이번에 미실행 | live.spec과 schematic의 live 그룹만(없으면 skip) |
| `cd web && npx playwright test e2e/ux-guide.spec.ts` | spec 하나만 | – | – | spec에 따라 |
| `cd web && npm run verify:artifact` | 정적/잠긴 빌드 검사 | – | 미실행 | 아니오(`/api`는 404) |

pytest:
- **저장소 루트에서 실행한다.** `test_api_process_never_imports_numba`는 현재 cwd에서 `python -c 'import server.main'`을 실행한다. `server/` 안에서 돌리면 실패한다.
- conftest는 `STL_WORKERS=2`와 임시 `STL_CACHE_DIR`을 쓴다. 확률 노드 캐시(`server/.cache/stochastic`)는 공유된다. 그래서 첫 실행이 느리다: numba 컴파일에 더해, 첫 stochastic photo 테스트가 fold 표와 hazard 노드를 약 20 s 동안 만든다.
- `slow` marker는 `server/tests/conftest.py`에 등록돼 있다. pytest.ini와 pyproject는 없다.

Playwright:
- 설정은 Chromium, worker 1개, 1440×900, 테스트당 60 s다. Vite를 `E2E_PORT`(5174)로 띄운다.
- `reuseExistingServer: true`라서 5174에 떠 있는 옛 Vite를 조용히 재사용한다. 새로 띄우려면 `E2E_PORT=5180`처럼 포트를 바꾼다.
- 예전 spec(app, schematic, stats)은 `/?mock=1&view=all#tab=…&mode=…`로, live.spec과 schematic의 live 그룹은 `/?view=all#tab=…&mode=…`(실제 백엔드)로 연다. 스크린샷은 `e2e/screenshots/all/`에 쓴다.
- ux-* spec은 기본 simple 레이아웃을 다루고, 스크린샷을 `e2e/screenshots/`에 쓴다. 두 폴더 모두 커밋돼 있어서 다시 돌리면 덮어쓴다.
- live.spec은 `http://127.0.0.1:8000/api/health`가 ok일 때만 돈다. 백엔드가 없어서 skip되는 것은 실패가 아니다.

---

## 8. 공개 링크 갱신 절차

현재 링크: https://claude.ai/artifact/FkxbAC39Pfe3fMvEPffC49
- 잠긴 빌드이고, 파일은 79개다: 암호화 .wasm 56개, KaTeX woff2 19개, `lock.js`, `lock.css`, `lock.json`, 진입 조각.
- claude.ai에 게시하는 일은 **Claude의 Artifact 도구로만** 할 수 있다. GPT가 할 수 있는 것은 아래 8.4의 (b)와 (c)다.

### 8.1 준비
- 백엔드를 띄운다(`scripts/dev.sh api` 또는 `uvicorn server.main:app --host 127.0.0.1 --port 8000`). 캐시를 미리 데우면 좋다.
- `cd web && npm ci`를 실행한다. 잠긴 빌드는 KaTeX 폰트가 `node_modules/katex/dist/fonts`와 바이트 단위로 같은지 확인한다.
- Chromium을 설치한다(`npx playwright install chromium`, Claude 컨테이너 밖에서만).

### 8.2 스냅샷 녹화
```bash
cd web && npm run snapshot:record     # 백엔드 주소: STL_API=… 또는 -- --api URL
# 옵션: --flows <목록>, --langs ko,en, --plain, --with-val-vg, --step-timeout 900(초), --headed, --out <dir>
```
- 백엔드(:8000)가 떠 있어야 하고 Chromium이 필요하다. 차가운 캐시에서 30–40 min 걸린다. 결과는 `web/snapshot/`(index.json + gzip JSON)을 덮어쓴다.
- 한도는 40 MB, 파일 150개다. 브라우저와 node의 키가 다르면 실패한다.
- 녹화 대상(기본 flow 6개):
  - 언어별(KO/EN): device-det, device-sto, circuit-schematic, circuit-benches, validation
  - 한 번만(device-grid): V_G 격자 −3.8…−0.9 V, 0.1 V 간격(두 프리셋), 광조사 프리셋은 파워 0 / 1.15 / 2.55 / 3.51 mW에서도
- 현재 녹화본: 2026-09-25T05:52:23Z, 결과 165개, 파일 54개, 약 25 MB. 그 뒤 커밋은 녹화기와 잠금 파일 형식(.wasm)만 바꿨지만, HEAD `16dd9ad`의 모든 payload를 덮는지 실제로 확인하지는 않았다.

### 8.3 잠긴 빌드와 검증
```bash
cd web
read -rs STL_ARTIFACT_PASSWORD; export STL_ARTIFACT_PASSWORD   # 쉘 기록에 남지 않게
npm run build:artifact -- --lock        # (= npm run build:artifact:lock) 옵션: --out <dir>, --iterations N(≥600000), --no-snapshot, --snapshot <녹화 폴더>
npm run verify:artifact                 # 같은 환경 변수 필요. 옵션: -- --full, --shots <dir>
node scripts/verify-artifact.mjs --serve --port 5211   # 손으로 열어 보기(비밀번호 없이 서빙만)
```
- **비밀번호는 `STL_ARTIFACT_PASSWORD` 환경 변수로만 넘긴다.**
  - `--password` 플래그는 build와 verify 모두 거부한다.
  - 앞뒤 공백이나 제어 문자도 거부한다(`$(cat file)`의 줄바꿈 포함).
  - 파일, 로그, 채팅, 커밋에 쓰지 않는다.
- 빌드는 임시 폴더에서 진행한다. 자체 검사를 모두 통과해야 `web/dist-artifact/`를 교체한다.
  - 모든 파일 복호화, 잘못된 AAD로는 실패하는지, 누출 검사, 비밀번호 문자열 검사.
- verify는 다음을 확인한다.
  - 틀린 비밀번호로는 앱이 돌지 않는다.
  - 맞는 비밀번호로는 V_LU 3.704 / V_LD 2.598이 나온다.
  - remember-me, 만료되거나 손상된 키, `#lock`, 키보드만으로 잠금 해제, 390 px 라이트/다크, CSP 매트릭스.
- **`--lock` 없이 `npm run build:artifact`를 돌리면** 잠긴 빌드를 **평문 빌드로 덮어쓴다.** 평문 빌드는 공개하지 않는다.
- 다시 빌드하면 salt와 파일 이름이 바뀐다. 모든 사용자가 비밀번호를 다시 입력해야 한다.

### 8.4 게시
- **(a) claude.ai 아티팩트(Claude 전용)**
  - `web/dist-artifact/stl-simulator.html`을 진입점으로, `files.json`의 모든 항목을 contentType과 함께 올린다(`.wasm` 56개는 `application/wasm`, 나머지는 확장자로 추론). `files.json` 자체는 올리지 않는다.
  - 2026-09-25 확인: 게시본(79개 파일, 진입점은 호스트에 `index.html`로 저장됨)과 Claude 컨테이너의 `web/dist-artifact/`는 파일 이름이 같다.
  - 이전 버전에만 있던 경로는 모두 `null`로 지운다. 그러지 않으면 옛 평문 파일이 남는다.
  - 공유는 소유자가 claude.ai Share 메뉴에서 한다.
- **(b) 일반 HTTPS 정적 호스트(GPT 사용자용, 실제 호스트에서는 아직 시험하지 않음)**
  - 잠긴 `web/dist-artifact/` 폴더를 **이름과 경로를 바꾸지 않고** 통째로 올린다. AAD가 경로에 묶여 있어서 옮기거나 이름을 바꾸면 복호화에 실패한다.
  - 진입 파일은 doctype이 없는 조각이다. 폴더 루트에 감싸는 `index.html`을 만든다(verify-artifact의 `page()`와 같은 방식).
    ```bash
    cd web/dist-artifact
    { printf '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'; cat stl-simulator.html; printf '</body></html>'; } > index.html
    ```
  - HTTPS(Web Crypto secure context)와 DecompressionStream이 필요하다. .wasm의 MIME 타입은 상관없다(fetch만 한다).
  - 페이지는 폴더 루트에서 서빙해 `lock.json`, `assets/`, `snapshot/`이 상대 경로로 풀리게 한다.
  - 정적 페이지는 녹화된 결과만 보여 준다. 녹화되지 않은 요청에는 데모 데이터가 표시된다.
- **(c) 실시간 서버(Docker)와 로그인 게이트** (`docs/DEPLOY.md`)
  - Render: `render.yaml` blueprint를 쓴다.
    - `STL_ACCESS_PASSWORD`는 `sync:false`라서 Render가 값을 묻는다.
    - 이미 있는 서비스를 다시 sync하면 묻지 않는다. Environment 탭에서 직접 넣어야 한다.
  - HF Spaces: Docker, `app_port 8000`.
  - 연구실 서버 예:
    ```bash
    docker build -t stl-websim .
    docker run -d --restart unless-stopped -p 80:8000 -e STL_WORKERS=3 -e FORWARDED_ALLOW_IPS=127.0.0.1 \
      --env-file /etc/stl/secrets.env -e STL_REQUIRE_PASSWORD=1 -v stl-cache:/app/server/.cache --name stl stl-websim
    ```
    - `secrets.env`는 저장소 밖에 mode 600으로 둔다. `STL_ACCESS_PASSWORD`와 `STL_SESSION_SECRET`을 따옴표 없이 적는다.
    - 세션 키는 `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`로 만든다.
    - Caddy나 nginx 뒤에서는 `-p 127.0.0.1:8000:8000 -e STL_TRUST_PROXY=1`을 쓴다.
  - 링크를 공유하기 전 확인:
    ```bash
    curl -s -o /dev/null -w "%{http_code}\n" https://<host>/api/meta   # 401이어야 함
    curl -s https://<host>/api/health                                 # 200, "access_gate":"on"
    ```
  - Docker 이미지는 일반 `web/dist`(암호화 안 함)를 로그인 쿠키 뒤에서 서빙한다.

---

## 9. 보안 현황 (2026-09-25 기준)

- **저장소 공개 여부: PUBLIC.**
  - GitHub API 결과는 `"private": false, "visibility": "public"`이다.
  - `engine/`(추적 파일 66개)과 모든 소스를 누구나 받을 수 있다. 그래서 잠긴 페이지와 로그인 게이트가 사실상 아무것도 보호하지 못한다.
  - 해결: GitHub → Settings → General → Danger Zone → Change visibility → Private. Render와 HF는 private 저장소에서도 빌드할 수 있다.
  - **전환하기 전에** GPT/Codex가 GitHub 연결로 private 저장소를 읽을 수 있는지 확인한다(미확인). 읽지 못하면 zip으로 넘긴다.
- **잠긴 빌드에서 암호화한 것**: 앱 번들(단일 IIFE), CSS, 모든 스냅샷 파일.
  - 평문으로 남는 것: 비밀번호 카드(`lock.js`, `lock.css`, 진입 조각), `lock.json`(공개 키만), 기본 KaTeX 폰트.
  - 누출 검사(`LEAK_PATTERNS`): 평문 파일에 latch, GIDL, calib, avalanche, V_LU, FDSOI, KAIST/NOBEL, 사람 이름, 보정 숫자 지문이 있으면 빌드가 실패한다.
- **암호 방식**
  - 키: PBKDF2-HMAC-SHA256, 600,000회, 16바이트 salt.
  - 암호화: AES-256-GCM. 파일 형식은 `'STLENC1\0'` · IV 12 B · 암호문 · tag 16 B, 평문은 gzip.
  - AAD는 게시 경로다.
  - 이 형식은 `web/scripts/lock-crypto.mjs`, `web/scripts/lock/lock.js`, `web/src/api/snapshot.ts` 세 곳에서 똑같아야 한다.
- **비밀번호 강도가 곧 보호 강도다.**
  - 암호화된 파일은 누구나 받을 수 있어서 오프라인으로 추측할 수 있다.
  - 무작위 단어 4개 이상이나 무작위 16자 이상을 권한다. 연구실·학교 이름에 숫자를 붙인 형태는 피한다.
  - 페이지가 있다는 사실과 파일 크기는 숨겨지지 않는다.
- **브라우저 저장**
  - remember-me를 켜면 원시 키를 localStorage `stl-lock:<salt>`에 30일 저장한다. 끄면 sessionStorage에 1일 저장한다.
  - `#lock` 또는 `__STL_LOCK__.lock()`으로 저장된 키를 지운다.
- **서버 로그인 게이트** (`server/auth.py`, 기본은 꺼짐)
  - `STL_ACCESS_PASSWORD`가 비어 있지 않으면 켜진다.
  - 비밀번호 해시는 PBKDF2 200,000회, 세션은 HMAC 쿠키 `stl_session`이다. remember-me면 30일, 아니면 브라우저 세션(24 h 만료).
  - 잠금 없이 열리는 경로: `/login`, `/logout`, `/api/health`, `/favicon.*`.
  - 그 밖의 API는 401 `{"error":"login required"}`, 페이지는 `/login?next=`로 303 리다이렉트한다.
  - 무차별 대입 제한: 5회 실패하면 30 s 잠금, 이후 두 배씩 늘어 최대 15 min. 서버 전체 60 s당 실패 20회 + 유예 10회, 곧 분당 최대 약 30회 검사.
  - `STL_REQUIRE_PASSWORD=1`인데 비밀번호가 없으면 `/api/health`까지 모든 요청이 503을 받는다(fail closed).
  - `STL_TRUST_PROXY`는 모든 트래픽이 프록시를 거칠 때만 켠다. 직접 노출하면 `FORWARDED_ALLOW_IPS=127.0.0.1`로 X-Forwarded-For 위조를 막는다.
- **호스트별 주의**
  - 공개 HF Space는 Files 탭에서 소스가 보인다.
  - huggingface.co의 임베드 화면은 로그인 쿠키를 유지하지 못한다. `https://<user>-<space>.hf.space`로 직접 연다.
- **비밀 관리**: 비밀번호와 세션 키는 저장소, `render.yaml`, Dockerfile, README, 로그에 절대 넣지 않는다. 호스트의 secret으로만 설정한다.
- **생성물 폴더**: `web/snapshot/`에는 보정 파라미터가 평문으로 들어 있다. 공개 저장소에 커밋하지 않고, HF Space에 `web/`을 손으로 올릴 때도 빼 둔다. `.dockerignore`에도 빠져 있어 Docker 빌드 컨텍스트에는 들어가지만, 런타임 이미지에는 `web/dist`만 복사되므로 들어가지 않는다.

---

## 10. 알려진 한계와 남은 작업 (우선순위 순)

### P0 — 소유자 조치
1. **저장소를 private으로 전환한다**(§9). 먼저 GPT 도구가 private 저장소에 접근할 수 있는지 확인한다.
2. **강한 비밀번호로 잠긴 빌드를 다시 만든다**: build → verify → 게시(§8). 서버를 배포한다면 `STL_ACCESS_PASSWORD`와 `STL_SESSION_SECRET`을 호스트 secret으로 설정한다.
3. claude.ai 링크를 Share 메뉴로 공유한다. 비공개 아티팩트는 소유자만 열 수 있다.
4. PR #1을 병합한다(open, mergeable_state clean).
5. 선택: 실시간 백엔드를 배포한다(Render, HF, 연구실 서버). 현재 배포는 없는 것으로 보인다(미확인).
6. GPT가 정적 페이지를 다시 게시해야 한다면, 소유자가 gitignore된 `web/snapshot/`과 `web/dist-artifact/`를 따로 넘긴다. 이 폴더는 Claude가 따로 보낸 `stl-websim-snapshot.zip`(web/snapshot)과 `stl-websim-dist-artifact-locked.zip`(web/dist-artifact)에 들어 있다(없으면 Claude 세션이 끝나기 전에 받아 둔다). 넘기지 않으면 다시 녹화해야 한다(30–40 min). 이 폴더들은 공개 저장소에 커밋하지 않는다.
7. (완료) 인수인계 문서 두 개는 커밋되어 저장소 `docs/`에 있다.

### P1 — 작고 안전한 코드·문서 정리
1. numpy 하한 불일치: `server/requirements.txt`의 `numpy>=1.26`와 `server/compute/circuit/oscillator.py:167-168`의 `np.trapezoid`(NumPy 2.0+). 하한을 `numpy>=2.0`으로 올리거나 fallback을 둔다.
2. 프런트 reltol 하한: `web/src/schematic/erc.ts:187`은 `(0, 0.1]`을 받지만 서버는 `[1e-5, 0.1]`을 요구한다.
3. 낡은 주석과 수치 정리.
   - `server/compute/circuit/benches.py`
     - SOLVER_DEFAULTS 주석 "LTE(u) ≤ 1 mV" → 실제 30 µV.
     - noise_z_max 주석의 "V_LU + 0.25 V 상한" → 실제로는 상한이 없다.
     - 모듈 docstring의 P_sw 정의(I_D 문턱) → 실제로는 바디 branch.
   - `server/tests/test_circuit.py` 머리말의 옛 V2 수치(3.2091 / 34.6 mV, 0.552) → `docs/CIRCUIT_SIMULATOR.md` §7의 현재 값(3.2049 V / 35.5 mV, 0.580)으로.
   - `server/params.py` 6행 docstring의 `beta (diffusion ratio)` → injection ratio(주입 비율). `engine/docs/MODEL_SPEC.md`의 같은 표현은 엔진 문서라 고치지 않는다.
4. `web/src/types/plotly-dist.d.ts`에 쓰지 않는 `plotly.js-dist-min` 선언이 남아 있다.
5. `.dockerignore`에 `web/snapshot`, `web/dist-artifact`를 넣어 Docker 빌드 컨텍스트를 약 50 MB 줄인다(런타임 이미지에는 원래 들어가지 않음).
6. 문서 갱신.
   - `web/README.md`에 추가할 내용: 간단히/모두 보기 레이아웃, `?view=`, FocusLayout/MoreCard, 가이드 UI, strings.ux/device/guide.ts와 `fill()`, E2E_PORT, 스크린샷 폴더 구분, ux-* spec. 'Add a panel' 절차에 MoreTab 등록 단계와 새 Panel props를 넣고, 폴더 트리도 고친다.
   - `docs/WEB_CONTRACT.md`
     - §5: React 18 → 19, `plotly.js-dist-min` → `plotly.js-cartesian-dist-min`, 전체 격자 전용이라는 설명 → simple/all.
     - §2: 빠진 결과 키(§4.7).
     - §1 예시의 `ld_carrier_noise: false`: 두 프리셋 모두 true다.
   - `docs/DECISIONS.md`: 간단히 레이아웃 결정 항목이 없다(append-only로 추가).
   - `web/e2e/ux-shell.spec.ts` 4행이 존재하지 않는 'UX spec (§6)'을 인용한다.
   - sine 파형: td > 0이고 phase ≠ 0일 때 td 전 값이 코드에서는 `vo + va·sin(phase)`(`custom.py:228`)인데, 문서는 "vo"라고 한다.

### P2 — 기능과 모델
1. CI가 없다. GitHub Actions로 typecheck, vitest, pytest `-m "not slow"`를 도는 방안(numba 컴파일 시간 고려).
2. 스냅샷이 HEAD의 payload를 모두 덮는지 점검하고, 필요하면 다시 녹화한다.
3. 회로도 편집기에 `tran.initial`(auto|op|zero)과 CMP `width`를 노출한다. 지금 프런트는 둘 다 보내지 않아 서버 기본값이 쓰인다.
4. `p[24] = 2`(edge only)를 별도 선택지로 살리려면 `server/`에 wrapper가 필요하다. 엔진 152행은 고치지 않는다.
5. 모델 미해결 문제. **선택지로 유지하고 답을 강제하지 않는다.**
   - 광조사 소자의 −1.1 V 거동에 필요한 channel seed: p[17]/p[18](high_vd_seed: seed_ip_pA 1.33, seed_S 0.8) 또는 γ p[15](0.2794). 값은 확정되지 않았다.
   - 국소 상태가 작용하는 위치: GIDL p[9](보정됨) 또는 국소 avalanche p[21]–p[25](실험). junction p[19]와 multiplication p[20]은 경고를 내는 실험용 레버다.
   - sweep 사이의 잔류 바디 정공은 소자 MC에서 무시한다. 회로에서는 Q_B를 연속으로 적분한다.
6. 엔진 한계.
   - V_D ≳ 5.1 V에서는 hazard를 계산할 수 없어 탈출을 fold에 둔다. 캐리어 잡음이 과소평가된다.
   - `calibrated_lookup`은 기준 조건에서만 쓸 수 있다.
   - 캐리어 잡음 분해와 설계 지도 관계식은 원본 코드가 없어서 재유도했다.
7. 회로 한계(CIRCUIT_SIMULATOR §9, §12.8, §13.6).
   - 단자 전류는 준정적이다. dQ_B/dt 변위 전류를 stamp하지 않고, 게이트는 이상적이다. 그래서 fF 규모 발진은 정성적이다.
   - r<0, u<0 확장은 보정 범위 밖이다.
   - LRS 캐리어 잡음은 기본으로 꺼져 있다(켜면 약 20배 비용).
   - 1200 V/s에서 평균 V_LU가 FPT보다 20–30 mV 높다.
   - L, 제어 전원, 스위치가 없다. TRAP은 울릴 수 있다. 선행 감지는 지배 구동원 하나만 본다.
   - 발진 예측은 >100 MΩ 셀에만 적용한다. 느린 사건 수준 실행은 거부될 수 있다(광조사 0.4 V/s는 약 1.4e6 step).
8. PDSOI와 Bulk 모델은 없다(UI는 "준비 중"). 형상은 모델이 고정한다.
9. 기타.
   - KS p값은 사이클 상관 때문에 낙관적이다(툴팁으로 경고).
   - 히스토그램 bin 한도가 프런트 1–100, 서버 10–80으로 다르다.
   - 회로도 편집기의 실행 가능성 힌트는 휴리스틱이다. 서버 검사가 최종 판단이다.

---

## 11. 작업 규칙 (GPT용)

1. **`engine/`은 절대 수정하지 않는다.**
   - 엔진 버그는 `server/`에 wrapper로 우회한다.
   - `docs/DECISIONS.md`에 `[package]` 접두어를 붙인 bullet 하나로 기록한다. 이 파일은 append-only다.
2. API 프로세스에서는 numba와 engine을 import하지 않는다. 엔진은 `server.engine_bridge`로, 워커 안에서만 쓴다.
3. `server/params.py`가 보정값의 단일 출처다. `build_p` 시그니처는 바꾸지 않는다. 값은 SI 단위나 키에 적힌 단위(`iph_pA`, `l_gidl_nm`)로 저장한다.
4. 한국어와 영어 모두 자연스럽게 쓴다(§1.2–1.3).
   - "paper / 논문" 금지.
   - 소자는 "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm"로 표기한다.
   - UI에 e-mail이나 URL을 넣지 않는다.
   - 내부 id(`paper`/`photo`/`custom`, check id)는 그대로 둔다.
5. **비밀번호, 세션 키, 토큰을 커밋하거나 출력하지 않는다.** 잠긴 빌드의 비밀번호는 `STL_ARTIFACT_PASSWORD` 환경 변수로만 받는다.
6. 모델의 미해결 문제는 선택지로 남긴다. 강제로 답하지 않는다.
7. 커밋 전에 다음을 돌린다.
   - `cd web && npm run typecheck && npm test`
   - 관련 pytest(최소 `python3 -m pytest server/tests -q -m "not slow"`, 저장소 루트에서)
   - UI를 바꿨다면 `npm run e2e`(또는 관련 spec)
   - 돌리지 못한 검사는 "미실행"이라고 분명히 적는다. 결과를 지어내지 않는다.
8. `?view=all`, URL hash, testid, 스냅샷 키 규약이 계속 동작해야 한다. payload, 프리셋, 예제를 바꿨다면 다시 녹화해야 한다고 알린다.
9. 플롯 색은 `usePalette()`로 가져온다. 새 문자열은 적절한 사전(STRINGS 또는 UX/DEV/GUIDE)에 KO와 EN을 함께 넣는다.
10. 가이드 숫자를 바꾸면 `scripts/param_sensitivity.py`로 `sensitivity.json`을 다시 만든다.
11. `uvicorn --workers`는 쓰지 않는다. 폴더 이름을 `lib`로 짓지 않는다. 예전 `app.py`, `config.toml`, 루트 `requirements.txt`는 건드리지 않는다.
12. 커밋 메시지는 기존 형식을 따른다.
    - 제목: `영역: 요약`, 영어, sentence case, 마침표 없음. 예: `Circuit fixes: latch state from the body branch, …`, `Docs: encrypted artifact files are .wasm`.
    - 본문: 필요하면 이유를 적고 약 80자에서 줄을 바꾼다.
    - 기존 커밋 끝에 붙은 Claude 세션용 꼬리말(`Co-Authored-By: Claude …`, `Claude-Session: …`)은 따라 쓰지 않는다.
13. 브랜치와 PR
    - 현재 작업 브랜치는 `claude/stl-simulator-web-j0yy9i`(PR #1 → main)다.
    - 새 작업은 소유자와 정한 브랜치에서 한다(예: PR #1 병합 후 main에서 새 브랜치). 기본 브랜치에 직접 push하지 않는다.
    - push와 PR 생성은 소유자가 요청할 때만 한다.

---

## 12. 참고 문서 목록

| 문서 | 내용 |
|---|---|
| `docs/HANDOFF_GPT.md` | 이 문서(전체 요약과 인수인계) |
| `docs/GPT_PROMPT.md` | GPT에 붙여 넣을 시작 프롬프트 |
| `docs/WEB_CONTRACT.md` | **단일 기준 문서.** 소유자 요구사항 1–5, §0 소유권/배치, §1 계산 규약·payload·프리셋·한도, §2 결과 형식, §3 HTTP API, §4 회로, §5 프런트(일부 낡음), Phase 2 addendum, §6 custom 회로(6.1 부호, 6.2 응답, 6.3 구현 메모), §7 소자 라이브러리, §8 브랜딩·문구, §9 통계 |
| `docs/DECISIONS.md` | append-only 결정 기록(147개 bullet, `[package]` 접두어). 엔진 특이점, 우회, "그대로 둠" 메모 |
| `docs/API.md` | HTTP 규약, 오류 코드, JobStatus, 중복 제거와 캐시, GET 별칭 쿼리, 데이터 엔드포인트 키 |
| `docs/RUNNING.md` | 설치, 개발, 도커 없는 운영, Docker, 호스팅 메모, 환경 변수 표, 테스트 명령(KO/EN) |
| `docs/DEPLOY.md` | Render(render.yaml), HF Spaces, 연구실 서버, §4 비밀번호 보호와 공유 전 확인, 저장소 private 권고 |
| `docs/CIRCUIT_SIMULATOR.md` | 회로 엔진 명세(975행). §1 소자, §2 MNA/Newton, §3 BE/TRAP, §4 확률 tier·잡음 대역, §5 벤치, §6 요청 파라미터·실행 가능성, §7 검증 V1–V6, §8 성능, §9 한계, §10 결과 형식, §11 코드 지도, §12 custom 회로(§12.9 비교기), §13 전류 구동 발진기, §14 p-bit |
| `engine/docs/00_START_HERE_websim_KO.md` | 원본 인수인계(2026-09-24, 한국어): 목표, 패키지 지도, 권장 구조, 작업 순서, 미해결 문제 3개 |
| `engine/docs/MODEL_SPEC.md` | 원본 모델 명세: u, r, Eq. 1, p[0..25], Eq. 2, 회로 공식, sweep 프로토콜 |
| `engine/docs/VALIDATION.md` | 재현해야 할 기준 수치 |
| `engine/docs/CIRCUIT_ELEMENT_DESIGN.md` | STL 회로 소자 원본 설계와 제안 벤치 4개 |
| `README.md` | 최상위 개요(KO/EN), 빠른 시작, 예전 Streamlit 메모('About Me'의 연락처는 UI로 옮기지 않음) |
| `web/README.md` | 프런트 가이드: 개발/빌드/테스트, 정적 스냅샷, 잠긴 게시, UI 개요, 회로 편집기, 폴더 구조, 패널·물리 주제 추가법, 문구 규칙(일부 낡음, §10 P1) |
