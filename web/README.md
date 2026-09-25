# STL Simulator — web frontend

SOI 단일 트랜지스터 래치(STL)의 결정론 + 확률 모델을 위한 웹 UI (Vite + React 19 + TypeScript strict,
Plotly, KaTeX, Zustand). 백엔드(FastAPI, `server/`)의 계산 작업을 호출하고, 백엔드가 없으면 데모(mock)
데이터로 동작한다. 계약: `docs/WEB_CONTRACT.md`, API 세부: `docs/API.md`.

Web UI for the deterministic + stochastic STL (single-transistor latch, SOI n-MOSFET) model. It calls the
FastAPI compute jobs in `server/` and falls back to demo (mock) data when the backend is unreachable.
Contract: `docs/WEB_CONTRACT.md`; API details: `docs/API.md`.

---

## 한국어

### 개발
```bash
cd web
npm ci
npm run dev            # http://127.0.0.1:5173 — /api 는 :8000 (uvicorn server.main:app --port 8000) 으로 프록시
# 백엔드 없이: http://127.0.0.1:5173/?mock=1  (또는 백엔드가 응답하지 않으면 자동으로 데모 모드)
```
`scripts/dev.sh`(저장소 루트)는 백엔드와 Vite를 함께 띄운다.

### 빌드 · 검사
```bash
npm run typecheck      # tsc -b --noEmit
npm run build          # web/dist — 서버가 / 에서 제공
npx vitest run         # 단위 테스트 (포맷터, 페이로드, 단위 변환, 결과 형식 가드, RichText 파서, 물리 콘텐츠)
npx playwright test    # e2e (Chromium, headless). mock 모드 + 백엔드가 :8000에 있으면 live 테스트
```
e2e 스크린샷은 `e2e/screenshots/`에 저장된다. `playwright install`은 실행하지 말 것(브라우저는 `/opt/pw-browsers`).

### 정적 스냅샷 (서버 없이 공개하는 페이지)
백엔드 없이 올리는 정적 페이지(claude.ai 아티팩트)는 `/api`에 닿지 못하면 `./snapshot/index.json`을 찾아
**스냅샷 모드**로 동작합니다(`src/api/snapshot.ts`). 각 계산 요청은 `sha256(정렬된 JSON {kind, payload})` 키로
찾아 기록된 실제 모델 결과(`snapshot/<키>.json.gz`, DecompressionStream으로 해제, 없으면 `.json`)를 돌려주고,
스냅샷에 없는 조건은 데모 데이터로 계산하며 패널과 배너에 그렇게 표시합니다. 결정론 소자 결과(branch, 전하 균형,
V_G 곡선)는 Device 1의 V_G 격자(−3.8 … −0.9 V, 0.1 V 간격)에서도 기록되어, V_G만 바뀐 요청은 가장 가까운 격자점의
실제 결과를 “가장 가까운 미리 계산된 V_G = x V” 표시와 함께 보여 줍니다. CSVM은 기록된 기본 조건만 재생하고, 치수·V_BG를
바꾸거나 CSVM 설정을 바꾼 계산은 데모 데이터 대신 “계산 서버 연결 필요” 오류를 보여 줍니다.
`?mock=1`과 “백엔드 없음 + 스냅샷 없음”은 예전처럼 데모 모드입니다.
```bash
# 백엔드가 :8000에서 실행 중일 때 (다른 주소: STL_API=http://127.0.0.1:8011)
npm run snapshot:record && npm run build:artifact   # web/snapshot/ 기록 → web/dist-artifact/ (gitignore)
npm run verify:artifact -- --shots /tmp/shots     # 정적 호스트처럼 제공(/api 404)하고 Chromium으로 확인
```
기록기(`scripts/record-snapshot.mjs`)는 Vite 개발 서버 + Chromium으로 두 언어의 기본 흐름(소자 탭 결정론·확률의
VSCM과 기본 CSVM, 확률 V_G 곡선, 회로도 예제 전부 결정론·확률, 레퍼런스 탭)과 V_G 격자(한 번)를 실행하고 앱이 받은 결과를 그대로 저장합니다(개발 모드에서
`localStorage["stl-websim:record"]="1"`일 때만 켜지는 훅, 빌드에는 포함되지 않음). 모델·프리셋·예제·페이로드가
바뀌면 다시 기록해야 합니다(키가 달라지면 그 조건은 데모 데이터로 표시됨). `web/snapshot/`은 생성물이므로
커밋하지 않습니다. 게시: `dist-artifact/stl-simulator.html`(doctype/html/head/body 없는 조각) + `files.json`의 파일들.

### 비밀번호로 잠근 게시 (잠긴 빌드)
링크만 가진 사람이 내용을 볼 수 없도록 게시용 빌드 전체를 비밀번호로 암호화할 수 있습니다(링크 + 비밀번호).
```bash
STL_ARTIFACT_PASSWORD='…' npm run build:artifact -- --lock   # → web/dist-artifact/ (잠긴 빌드, --out으로 다른 폴더)
STL_ARTIFACT_PASSWORD='…' npm run verify:artifact           # 틀린/맞는 비밀번호, 기억하기, CSP 매트릭스를 Chromium으로 확인
```
- 비밀번호는 환경 변수로만 받습니다(명령줄 인자·파일·로그에 남기지 않음; `verify:artifact`도 `--password`를 거부).
  변수가 없거나 앞뒤 공백·줄바꿈 같은 제어 문자가 섞여 있으면 `--lock` 빌드를 거부합니다.
- 암호화: 앱 전체(IIFE 스크립트 하나, `assets/app-*.wasm`), 앱 스타일시트(`assets/style-*.wasm`), 스냅샷 파일
  전부(`snapshot/index.wasm` 포함)를 gzip 후 AES-256-GCM으로 암호화합니다. 키 = PBKDF2-SHA256(비밀번호, 무작위
  16바이트 salt, 600 000회), 파일마다 무작위 IV, 게시 경로가 AAD입니다(형식: `scripts/lock-crypto.mjs`). 평문으로
  남는 것은 비밀번호 카드(`stl-simulator.html`, `lock.js`, `lock.css` — 연구실·사람 이름 없음), 공개 파라미터
  (`lock.json`: salt·반복 횟수·파일 이름), KaTeX 원본 글꼴(원본과 바이트 단위로 같은지 확인)뿐입니다.
- 빌드는 임시 폴더에 쓰고 자체 검사(전 파일 복호화, 평문 파일의 모델 문자열·보정 수치·이름 검사, 모든 파일의
  비밀번호 검사)를 통과해야만 출력 폴더를 교체합니다. 실패하면 이전 빌드가 그대로 남고, 게시할 수 있는 결과물은
  생기지 않습니다.
- 페이지: 비밀번호 입력(0.5초 정도 키 계산) → 앱·스타일시트 복호화 → 실행(`blob:` 스크립트 → 인라인 스크립트 →
  `new Function` 순서로, 호스트의 CSP가 허용하는 방법). “이 브라우저에서 기억”을 켜면 파생된 키(비밀번호가 아님)를
  localStorage에 30일 동안, 끄면 sessionStorage(탭을 닫으면 사라짐)에 둡니다. 앱이 저장하는 설정(모델 파라미터
  포함)도 평문으로 남지 않고 암호화되어 localStorage에 저장됩니다. 링크 끝에 `#lock`을 붙여 열면 이 브라우저에
  저장된 키를 지웁니다. 다시 빌드하면 salt가 바뀌므로 한 번 더 입력해야 합니다.
- 보호 범위: 비밀번호 없이는 코드·보정값·결과를 읽을 수 없습니다. **강도는 비밀번호 강도가 전부입니다** — 암호화된
  파일은 누구나 받을 수 있으므로, 연구실 이름·단어에 숫자를 붙인 것 같은 비밀번호는 오프라인 대입으로 금방 풀릴 수
  있습니다. 우연히 여는 것은 막지만 작정한 공격은 막지 못합니다. 긴 무작위 문구(무작위 단어 4개 이상)를 권합니다
  (바꾸려면 새 비밀번호로 다시 빌드·게시). 비밀번호를 아는 사람은 모든 내용을 볼 수 있고, 페이지의 존재·파일 크기는
  숨겨지지 않습니다. 저장소(GitHub)가 공개되어 있으면 소스와 모델이 그대로 보이므로 이 잠금은 의미가 없습니다 —
  저장소는 비공개여야 합니다.
- claude.ai 링크는 소유자가 claude.ai의 공유 설정에서 공유해야 다른 사람이 열 수 있습니다(비공개 아티팩트는 소유자만 열림).
  같은 링크에 다시 게시할 때는 이전 평문 파일(옛 JS 청크, `snapshot/*.json.gz`, `snapshot/index.json`)을 지워야 합니다.

### 화면 구성
- 헤더: 로고(biristor 기호 — 원, 기울어진 평행사변형 루프, 좌우로 마주 보는 리드; `components/Logo.tsx`·`public/favicon.svg`), 제목과
  기술 칩(FDSOI — 마우스를 올리면 L_g·W·T_Si·EOT), 탭(소자 · 회로 · 레퍼런스 · 문서), **Deterministic |
  Stochastic** 토글(모드별 강조색: 청록/보라), 백엔드 상태 점, KO/EN, 밝은/어두운 테마.
- 모드 막대 오른쪽 끝: **정보**(KAIST · NOBEL 연구실). 누르면 소개 창(`components/Credits.tsx`: 설명, 소자·모델
  범위, 연구실, 버전)이 열립니다.
- 왼쪽 사이드바: 프리셋(기준 보정/광조사 보정/사용자 정의 — 값을 바꾸면 “사용자 정의 (…에서 수정)”), 그룹별 파라미터
  카드(각 카드에 **상세** 버튼, 초기화 링크, 기본값 대비 변경 점, ⓘ 툴팁), 하단 고정 실행 바(진행률, 메시지,
  경과 시간, 취소, 결정론 자동 실행, Ctrl/⌘+Enter).
- 본문: KPI 줄 + 패널 격자. 모든 패널에 **상세**, CSV/PNG 내보내기. Stochastic 모드에서는 KPI 아래에 **통계 요약**
  (평균·95 % 신뢰구간·SD·IQR·분위수·왜도·첨도·lag-1·중도절단, 측정값 대비 Δ평균·SD 비·KS 검정)이 붙습니다.
- **소자 라이브러리**(`devices/`): 사이드바 소자 카드(기술 FDSOI / PDSOI·Bulk 준비 중, 형상, 보정 프리셋,
  “소자로 저장”), 관리 창(불러오기·회로에 배치·이름 바꾸기·복제·삭제·JSON 내보내기/가져오기).
- **회로 탭**: **회로도 편집기**(기본, `schematic/`)와 **빠른 벤치**. 편집기는 LTspice처럼 R·C·V·I 전원(DC/PULSE/
  PWL/SINE)·접지·STL(라이브러리 소자)을 배치·배선하고(단축키 R C V I G X W N P, Ctrl+R/Z/Y/D), `.tran` 설정(정지
  시간, 저장 시작, 최대/최소 time step, BE/TRAP, reltol)으로 `bench: "custom"`을 실행합니다. 노드/소자를 클릭하면
  V(노드)·I(소자) 파형이 추가되고, 시간 커서 위치의 노드 전압과 단자 전류(방향 화살표)가 회로도 위에 표시됩니다.
  Stochastic 모드는 run별 파형 + 평균 ± SD 대역, STL별 사건 통계, 분포 통계 표를 보여 줍니다.
- **상세 창**(`PhysicsWindow`): 화면 전환 없이 버튼 옆에 뜨는 작은 창. 헤더로 끌어 이동, 오른쪽 아래 모서리로
  크기 조절, Esc/×로 닫기(포커스는 버튼으로 복귀). 섹션 알약, 번호 붙은 KaTeX 수식(LaTeX 복사), 변수 표,
  가정/주의, 관련 주제(뒤로 가기), “물리 모델 탭에서 열기”.

### 폴더 구조
```
src/
  api/        types.ts(계약 §2/§4 타입) client.ts(HTTP·작업 폴링) mock.ts(데모 데이터) guards.ts measured.ts
  state/      store.ts(Zustand: 모드·탭·언어·테마·파라미터·결과·상세 창) runner.ts(작업 실행·자동 실행) presets.ts
  params/     schema.ts(파라미터 그룹/필드 정의) benches.ts(회로 벤치)
  i18n/       strings.ts(타입이 있는 KO/EN 사전) strings.{brand,schematic,stats}.ts(패키지별 키) index.ts(useT)
  components/ Header, Logo, Credits(크레딧 + 소개 창), Panel, Plot(지연 로딩 Plotly), Tex, RichText, Tooltip, …
  sidebar/    Sidebar(프리셋·실행 바) ParamGroup Field
  device/     DeviceTab KpiStrip DetPanels(결정론 4개) StoPanels(확률 6개) common
  circuit/    CircuitTab(편집기/빠른 벤치 전환) Schematic(벤치 넷리스트 → SVG) BenchIcons
  schematic/  회로도 편집기(모델·넷·ERC·netlist·편집·결과 뷰어)
  devices/    소자 라이브러리(저장소·소자 카드·관리 창)
  stats/      통계 모듈(describe, ks2, histogram, ecdf, StatsTable)
  validation/ ValidationTab
  physics/    PhysicsWindow TopicBody(공용 렌더러) PhysicsTab
  plots/      theme.ts(Plotly 테마·의미 색)
  utils/      format(SI 접두사) payload(페이로드 빌더) object csv
  content/physics/   물리 콘텐츠 (physics-content 패키지 소유 — 여기서는 import만)
e2e/          Playwright 테스트, screenshots/
```

### 패널 추가하기
1. 결과 타입을 `src/api/types.ts`에, 필수 키를 `src/api/guards.ts`의 `SPECS`에 추가(새 kind라면
   `Kind`와 `mock.ts`의 데모 생성기도).
2. `device/`(또는 해당 탭)에 컴포넌트를 만든다: `useEntry<T>(key)`로 결과를 읽고, Plotly `data/layout`을
   만든 뒤 `<Panel id title desc topic entry hasData currentKey plot toolbar />`로 감싼다. 로딩/빈 상태/오류/
   CSV·PNG/“파라미터 변경됨” 배지는 `Panel`이 처리한다. 색은 `usePalette()`(HRS/LRS/측정/확률…)를 쓴다.
3. 계산 실행: `state/runner.ts`의 `runKey(key, kind, payload)`(페이로드는 `utils/payload.ts`에 순수 함수로).
   실행 버튼에 묶으려면 `runDeterministic`/`runStochastic`의 목록에 추가.
4. 문자열은 `src/i18n/strings.ts`에 KO/EN으로 추가.
   축 제목·범례·hover 문구는 `axis.*` 키(`src/i18n/strings.brand.ts`): x축과 단일 y축은 “이름 기호 (단위)”
   (예: `드레인 전압 V<sub>D</sub> (V)`), 위아래로 쌓인 subplot의 y축은 “기호 (단위)”.

### 문구 규칙
한국어 라벨은 명사형, 문장은 합니다체로 씁니다. 영어는 문장형 대소문자(sentence case)와 미국식 철자.
branch, fold, hazard, latch-up/latch-down 같은 모델 용어는 라틴 문자로 둡니다. 용어:
latch-up 전압 V_LU / latch-up voltage, latch-down 전압 V_LD, 히스테리시스 창 ΔV / hysteresis window,
래치 창(fold 쌍이 있는 V_G 범위) / latch window, 국소 상태 / local state, 캐리어 잡음 / carrier noise,
첫 통과 / first passage, 중도절단 / censored, 바디 / body, 부하선 / load line, 기준 측정 기록 / reference
record, 기준 보정 / reference calibration(프리셋 id `paper`), 광조사 보정 / illumination calibration(`photo`),
기본 모델 / base model(확장 항 0), 고정·진화·없음 / frozen·evolving·none. 모델은 아직 발표되지 않았으므로
“논문/paper”라고 쓰지 않고 소자·기록으로 설명합니다(예: “FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm ·
EOT 14.1 nm”).

### 물리 주제 추가하기
콘텐츠는 physics-content 패키지가 `src/content/physics/topics/<id>.ts`로 관리한다(형식: `types.ts`).
새 주제는 `TopicId`/`TOPIC_ORDER`에 추가하고 파일을 만들면 물리 모델 탭과 상세 창에 자동으로 나타난다.
UI에서 연결하려면 `<DetailsButton topic="<id>" />` 또는 `GroupDef.topic`/`Panel topic`에 id를 지정.

---

## English

### Develop
```bash
cd web
npm ci
npm run dev            # http://127.0.0.1:5173 — /api proxied to :8000 (uvicorn server.main:app --port 8000)
# without a backend: http://127.0.0.1:5173/?mock=1 (demo mode is also used automatically if /api/health fails)
```

### Build & check
```bash
npm run typecheck      # tsc -b --noEmit
npm run build          # web/dist — served by the FastAPI app at /
npx vitest run         # unit tests
npx playwright test    # e2e (Chromium, headless): mock-mode specs + a live spec when :8000 answers
```
Screenshots are written to `e2e/screenshots/`. Never run `playwright install` (browsers are in `/opt/pw-browsers`).

### Static snapshot (published page without a server)
A static copy of the app (the claude.ai artifact) cannot reach `/api`; it then loads `./snapshot/index.json` and runs
in **snapshot mode** (`src/api/snapshot.ts`): each compute request is looked up by `sha256(sorted-key JSON of
{kind, payload})` and answered with the recorded model result (`snapshot/<key>.json.gz`, inflated with
DecompressionStream, plain `.json` fallback); requests that are not in the snapshot get demo data and the panel and
banner say so. Deterministic device results (branches, charge balance, V_G curve) are also recorded on Device 1's
V_G grid (−3.8 … −0.9 V in 0.1 V steps); a request that only changes V_G shows the nearest grid point's real result,
marked "nearest precomputed V_G = x V". CSVM replays only the recorded default run; changed geometry, V_BG or CSVM
settings show a "needs the compute server" error instead of demo data. `?mock=1` and "no backend, no snapshot" still give the demo mode.
```bash
# with the backend on :8000 (elsewhere: STL_API=http://127.0.0.1:8011)
npm run snapshot:record && npm run build:artifact   # records web/snapshot/ → builds web/dist-artifact/ (gitignored)
npm run verify:artifact -- --shots /tmp/shots     # serves it like the host (/api → 404) and checks it in Chromium
```
The recorder (`scripts/record-snapshot.mjs`) drives a Vite dev server + Chromium through the default flows in both
languages (Device tab deterministic/stochastic with VSCM and the default CSVM run, stochastic V_G curve, every
schematic example deterministic and stochastic, the Reference tab) plus the V_G grid (once) and stores exactly what the app received (dev-only hook enabled by
`localStorage["stl-websim:record"]="1"`, compiled out of builds). Re-record whenever the model, presets, examples or
payloads change (a changed key simply shows demo data). `web/snapshot/` is generated and not committed. Publish
`dist-artifact/stl-simulator.html` (a fragment without doctype/html/head/body) plus the files in `files.json`.

### Password-locked publish (locked build)
The published build can be encrypted as a whole so that the link alone shows nothing (link + password).
```bash
STL_ARTIFACT_PASSWORD='…' npm run build:artifact -- --lock   # → web/dist-artifact/ (locked; --out for another folder)
STL_ARTIFACT_PASSWORD='…' npm run verify:artifact           # wrong/right password, remember-me, CSP matrix in Chromium
```
- The password is read only from the environment variable (never a flag, file or log line; `verify:artifact` refuses
  `--password` too). `--lock` is refused when the variable is missing or has leading/trailing whitespace or a control
  character such as a newline.
- Encryption: the whole app (one IIFE script, `assets/app-*.wasm`), the app stylesheet (`assets/style-*.wasm`) and every
  snapshot file (`snapshot/index.wasm` included) are gzipped and encrypted with AES-256-GCM; key = PBKDF2-SHA256(password,
  random 16-byte salt, 600 000 iterations), a random IV per file, the published path as AAD (format:
  `scripts/lock-crypto.mjs`). Plaintext is only the password card (`stl-simulator.html`, `lock.js`, `lock.css` — no lab
  or people names), the public parameters (`lock.json`: salt, iterations, file names) and the stock KaTeX fonts
  (checked byte-identical to the KaTeX package).
- The build writes to a temporary folder and replaces the output folder only after its self-check passes (every file
  decrypts; no model string, calibrated number or name in any plaintext file; the password in no file). A failed build
  leaves the previous build untouched and produces nothing publishable.
- The page: enter the password (≈ 0.5 s key derivation) → the app and its stylesheet are decrypted → started with
  whatever the host's CSP allows (`blob:` script → inline script → `new Function`). "Remember on this browser" keeps
  the derived key (not the password) in localStorage for 30 days, otherwise in sessionStorage (gone when the tab
  closes). The app's own saved settings (model parameters included) are stored encrypted, never in plaintext. Opening
  the link with `#lock` appended removes the saved key from that browser. A rebuild changes the salt, so everyone
  enters the password once more.
- What it protects: without the password the code, calibration and results are unreadable. **Its strength is the
  password's strength**: anyone can download the encrypted files, so a password made of a lab name or a word plus
  digits can be guessed offline quickly. It stops casual access, not a determined attacker. Use a long random
  passphrase (4 or more random words); changing it means rebuilding and republishing. Whoever knows the password sees
  everything, and the page's existence and file sizes are not hidden. If the GitHub repository is public, the source
  and the model are readable there and this lock protects nothing: the repository must be private.
- The claude.ai link itself must also be shared by the owner in claude.ai's share settings (a private artifact opens
  only for its owner). When republishing to the same link, remove the old plaintext files (old JS chunks,
  `snapshot/*.json.gz`, `snapshot/index.json`).

### UI overview
- Header: logo (the biristor symbol: a circle, a slanted parallelogram loop and opposed horizontal leads;
  `components/Logo.tsx` and `public/favicon.svg`), title and technology chip (FDSOI; hover for L_g, W, T_Si,
  EOT), tabs (Device · Circuit · Reference · Docs), the **Deterministic | Stochastic** toggle (teal / violet
  accent follows the mode), backend status dot, KO/EN, light/dark theme.
- Mode strip, right end: **About** (KAIST · NOBEL Lab) opening the About card (`components/Credits.tsx`:
  description, device and model scope, lab, version from `/api/health`).
- Sidebar: preset selector (reference / illumination / custom — editing shows “Custom (edited from …)”), grouped
  collapsible parameter cards (Details button, reset link, changed-from-default dot, ⓘ tooltip with symbol,
  meaning, `p[i]` index and default), sticky Run bar (progress, job message, elapsed time, Cancel,
  deterministic auto-run, Ctrl/⌘+Enter).
- Main: KPI strip + responsive panel grid; every panel has Details and CSV/PNG export.
- **Details window** (`PhysicsWindow`): compact non-modal floating window opened next to the clicked button,
  draggable by its header, resizable (bottom-right corner), Esc/× closes and returns focus. Section pills,
  numbered KaTeX equations (copy LaTeX), variable tables, notes, related topics with back navigation,
  “Open in Physics tab”.

### Circuit editor and device library
- **Device library** (`devices/`): the sidebar Device card (technology FDSOI; PDSOI/Bulk coming later; geometry;
  calibration presets; “Save as device”) and a manager (load, place in schematic, rename, duplicate, delete,
  JSON export/import).
- **Circuit tab**: the **Schematic editor** (default, `schematic/`) and **Quick benches**. Place and wire R, C,
  V/I sources (DC/PULSE/PWL/SINE), ground and STL cells from the library (shortcuts R C V I G X W N P,
  Ctrl+R/Z/Y/D), set the `.tran` options (stop time, start saving, max/min time step, BE/TRAP, reltol) and run
  `bench: "custom"`. Clicking a node or part adds V(node)/I(part) traces; the time cursor writes node voltages and
  terminal currents (with direction arrows) onto the schematic. Stochastic runs add a mean ± SD band, per-STL
  event statistics and a statistics table.
- **Statistics** (`stats/`): `describe`, `ks2`, `histogram`, `ecdf` and `StatsTable`, shared by the Device tab
  (stochastic) and the circuit editor.

### Structure
See the Korean section above (same tree). Units: the store keeps values exactly as the payload schema
defines them (SI, or the unit in the key name such as `iph_pA`, `l_gidl_nm`); fields convert for display
with a `scale` (e.g. τ_bulk in µs, σ_φ in mV).

### Add a panel
1. Add the result type to `src/api/types.ts` and its required keys to `SPECS` in `src/api/guards.ts`
   (for a new kind also `Kind` and a demo generator in `src/api/mock.ts`).
2. Write the component: read with `useEntry<T>(key)`, build Plotly `data/layout`, wrap in
   `<Panel id title desc topic entry hasData currentKey plot toolbar />` — loading, empty, error, CSV/PNG
   and the “parameters changed” badge are handled by `Panel`. Use `usePalette()` for semantic colours.
3. Compute with `runKey(key, kind, payload)` from `src/state/runner.ts` (payload builders are pure functions
   in `src/utils/payload.ts`); add the key to `runDeterministic` / `runStochastic` to bind it to Run.
4. Add KO/EN strings to `src/i18n/strings.ts`.
   Axis titles, legend names and hover words use the `axis.*` keys (`src/i18n/strings.brand.ts`): x-axes and
   single-panel y-axes are “Name symbol (unit)” (e.g. `Drain voltage V<sub>D</sub> (V)`), stacked subplot
   y-axes are “symbol (unit)”.

### Wording
Korean labels are noun phrases and sentences use 합니다체; English uses sentence case and American spelling.
Terms: latch-up voltage V_LU, latch-down voltage V_LD, hysteresis window ΔV, latch window (V_G range with a pair
of folds), local state, carrier noise, reference calibration (preset id `paper`), illumination calibration
(`photo`), base model (all extensions at zero). The model is unpublished: never write “paper”; describe the device
or record instead (e.g. “FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm”).

### Add a physics topic
Content is owned by the physics-content package (`src/content/physics/topics/<id>.ts`, format in `types.ts`).
Add the id to `TopicId`/`TOPIC_ORDER` and create the file; it appears in the Physics tab and Details window.
Link it from the UI with `<DetailsButton topic="<id>" />`, `GroupDef.topic` or `Panel topic`.

### Backend assumptions
- Jobs: `POST /api/compute/{kind}?wait=1.5`, poll `GET /api/jobs/{id}` every 400 ms, `DELETE` to cancel; a
  newer run of the same panel supersedes (and cancels) the older job. HTTP 429 (server queue full): the panel
  shows “server busy — retrying in N s” and the submit is retried once after `Retry-After` (bounded 0.2–60 s);
  a second 429 shows a friendly error. Error bodies are `{"detail": "<string>"}`.
- Result shapes per `docs/WEB_CONTRACT.md` §2/§4; extra keys are ignored, missing required keys show an
  “unexpected result shape” message in the panel. Warnings are shown as a collapsible notice.
- `GET /api/data/measured` and `/api/data/design_map` are normalised in `src/api/measured.ts`
  (`photo.conditions/V_LU`, `light_iv`, `paper_idvd.up/down.{median,p10,p90}`; design map `arrays`/`scalars`).
- `device.preset` is sent as the preset the values were loaded from (`paper` = reference calibration, `photo` =
  illumination calibration, `custom`) together with
  every field, so the server-side resolution is a no-op.
- Circuit `bench_params`: fields left on “auto” (`null`) are omitted from the request, so the server's
  `BENCH_DEFAULTS` apply (auto-resolved values such as `v_max_V`, or documented defaults such as
  `rise_s`/`fall_s` = 10 µs, p-bit 20 µs); the resolved values come back in `result.bench_params` (chips).
- Persisted UI state (`localStorage["stl-websim:v1"]`, schema version `v`) is validated field by field on
  load (`src/state/persist.ts`); junk or older data falls back to defaults instead of breaking the app.
