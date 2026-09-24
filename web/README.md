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

### 화면 구성
- 헤더: 탭(소자 · 회로 · 검증 · 물리 모델), **Deterministic | Stochastic** 토글(모드별 강조색: 청록/보라),
  백엔드 상태 점, KO/EN, 밝은/어두운 테마.
- 왼쪽 사이드바: 프리셋(논문/광조사/사용자 정의 — 값을 바꾸면 “사용자 정의 (…에서 수정)”), 그룹별 파라미터
  카드(각 카드에 **상세** 버튼, 초기화 링크, 기본값 대비 변경 점, ⓘ 툴팁), 하단 고정 실행 바(진행률, 메시지,
  경과 시간, 취소, 결정론 자동 실행, Ctrl/⌘+Enter).
- 본문: KPI 줄 + 패널 격자. 모든 패널에 **상세**, CSV/PNG 내보내기.
- **상세 창**(`PhysicsWindow`): 화면 전환 없이 버튼 옆에 뜨는 작은 창. 헤더로 끌어 이동, 오른쪽 아래 모서리로
  크기 조절, Esc/×로 닫기(포커스는 버튼으로 복귀). 섹션 알약, 번호 붙은 KaTeX 수식(LaTeX 복사), 변수 표,
  가정/주의, 관련 주제(뒤로 가기), “물리 모델 탭에서 열기”.

### 폴더 구조
```
src/
  api/        types.ts(계약 §2/§4 타입) client.ts(HTTP·작업 폴링) mock.ts(데모 데이터) guards.ts measured.ts
  state/      store.ts(Zustand: 모드·탭·언어·테마·파라미터·결과·상세 창) runner.ts(작업 실행·자동 실행) presets.ts
  params/     schema.ts(파라미터 그룹/필드 정의) benches.ts(회로 벤치)
  i18n/       strings.ts(타입이 있는 KO/EN 사전) index.ts(useT)
  components/ Header, Panel, Plot(지연 로딩 Plotly), Tex, RichText, Tooltip, DetailsButton, icons …
  sidebar/    Sidebar(프리셋·실행 바) ParamGroup Field
  device/     DeviceTab KpiStrip DetPanels(결정론 4개) StoPanels(확률 6개) common
  circuit/    CircuitTab Schematic(넷리스트 → SVG) BenchIcons
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

### UI overview
- Header: tabs (Device · Circuit · Validation · Physics), the **Deterministic | Stochastic** toggle (teal /
  violet accent follows the mode), backend status dot, KO/EN, light/dark theme.
- Sidebar: preset selector (paper / photo / custom — editing shows “Custom (modified from …)”), grouped
  collapsible parameter cards (Details button, reset link, changed-from-default dot, ⓘ tooltip with symbol,
  meaning, `p[i]` index and default), sticky Run bar (progress, job message, elapsed time, Cancel,
  deterministic auto-run, Ctrl/⌘+Enter).
- Main: KPI strip + responsive panel grid; every panel has Details and CSV/PNG export.
- **Details window** (`PhysicsWindow`): compact non-modal floating window opened next to the clicked button,
  draggable by its header, resizable (bottom-right corner), Esc/× closes and returns focus. Section pills,
  numbered KaTeX equations (copy LaTeX), variable tables, notes, related topics with back navigation,
  “Open in Physics tab”.

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
- `device.preset` is sent as the preset the values were loaded from (paper / photo / custom) together with
  every field, so the server-side resolution is a no-op.
- Circuit `bench_params`: fields left on “auto” (`null`) are omitted from the request, so the server's
  `BENCH_DEFAULTS` apply (auto-resolved values such as `v_max_V`, or documented defaults such as
  `rise_s`/`fall_s` = 10 µs, p-bit 20 µs); the resolved values come back in `result.bench_params` (chips).
- Persisted UI state (`localStorage["stl-websim:v1"]`, schema version `v`) is validated field by field on
  load (`src/state/persist.ts`); junk or older data falls back to defaults instead of breaking the app.
