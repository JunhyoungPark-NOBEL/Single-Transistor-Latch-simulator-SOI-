// UI strings owned by the brand work package (merged into STRINGS in strings.ts).
// Keys must be prefixed "brand." (or "axis." for the plot-label block) to avoid collisions.
// Natural Korean and natural English.
// Facts only: no e-mail addresses, no URLs, the NOBEL acronym is not expanded.
import type { L10n } from "../content/physics/types";

export const BRAND_STRINGS = {
  // header technology chip
  "brand.tech": { ko: "FDSOI", en: "FDSOI" },
  "brand.tech.title": {
    ko: "소자 기술: FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm (PDSOI · Bulk 준비 중)",
    en: "Technology: FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm (PDSOI and bulk coming later)",
  },

  // credits corner (context strip, right end): full text at ≥ 1280 px, the short form below (CSS)
  "brand.credits.full": { ko: "제작자", en: "Credits" },
  "brand.credits.short": { ko: "제작자", en: "Credits" },
  "brand.credits.aria": { ko: "STL Simulator 정보와 만든 사람 보기", en: "About STL Simulator and credits" },

  // About popover
  "brand.about.title": { ko: "STL Simulator 정보", en: "About STL Simulator" },
  "brand.about.tagline": { ko: "SOI 단일 트랜지스터 래치 시뮬레이터", en: "Single-transistor latch simulator for SOI" },
  "brand.about.body": {
    ko: "SOI n-MOSFET의 단일 트랜지스터 래치(STL) 동작을 계산하는 웹 시뮬레이터입니다. 결정론적(평균) 모델은 정상상태 branch, 래치업·래치다운 전압과 바디 전하 균형을 구하고, 확률적 모델(Eq. 2 캐리어 잡음 + 국소 상태)은 사이클 간 변동, V_LU/V_LD 분포와 hazard를 Monte Carlo로 계산합니다. 같은 소자 모델을 회로에 넣어 과도해석도 할 수 있습니다.",
    en: "A web simulator for the single-transistor latch (STL) in SOI n-MOSFETs. The deterministic (mean) model gives the steady-state branches, the latch-up and latch-down voltages and the body-charge balance; the stochastic model (Eq. 2 carrier noise + local states) gives cycle-to-cycle variability, V_LU/V_LD distributions and the hazard by Monte Carlo. The same device model can also be simulated in circuits (transient analysis).",
  },
  "brand.about.device": { ko: "소자", en: "Device" },
  "brand.about.device.value": { ko: "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm", en: "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm" },
  "brand.about.glyph": { ko: "Biristor 심볼", en: "Biristor symbol" },
  "brand.about.device.soon": { ko: "PDSOI · Bulk 모델은 준비 중입니다.", en: "PDSOI and bulk models are coming later." },
  "brand.about.scope": { ko: "미발표 모델 — 연구용으로만 사용하세요.", en: "Unpublished model — for research use only." },
  "brand.about.lab": { ko: "연구실", en: "Lab" },
  "brand.about.lab.value": { ko: "NOBEL 연구실", en: "NOBEL Lab" },
  "brand.about.advisor": { ko: "지도교수", en: "Advisor" },
  "brand.about.advisor.value": { ko: "", en: "" },
  "brand.about.institution": { ko: "소속", en: "Institution" },
  "brand.about.institution.value": { ko: "KAIST", en: "KAIST" },
  "brand.about.developer": { ko: "개발", en: "Developer" },
  "brand.about.developer.value": { ko: "박준형", en: "Junhyoung Park" },
  "brand.about.version": { ko: "버전", en: "Version" },
  "brand.about.version.app": { ko: "앱 {v}", en: "app {v}" },
  "brand.about.version.engine": { ko: "엔진 {v}", en: "engine {v}" },
  "brand.about.version.demo": { ko: "데모 모드 — 백엔드 버전 정보 없음", en: "Demo mode — no backend version" },
  "brand.about.version.offline": { ko: "백엔드에 연결되지 않음", en: "Backend offline" },

  // validation: reference markers on the V_G-dependence figure (Plotly legend names: HTML subscripts)
  "brand.v.refMeanPeak": { ko: "기준값: 평균 V<sub>LU</sub> 최대 4.354 V (V<sub>G</sub> −1.10 V)", en: "Reference: mean V<sub>LU</sub> max 4.354 V (V<sub>G</sub> −1.10 V)" },
  "brand.v.refSdPeak": { ko: "기준값: σ<sub>LU</sub> 최대 129.8 mV (V<sub>G</sub> −1.25 V)", en: "Reference: σ<sub>LU</sub> max 129.8 mV (V<sub>G</sub> −1.25 V)" },

  // ================================================================ axis.* — plot labels (axes-logo package)
  // Every Plotly axis title, colorbar title, legend name, hover word and annotation in the result panels.
  // Plotly HTML: <sub>…</sub> for subscripts. Units in parentheses; dimensionless quantities have none.
  // LONG  "Name symbol (unit)" — every x-axis, single-panel y-axes, colorbars (e.g. "Drain voltage V_D (V)").
  // SHORT "symbol (unit)" (axis.s.*) — y-axes of vertically stacked subplots, where the legend names the
  //   quantity. Circuit signal axes hold many signals and have no single symbol: their compact form is the
  //   name (axis.c.*). Currents: SI tick prefixes with an "A" suffix; time scales to s/ms/µs/ns.
  "axis.vd": { ko: "드레인 전압 V<sub>D</sub> (V)", en: "Drain voltage V<sub>D</sub> (V)" },
  "axis.vg": { ko: "게이트 전압 V<sub>G</sub> (V)", en: "Gate voltage V<sub>G</sub> (V)" },
  "axis.idAbs": { ko: "드레인 전류 |I<sub>D</sub>| (A)", en: "Drain current |I<sub>D</sub>| (A)" },
  "axis.id": { ko: "드레인 전류 I<sub>D</sub> (A)", en: "Drain current I<sub>D</sub> (A)" },
  "axis.vdsCell": { ko: "{cell} 드레인–소스 전압 V<sub>DS</sub> (V)", en: "{cell} drain–source voltage V<sub>DS</sub> (V)" },
  "axis.icomp": { ko: "성분 전류 |I| (A)", en: "Component current |I| (A)" },
  "axis.cb.u": { ko: "소스–바디 바이어스 u (V)", en: "Source–body bias u (V)" },
  "axis.cb.q": { ko: "바디 전하 Q<sub>B</sub> (fC)", en: "Body charge Q<sub>B</sub> (fC)" },
  "axis.foldV": { ko: "래치업·래치다운 전압 V<sub>LU</sub>, V<sub>LD</sub> (V)", en: "Latch-up / latch-down voltage V<sub>LU</sub>, V<sub>LD</sub> (V)" },
  "axis.vSwitch": { ko: "래치업·래치다운 전압 V<sub>LU</sub>, V<sub>LD</sub> (V)", en: "Latch-up / latch-down voltage V<sub>LU</sub>, V<sub>LD</sub> (V)" },
  "axis.vlu": { ko: "래치업 전압 V<sub>LU</sub> (V)", en: "Latch-up voltage V<sub>LU</sub> (V)" },
  "axis.vld": { ko: "래치다운 전압 V<sub>LD</sub> (V)", en: "Latch-down voltage V<sub>LD</sub> (V)" },
  "axis.count": { ko: "빈도", en: "Count" },
  "axis.cdf": { ko: "누적 확률 P(V ≤ v)", en: "Cumulative probability P(V ≤ v)" },
  "axis.cycle": { ko: "사이클", en: "Cycle" },
  "axis.time": { ko: "시간 ({u})", en: "Time ({u})" },
  "axis.power": { ko: "광 파워 P (mW)", en: "Optical power P (mW)" },
  "axis.dmap.L": { ko: "국소 영역 크기 L (nm)", en: "Local-region size L (nm)" },
  "axis.dmap.d": { ko: "깊이 비율 d", en: "Depth fraction d" },
  // design-map colorbars (log colour keeps these titles; the ticks show the values, not log₁₀)
  "axis.dmap.sigma_VLU_mV": { ko: "표준편차 σ<sub>LU</sub>, 0–4 V 스윕 (mV)", en: "Standard deviation σ<sub>LU</sub>, 0–4 V sweep (mV)" },
  "axis.dmap.sigma_phi_mV": { ko: "국소 전위 표준편차 σ<sub>φ</sub> (mV)", en: "Local-potential standard deviation σ<sub>φ</sub> (mV)" },
  "axis.dmap.latched_fraction": { ko: "래치 비율 f<sub>latch</sub>", en: "Latched fraction f<sub>latch</sub>" },
  "axis.dmap.sigma_VLU_sweep5p2V_mV": { ko: "표준편차 σ<sub>LU</sub>, 5.2 V 스윕 (mV)", en: "Standard deviation σ<sub>LU</sub>, 5.2 V sweep (mV)" },
  "axis.dmap.expected_trap_count": { ko: "기대 결함(trap) 수 ⟨N⟩", en: "Expected trap count ⟨N⟩" },
  "axis.dmap.footDevice": { ko: "기준 보정 σ<sub>φ</sub> = {v} mV의 L₀", en: "L₀ at the reference-calibration σ<sub>φ</sub> = {v} mV" },
  "axis.dmap.foot50": { ko: "σ<sub>φ</sub> = {v} mV(50 mV 기준선)의 L₀", en: "L₀ at σ<sub>φ</sub> = {v} mV (50 mV reference)" },
  "axis.dmap.footNt": { ko: "지도 위 숫자는 각 기준선의 N<sub>t</sub> (cm<sup>−2</sup>)", en: "numbers above the map: N<sub>t</sub> of each line (cm<sup>−2</sup>)" },
  // short forms for stacked subplots
  "axis.s.gl": { ko: "G, L (A)", en: "G, L (A)" },
  "axis.s.u": { ko: "U (k<sub>B</sub>T)", en: "U (k<sub>B</sub>T)" },
  "axis.s.h": { ko: "h (1/s)", en: "h (1/s)" },
  "axis.s.S": { ko: "S", en: "S" },
  "axis.s.vluMean": { ko: "평균 V<sub>LU</sub> (V)", en: "Mean V<sub>LU</sub> (V)" },
  "axis.s.sigmaLu": { ko: "σ<sub>LU</sub> (mV)", en: "σ<sub>LU</sub> (mV)" },
  "axis.s.vSwitch": { ko: "V<sub>LU</sub>, V<sub>LD</sub> (V)", en: "V<sub>LU</sub>, V<sub>LD</sub> (V)" },
  "axis.s.delta": { ko: "δ ({u})", en: "δ ({u})" },
  "axis.s.censored": { ko: "중도절단 (%)", en: "Censored (%)" },
  // circuit signal axes (quick benches and the schematic results)
  "axis.c.voltage": { ko: "전압 ({u})", en: "Voltage ({u})" },
  "axis.c.currentAbs": { ko: "전류 |I| ({u})", en: "Current |I| ({u})" },
  "axis.c.current": { ko: "전류 I ({u})", en: "Current I ({u})" },
  "axis.c.charge": { ko: "전하 ({u})", en: "Charge ({u})" },
  "axis.c.state": { ko: "상태 변수 ({u})", en: "State variable ({u})" },
  "axis.c.stateNoUnit": { ko: "상태 변수", en: "State variable" },
  "axis.c.logic": { ko: "논리 레벨", en: "Logic level" },
  // circuit sweeps (server symbols V_amp, P_sw, V_G, I_PH, P(1))
  "axis.sw.vamp": { ko: "펄스 진폭 V<sub>amp</sub> ({u})", en: "Pulse amplitude V<sub>amp</sub> ({u})" },
  "axis.sw.psw": { ko: "스위칭 확률 P<sub>sw</sub>", en: "Switching probability P<sub>sw</sub>" },
  "axis.sw.iph": { ko: "광전류 I<sub>PH</sub> ({u})", en: "Photocurrent I<sub>PH</sub> ({u})" },
  "axis.sw.p1": { ko: "비트 1 확률 P(1)", en: "Bit-1 probability P(1)" },
  // legend names
  "axis.leg.up": { ko: "상향 스윕", en: "Up sweep" },
  "axis.leg.down": { ko: "하향 스윕", en: "Down sweep" },
  "axis.leg.updown": { ko: "상향/하향 스윕", en: "Up/down sweeps" },
  "axis.leg.modelUp": { ko: "모델 상향 스윕 (HRS→LRS)", en: "Model, up sweep (HRS→LRS)" },
  "axis.leg.modelDown": { ko: "모델 하향 스윕 (LRS→HRS)", en: "Model, down sweep (LRS→HRS)" },
  "axis.leg.centre": { ko: "{b} (중심 상태)", en: "{b} (center state)" },
  "axis.leg.rugLu": { ko: "사이클별 V<sub>LU</sub>", en: "V<sub>LU</sub> per cycle" },
  "axis.leg.rugLd": { ko: "사이클별 V<sub>LD</sub>", en: "V<sub>LD</sub> per cycle" },
  "axis.leg.model": { ko: "{s} 모델", en: "{s} model" },
  "axis.leg.meas": { ko: "{s} 측정", en: "{s} measured" },
  "axis.leg.h": { ko: "켜짐률 h(V<sub>D</sub>)", en: "Hazard h(V<sub>D</sub>)" },
  "axis.leg.S": { ko: "생존 확률 S(V<sub>D</sub>)", en: "Survival S(V<sub>D</sub>)" },
  "axis.leg.vluMean": { ko: "평균 V<sub>LU</sub>", en: "Mean V<sub>LU</sub>" },
  "axis.leg.vldFold": { ko: "V<sub>LD</sub> fold", en: "V<sub>LD</sub> fold" },
  "axis.leg.sdTotal": { ko: "전체 σ<sub>LU</sub>", en: "Total σ<sub>LU</sub>" },
  "axis.leg.meanMeas": { ko: "측정 평균 V<sub>LU</sub>", en: "Measured mean V<sub>LU</sub>" },
  "axis.leg.sdMeas": { ko: "측정 σ<sub>LU</sub>", en: "Measured σ<sub>LU</sub>" },
  "axis.leg.noLatch": { ko: "래치 없음 (fold 없음)", en: "No latch (no fold)" },
  "axis.leg.beyond": { ko: "중도절단: {v} V 스윕 밖", en: "Censored: beyond the {v} V sweep" },
  "axis.leg.measVg": { ko: "측정, V<sub>G</sub> = {vg} V", en: "Measured, V<sub>G</sub> = {vg} V" },
  "axis.leg.modelVg": { ko: "모델, V<sub>G</sub> = {vg} V", en: "Model, V<sub>G</sub> = {vg} V" },
  "axis.leg.deviceSigmaPhi": { ko: "기준 보정 σ<sub>φ</sub> = {v} mV", en: "Reference-calibration σ<sub>φ</sub> = {v} mV" },
  "axis.leg.run0": { ko: "과도 궤적 (실행 #0)", en: "Transient trajectory (run 0)" },
  // hover words and annotations
  "axis.h.cycle": { ko: "사이클 %{x}", en: "Cycle %{x}" },
  "axis.ann.setVg": { ko: "설정 V<sub>G</sub> {v} V", en: "Set V<sub>G</sub> {v} V" },
  "axis.ann.rate": { ko: "스윕 속도 {r} V/s", en: "Sweep rate {r} V/s" },
  "axis.ann.branch": { ko: "{b} 곡선", en: "{b} branch" },
  "axis.ann.full": { ko: "전체 궤적 (모든 곡선)", en: "Full locus (all branches)" },
  "axis.cb.roots": { ko: "평형점 {n}개", en: "{n} equilibria" },

  // static snapshot mode (published page without a backend; web/src/api/snapshot.ts)
  "snapshot.banner.title": { ko: "정적 스냅샷", en: "Static snapshot" },
  "snapshot.banner": {
    ko: "기본 프리셋과 예제는 실제 모델 계산 결과입니다. V_G를 바꾸면 가장 가까운 미리 계산된 결과를, 그 밖의 값을 바꾸면 예시 데이터를 표시합니다.",
    en: "built-in presets and examples show real model results; a changed V_G shows the nearest precomputed result, other changed settings show example data.",
  },
  "snapshot.banner.some": {
    ko: "지금 화면의 일부 결과는 스냅샷에 없는 조건이라 예시 데이터입니다.",
    en: "Some results on screen are example data because their settings are not in the snapshot.",
  },
  "snapshot.miss": {
    ko: "이 조건은 정적 스냅샷에 없어 예시 데이터로 표시합니다 — 실시간 계산은 서버에서 실행하세요.",
    en: "This setting is not in the static snapshot — showing example data. Run the server for live computation.",
  },
  "snapshot.status": { ko: "정적 스냅샷 — 미리 계산된 결과 {n}개 ({date} 기록)", en: "Static snapshot — {n} precomputed results (recorded {date})" },
  "snapshot.status.short": { ko: "스냅샷", en: "snapshot" },
  "snapshot.near.vg": {
    ko: "가장 가까운 미리 계산된 V_G = {vg} V의 결과입니다 (입력 {req} V).",
    en: "Nearest precomputed V_G = {vg} V (you set {req} V).",
  },
  "snapshot.near.vgp": {
    ko: "가장 가까운 미리 계산된 조건 V_G = {vg} V, P = {p} mW의 결과입니다 (입력 {rvg} V, {rp} mW).",
    en: "Nearest precomputed setting: V_G = {vg} V, P = {p} mW (you set {rvg} V, {rp} mW).",
  },
  "snapshot.banner.near": {
    ko: "일부 결과는 가장 가까운 미리 계산된 V_G(또는 광 파워)의 값입니다.",
    en: "Some results use the nearest precomputed V_G (or optical power).",
  },
} satisfies Record<string, L10n>;
