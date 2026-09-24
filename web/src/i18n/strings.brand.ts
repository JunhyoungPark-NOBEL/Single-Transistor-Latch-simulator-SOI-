// UI strings owned by the brand work package (merged into STRINGS in strings.ts).
// Keys must be prefixed "brand." to avoid collisions. Natural Korean and natural English.
// Facts only: no e-mail addresses, no URLs, the NOBEL acronym is not expanded.
import type { L10n } from "../content/physics/types";

export const BRAND_STRINGS = {
  // header technology chip
  "brand.tech": { ko: "FDSOI", en: "FDSOI" },
  "brand.tech.title": {
    ko: "소자 기술: FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm (PDSOI · Bulk 준비 중)",
    en: "Technology: FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm (PDSOI and bulk coming later)",
  },

  // credits corner (mode strip, right end); full / compact variants are chosen by width in CSS
  "brand.credits.full": { ko: "KAIST 전기및전자공학부 · NOBEL 연구실 (지도교수 최양규) · 개발 박준형", en: "NOBEL Lab · Prof. Yang-Kyu Choi · School of Electrical Engineering, KAIST · Developed by Junhyoung Park" },
  "brand.credits.mid": { ko: "KAIST · NOBEL 연구실 (최양규 교수) · 개발 박준형", en: "NOBEL Lab (Prof. Yang-Kyu Choi), KAIST · by Junhyoung Park" },
  "brand.credits.short": { ko: "KAIST · NOBEL 연구실", en: "NOBEL Lab · KAIST" },
  "brand.credits.aria": { ko: "STL Simulator 정보와 만든 사람 보기", en: "About STL Simulator and credits" },

  // About popover
  "brand.about.title": { ko: "STL Simulator 정보", en: "About STL Simulator" },
  "brand.about.tagline": { ko: "SOI 단일 트랜지스터 래치 시뮬레이터", en: "Single-transistor latch simulator for SOI" },
  "brand.about.body": {
    ko: "SOI n-MOSFET의 단일 트랜지스터 래치(STL) 동작을 계산하는 웹 시뮬레이터입니다. 결정론(평균) 모델은 정상상태 branch, 래치업·래치다운 전압과 바디 전하 균형을 구하고, 확률 모델(Eq. 2 캐리어 잡음 + 국소 상태)은 사이클 간 변동, V_LU/V_LD 분포와 hazard를 Monte Carlo로 계산합니다. 같은 소자 모델을 회로에 넣어 과도해석도 할 수 있습니다.",
    en: "A web simulator for the single-transistor latch (STL) in SOI n-MOSFETs. The deterministic (mean) model gives the steady-state branches, the latch-up and latch-down voltages and the body-charge balance; the stochastic model (Eq. 2 carrier noise + local states) gives cycle-to-cycle variability, V_LU/V_LD distributions and the hazard by Monte Carlo. The same device model can also be simulated in circuits (transient analysis).",
  },
  "brand.about.device": { ko: "소자", en: "Device" },
  "brand.about.device.value": { ko: "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm", en: "FDSOI · L_g 500 nm · W 200 nm · T_Si 50 nm · EOT 14.1 nm" },
  "brand.about.device.soon": { ko: "PDSOI · Bulk 모델은 준비 중입니다.", en: "PDSOI and bulk models are coming later." },
  "brand.about.scope": { ko: "미발표 모델 — 연구용으로만 사용하세요.", en: "Unpublished model — for research use only." },
  "brand.about.lab": { ko: "연구실", en: "Lab" },
  "brand.about.lab.value": { ko: "NOBEL 연구실", en: "NOBEL Lab" },
  "brand.about.advisor": { ko: "지도교수", en: "Advisor" },
  "brand.about.advisor.value": { ko: "최양규 교수", en: "Prof. Yang-Kyu Choi" },
  "brand.about.institution": { ko: "소속", en: "Institution" },
  "brand.about.institution.value": { ko: "KAIST 전기및전자공학부", en: "School of Electrical Engineering, KAIST" },
  "brand.about.developer": { ko: "개발", en: "Developer" },
  "brand.about.developer.value": { ko: "박준형", en: "Junhyoung Park" },
  "brand.about.version": { ko: "버전", en: "Version" },
  "brand.about.version.app": { ko: "앱 {v}", en: "app {v}" },
  "brand.about.version.engine": { ko: "엔진 {v}", en: "engine {v}" },
  "brand.about.version.demo": { ko: "데모 모드 — 백엔드 버전 정보 없음", en: "Demo mode — no backend version" },
  "brand.about.version.offline": { ko: "백엔드에 연결되지 않음", en: "Backend offline" },

  // validation: reference markers on the V_G-dependence figure
  "brand.v.refMeanPeak": { ko: "기준값: 평균 V_LU 최대 4.354 V (V_G −1.10 V)", en: "Reference: mean V_LU peak 4.354 V at V_G −1.10 V" },
  "brand.v.refSdPeak": { ko: "기준값: σ_LU 최대 129.8 mV (V_G −1.25 V)", en: "Reference: σ_LU peak 129.8 mV at V_G −1.25 V" },
} satisfies Record<string, L10n>;
