// Physics topic "overview" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "overview",
  title: { ko: "STL 개요: 동작 원리, 소자, 상태 변수", en: "STL overview: operating principle, device and state variables" },
  summary: {
    ko: String.raw`단일 트랜지스터 래치(single-transistor latch, STL)는 플로팅 바디(floating body) SOI n-MOSFET이다. 기생 n-p-n 바이폴라 작용과 드레인 접합의 충돌 이온화(impact ionization)가 양의 되먹임을 이루어, 래치업 전압 $V_{\mathrm{LU}}$와 래치다운 전압 $V_{\mathrm{LD}}$ 사이에 히스테리시스가 있는 S자형 I–V 특성이 나타난다. 모델은 두 내부 상태 변수 $u$, $r$로부터 모든 전류와 $V_D$를 계산한다.`,
    en: String.raw`The single-transistor latch (STL) is a floating-body SOI n-MOSFET in which parasitic n-p-n bipolar action and impact ionization at the drain junction form a positive feedback loop. The result is an S-shaped I–V characteristic with hysteresis between the latch-up voltage $V_{\mathrm{LU}}$ and the latch-down voltage $V_{\mathrm{LD}}$. The model computes every current and $V_D$ from two internal state variables, $u$ and $r$.`,
  },
  tags: ["Eq. 1", "deterministic", "stochastic"],
  sections: [
    {
      heading: { ko: "동작 원리: 양의 되먹임", en: "Operating principle: positive feedback" },
      body: {
        ko: String.raw`게이트에 음의 전압을 걸어 채널을 끈 상태에서 $V_D$를 올리면 다음과 같은 되먹임 고리가 닫힌다.

- 바디(body)에 정공이 쌓이면 소스–바디 접합이 $u$만큼 순방향으로 바이어스된다.
- 소스(이미터)가 주입한 전자는 중성 바디(베이스)를 지나 드레인 공핍층(컬렉터)에 도달한다. 이것이 BJT 시드(seed) 전류 $I_{\mathrm{seed}}$이다.
- 드레인 공핍층에서 전자가 증배되어 정공 전류 $(M-1)I_{\mathrm{seed}}$가 바디로 되돌아온다.
- GIDL, 접합 BTBT, 광생성 전류 $I_{\mathrm{PH}}$는 정공을 더하고, 이미터 확산, 접합 SRH, 바디 SRH 재결합은 정공을 없앤다.

생성이 손실보다 크면 $u$가 커지고 시드 전류가 지수적으로 늘어나 소자는 저저항 상태(LRS)로 넘어간다. 정상상태 해의 궤적 $V_D(u)$는 극대($V_{\mathrm{LU}}$)와 극소($V_{\mathrm{LD}}$)를 갖는 S자 곡선이며, 이 두 꺾임점을 fold라 한다. 두 fold 사이가 히스테리시스 창이다.`,
        en: String.raw`With the gate biased negative (channel off), raising $V_D$ closes the following loop.

- Holes stored in the body forward-bias the source–body junction by $u$.
- Electrons injected by the source (emitter) cross the neutral body (base) and reach the drain depletion region (collector); this is the BJT seed current $I_{\mathrm{seed}}$.
- The electrons multiply in the drain depletion region and return a hole current $(M-1)I_{\mathrm{seed}}$ to the body.
- GIDL, junction BTBT and photogeneration $I_{\mathrm{PH}}$ add holes; emitter diffusion, junction SRH and body SRH recombination remove them.

When generation exceeds loss, $u$ rises, the seed current grows exponentially and the device switches to the low-resistance state (LRS). The steady-state locus $V_D(u)$ is S-shaped, with a maximum ($V_{\mathrm{LU}}$) and a minimum ($V_{\mathrm{LD}}$); these two turning points are the folds, and the hysteresis window lies between them.`,
      },
      equations: [
        {
          id: "eq-ov-eq1",
          label: { ko: "Eq. 1 — 바디 정공 전하 균형", en: "Eq. 1 — body hole-charge balance" },
          tex: String.raw`\frac{dQ_B}{dt} = I_{\mathrm{II}} + I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{PH}} - I_{\mathrm{REC}} - I_{\mathrm{DIFF}} \equiv F(u, r;\, V_G, I_{\mathrm{PH}})`,
          note: {
            ko: "각 항과 코드의 대응은 `charge-balance` 주제에 정리되어 있다. 정상상태 조건 $F=0$에서 I–V branch가 얻어진다.",
            en: "The mapping of every term to the code is given in the `charge-balance` topic. The steady-state condition $F=0$ gives the I–V branches.",
          },
          code: "photo_mean.py · components() → net (z[2])",
        },
        {
          id: "eq-ov-folds",
          label: { ko: "fold 전압", en: "Fold voltages" },
          tex: String.raw`\begin{aligned} V_{\mathrm{LU}} &= V_D(u_i),\quad i = \text{first local maximum of } V_D(u)\\ V_{\mathrm{LD}} &= V_D(u_j),\quad j = \text{last local minimum after } i \end{aligned}`,
          note: {
            ko: String.raw`기준 보정($V_G=-2$ V, 암조건)의 계산값: $V_{\mathrm{LU}} = 3.7037$ V, $V_{\mathrm{LD}} = 2.5979$ V.`,
            en: String.raw`Computed for the reference calibration ($V_G=-2$ V, dark): $V_{\mathrm{LU}} = 3.7037$ V, $V_{\mathrm{LD}} = 2.5979$ V.`,
          },
          code: "photo_mean.py · FastModel.classify()",
        },
      ],
    },
    {
      heading: { ko: "소자와 상수", en: "Device and constants" },
      body: {
        ko: "대상 소자는 FDSOI n-MOSFET($L_g$ 500 nm · $W$ 200 nm · $T_{\\mathrm{Si}}$ 50 nm · EOT 14.1 nm)이다. 현재는 FDSOI만 다루며, PDSOI와 벌크(bulk) 소자는 추후 추가할 예정이다.\n\n코드는 기하 상수를 m 단위(`LENGTH_M` 등)로 두고, 정전기와 수송 계산에서는 100을 곱해 cm 단위로 쓴다. 전류는 A, 전위는 V, 전계는 V/cm, 농도는 cm⁻³ 단위다. 아래 값은 기준 보정(암조건 100회 스윕 기록, $V_G=-2$ V, 0.4 V/s)의 값이다. 광조사 보정($V_G$ = −1.8/−1.1 V, 1200 V/s, 400 사이클 기록)은 이 값을 그대로 쓰고, 국소 상태 중심 $\\delta\\varphi_{G0}$와 산포 $\\sigma_\\phi$(−1.8 V 암조건 기록), 바디 결합 $\\gamma$(−1.1 V 암조건 평균), 광 변환 응답도 $R$(래치 전 plateau)만 다시 맞췄다.",
        en: "The device is an FDSOI n-MOSFET ($L_g$ 500 nm · $W$ 200 nm · $T_{\\mathrm{Si}}$ 50 nm · EOT 14.1 nm). Only FDSOI is covered at present; PDSOI and bulk devices will be added later.\n\nThe code keeps geometry in meters (`LENGTH_M`, …) and multiplies by 100 to work in cm for electrostatics and transport. Currents are in A, potentials in V, fields in V/cm and densities in cm⁻³. The values below belong to the reference calibration (dark 100-sweep record, $V_G=-2$ V, 0.4 V/s). The illumination calibration ($V_G$ = −1.8/−1.1 V, 1200 V/s, 400-cycle records) reuses them and re-fits only the local-state center $\\delta\\varphi_{G0}$ and spread $\\sigma_\\phi$ (−1.8 V dark record), the body coupling $\\gamma$ (−1.1 V dark mean) and the light-conversion responsivity $R$ (pre-latch plateau).",
      },
      variables: [
        {
          symbol: "L",
          name: { ko: "채널(바디) 길이 L_g", en: "Channel (body) length L_g" },
          value: "500 nm",
          unit: "m in code",
          code: "LENGTH_M",
        },
        {
          symbol: "W",
          name: { ko: "채널 폭", en: "Channel width" },
          value: "200 nm",
          unit: "m in code",
          code: "WIDTH_M",
        },
        {
          symbol: String.raw`T_{\mathrm{Si}}`,
          name: { ko: "실리콘 막 두께", en: "Silicon film thickness" },
          value: "50 nm",
          unit: "m in code",
          code: "TSI_M",
        },
        {
          symbol: String.raw`\mathrm{EOT}`,
          name: { ko: "등가 산화막 두께", en: "Equivalent oxide thickness" },
          value: "14.1 nm",
          unit: "m in code",
          code: "EOT_M",
        },
        {
          symbol: "N_A",
          name: { ko: "바디 억셉터 농도 (보정값)", en: "Body acceptor density (calibrated)" },
          value: "2.2958 × 10¹⁷",
          unit: "cm⁻³",
          code: "refit_3.json · NA_cm3",
        },
        {
          symbol: "N_D",
          name: { ko: "소스/드레인 도너 농도 (가정값)", en: "Source/drain donor density (assumed)" },
          value: "1 × 10²⁰",
          unit: "cm⁻³",
          code: "ND_CM3",
        },
        {
          symbol: "n_i",
          name: { ko: "진성 캐리어 농도", en: "Intrinsic carrier density" },
          value: "1 × 10¹⁰",
          unit: "cm⁻³",
          code: "NI_CM3",
        },
        {
          symbol: "T",
          name: { ko: "온도", en: "Temperature" },
          value: "300",
          unit: "K",
          code: "T",
        },
        {
          symbol: String.raw`A = W T_{\mathrm{Si}}`,
          name: { ko: "접합(이미터/컬렉터) 단면적", en: "Junction (emitter/collector) cross-section" },
          value: "1.000 × 10⁻¹⁰",
          unit: "cm²",
          code: "AREA_CM2",
        },
        {
          symbol: String.raw`C_{\mathrm{ox}}`,
          name: { ko: "게이트 산화막 용량 (전체)", en: "Gate-oxide capacitance (total)" },
          value: "2.449 × 10⁻¹⁶",
          unit: "F",
          code: "COX_F",
        },
        {
          symbol: String.raw`\mu_n,\ \mu_p`,
          name: { ko: "베이스 전자/정공 이동도 (상수)", en: "Base electron/hole mobility (constant)" },
          value: "450 / 150",
          unit: "cm²/(V·s)",
          code: "DN = 450·VT, DP = 150·VT",
        },
      ],
    },
    {
      heading: { ko: "상태 변수와 단자 관계", en: "State variables and the terminal relation" },
      body: {
        ko: "$u$는 소스–바디 접합의 준페르미 준위 분리(순방향 바이어스)이고, $r$은 드레인 접합의 역방향 바이어스다. $(u, r)$이 주어지면 `components(u, r, p, …)`가 19개 출력(V_D, I_D, F, 시드, 손실, BTBT, GIDL, 주입 수준, 길이, $R_{\\mathrm{acc}}$, 전하, 공핍폭, 채널, 정공 강하, $I_{\\mathrm{PH}}$)을 계산한다. 정상상태 모델에서 $V_G$는 채널 전류와 GIDL 전계(그리고 선택적인 국소 경로의 $\\kappa_F$ 항)를 통해서만 들어간다.",
        en: "$u$ is the quasi-Fermi-level splitting (forward bias) of the source–body junction and $r$ the reverse bias of the drain junction. Given $(u, r)$, `components(u, r, p, …)` returns 19 outputs (V_D, I_D, F, seed, losses, BTBT, GIDL, injection level, length, $R_{\\mathrm{acc}}$, charge, depletion widths, channel, hole drop, $I_{\\mathrm{PH}}$). In the steady-state model $V_G$ enters only through the channel current and the GIDL field (and the optional local-path term $\\kappa_F$).",
      },
      equations: [
        {
          id: "eq-ov-terminal",
          label: { ko: "단자 전압", en: "Terminal voltage" },
          tex: String.raw`V_D = u + r + V_T\,\Delta_p + (R_c + R_{\mathrm{acc}})\,I_D`,
          note: {
            ko: "$V_T\\Delta_p$는 중성 베이스에서의 정공 준페르미 준위 강하(전계 적분이 아님), $R_c$ = p[3], $R_{\\mathrm{acc}}$는 과잉 캐리어로 변조되는 접근 저항이다. 유도 과정은 `bjt-transport`에 있다.",
            en: "$V_T\\Delta_p$ is the hole quasi-Fermi-level drop across the neutral base (not an electrostatic field integral), $R_c$ = p[3], and $R_{\\mathrm{acc}}$ is the access resistance modulated by the excess carriers. The derivation is in `bjt-transport`.",
          },
          code: "photo_mean.py · components() (vd)",
        },
        {
          id: "eq-ov-id",
          label: { ko: "드레인 전류", en: "Drain current" },
          tex: String.raw`I_D = I_{\mathrm{seed}} + (M-1)\,I_{\mathrm{seed}} + I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{ch}} + I_{\mathrm{II,ch}} + I_{\mathrm{PH}} + I_{\mathrm{II,PH}} + I_{\mathrm{loc}}`,
          code: "photo_mean.py · components() (drain)",
        },
      ],
      variables: [
        {
          symbol: "u",
          name: { ko: "소스–바디 준페르미 준위 분리", en: "Source–body quasi-Fermi-level splitting" },
          unit: "V",
          code: "branch[:,17]",
        },
        {
          symbol: "r",
          name: { ko: "드레인 접합 역방향 바이어스", en: "Drain-junction reverse bias" },
          unit: "V",
          code: "branch[:,18]",
        },
        {
          symbol: String.raw`V_T\Delta_p`,
          name: { ko: "베이스 정공 준페르미 준위 강하", en: "Base hole quasi-Fermi-level drop" },
          value: "1.262 × 10⁻⁴ (LRS fold, V_G = −2 V)",
          unit: "V",
          code: "z[17]",
        },
        {
          symbol: "R_c",
          name: { ko: "접촉 저항", en: "Contact resistance" },
          value: "1.000",
          unit: "Ω",
          code: "p[3]",
        },
        {
          symbol: String.raw`R_{\mathrm{acc}}`,
          name: { ko: "접근 저항", en: "Access resistance" },
          value: "4.395 × 10⁵ (n̄ = 0)",
          unit: "Ω",
          code: "z[12]",
        },
        {
          symbol: "V_G",
          name: { ko: "게이트 전압", en: "Gate voltage" },
          value: "−2 (reference), −1.8 (illumination)",
          unit: "V",
          code: "p[11]",
        },
      ],
      notes: [
        {
          ko: "$u$는 준페르미 준위(주입)를 나타내는 대리 변수이며, 풀어서 얻은 정전위가 아니다(`mean_model.py` docstring). 정전기적 바디 전위는 $\\psi = u - V_T\\ln(1+\\delta/N_A)$로 따로 정의한다(`electrostatics`).",
          en: "$u$ is a quasi-Fermi/injection proxy, not a solved electrostatic potential (`mean_model.py` docstring). The electrostatic body potential is defined separately as $\\psi = u - V_T\\ln(1+\\delta/N_A)$ (`electrostatics`).",
        },
        {
          ko: String.raw`2차원 Poisson 방정식은 풀지 않는다. 두 접합은 1차원 계단 접합(abrupt junction)의 저주입 정전기로 다루며, 게이트는 채널 전류, GIDL, 전하 좌표($C_{\mathrm{ox}}$)를 통해서만 결합된다.`,
          en: String.raw`No 2D Poisson equation is solved. Both junctions use 1D abrupt-junction electrostatics at low injection; the gate couples only through the channel current, GIDL and the charge coordinate ($C_{\mathrm{ox}}$).`,
        },
      ],
    },
    {
      heading: { ko: "결정론 모델과 확률 모델", en: "Deterministic and stochastic models" },
      body: {
        ko: "**결정론(deterministic)**: Eq. 1의 정상상태 $F = 0$과 Kirchhoff 법칙으로 I–V branch(고저항 상태 HRS/불안정 branch/LRS)와 fold 전압을 구한다. 준정적 스윕은 정확히 fold에서 점프한다(`charge-balance`).\n\n**확률(stochastic)**: 정공은 이산적인 사건으로 들어오고 나간다(Eq. 2, $\\Delta Q_B = q\\sum_i s_i\\,\\Delta N_i$). GIDL, BTBT, 광생성, 재결합, 확산은 Poisson 단위 사건이고, 충돌 이온화는 클러스터 사건을 만든다(`stochastic-events`). 따라서 저전류 상태가 fold에 이르기 전에 탈출할 수 있고(첫 통과, first passage), $V_{\\mathrm{LU}}$는 fold보다 낮은 쪽에 분포한다(`first-passage`). 여기에 드레인 가장자리와 소스 가장자리의 국소 상태(local state) $\\delta\\varphi_G$, $\\delta\\varphi_E$가 사이클마다(고정, frozen) 또는 시간에 따라(OU 과정, evolving) 요동하면서 사이클 간 산포를 만든다(`local-states`, `sweep-mc`).",
        en: "**Deterministic**: the steady state $F = 0$ of Eq. 1 together with Kirchhoff's law gives the I–V branches (high-resistance state HRS/unstable branch/LRS) and the fold voltages; a quasi-static sweep jumps exactly at the folds (`charge-balance`).\n\n**Stochastic**: holes arrive and leave as discrete events (Eq. 2, $\\Delta Q_B = q\\sum_i s_i\\,\\Delta N_i$). GIDL, BTBT, photogeneration, recombination and diffusion are Poisson unit events; impact ionization produces cluster events (`stochastic-events`). The low-current state can therefore escape before the fold (first passage), so $V_{\\mathrm{LU}}$ is distributed below the fold (`first-passage`). On top of this, the drain-edge and source-edge local states $\\delta\\varphi_G$, $\\delta\\varphi_E$ fluctuate from cycle to cycle (frozen) or in time (OU process, evolving) and produce the cycle-to-cycle spread (`local-states`, `sweep-mc`).",
      },
      notes: [
        {
          ko: "`VALIDATION.md` 기준값($V_G=-2$ V): fold 3.7037 V, 중심 상태에서의 FPT 평균 ≈ 3.644 V(SD ≈ 8 mV), 진화하는 국소 상태를 넣은 100회 스윕 MC 평균 ≈ 3.63 V(SD ≈ 120 mV).",
          en: "Reference values from `VALIDATION.md` ($V_G=-2$ V): fold 3.7037 V, FPT mean at the center states ≈ 3.644 V (SD ≈ 8 mV), and mean ≈ 3.63 V (SD ≈ 120 mV) for the 100-sweep MC with evolving local states.",
        },
      ],
    },
    {
      heading: { ko: "모델 계보: Simple(논문)과 Detailed(updated accuracy)", en: "Model lineage: Simple (paper) and Detailed (updated accuracy)" },
      body: {
        ko: String.raw`시뮬레이터의 틀은 게재 확정 논문 J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi, "Analytical Model for Single Transistor Latch in MOSFETs," IEEE Electron Device Lett., 2026, doi: 10.1109/LED.2026.3737574 를 따른다.

- **Simple Model** — 논문의 식 (1)–(4), (7)과 Table I을 그대로 구현한 논문 모델이다. 바디를 한 노드로 보고 $V_{\mathrm{BS}}=V_{\mathrm{BS,curr}}+V_{\mathrm{BS,bias}}-I_{\mathrm D}R_{\mathrm{LRS}}$, $I_{\mathrm D}=M I_{\mathrm S}\exp(V_{\mathrm{BS}}/V_{\mathrm T})+I_{\mathrm{BTBT}}$로 정상상태를 푼다. 축적 상태($V_{\mathrm{FG}}<\varphi_{\mathrm{FB}}$)에서는 게이트 결합이 차폐되어 $V_{\mathrm{BS,bias}}$가 더 내려가지 않고 GIDL만 커지므로 $V_{\mathrm{LU}}(V_{\mathrm{FG}})$가 종 모양이 된다(논문 Fig. 5(a)). 확산 β는 상수로 두며 논문 식 (8)은 쓰지 않는다.
- **Detailed Model** — 같은 틀을 분포 SRH 수송, 전계 테이블, 채널 전류와 측정 보정으로 확장한 updated accuracy 모델이다. 이 물리 안내의 나머지 주제는 Detailed Model의 식을 설명한다.

Simple Model의 수식과 기본값은 \`SIMPLE_MODEL_KO.md\`에 있다.`,
        en: String.raw`The simulator's framework follows the published paper J.-H. Park, H.-B. Noh, S.-W. Lee, S.-Y. Yun, and Y.-K. Choi, "Analytical Model for Single Transistor Latch in MOSFETs," IEEE Electron Device Lett., 2026, doi: 10.1109/LED.2026.3737574.

- **Simple Model** — the paper's model as published: equations (1)–(4), (7) and Table I. The body is one node, $V_{\mathrm{BS}}=V_{\mathrm{BS,curr}}+V_{\mathrm{BS,bias}}-I_{\mathrm D}R_{\mathrm{LRS}}$, and the steady state solves $I_{\mathrm D}=M I_{\mathrm S}\exp(V_{\mathrm{BS}}/V_{\mathrm T})+I_{\mathrm{BTBT}}$. In accumulation ($V_{\mathrm{FG}}<\varphi_{\mathrm{FB}}$) the gate coupling is screened, so $V_{\mathrm{BS,bias}}$ stops falling while GIDL keeps growing, which gives the bell-shaped $V_{\mathrm{LU}}(V_{\mathrm{FG}})$ of the paper's Fig. 5(a). The diffusion β is kept constant; equation (8) is not used.
- **Detailed Model** — the updated-accuracy model that extends the same framework with distributed SRH transport, field tables, channel current and the measurement calibration. The remaining topics of this guide describe the Detailed Model's equations.

The Simple Model's equations and defaults are in \`SIMPLE_MODEL_KO.md\`.`,
      },
    },
    {
      heading: { ko: "주제 안내", en: "Topic map" },
      body: {
        ko: "- `electrostatics` — $V_T$, $V_{\\mathrm{bi}}$, 주입 수준 $\\delta(u)$, 공핍폭, 중성 길이 $L_n$, $C_{\\mathrm{ox}}$\n- `impact-ionization` — van Overstraeten–de Man 계수, 국소 전계 증배 $M(r)$, II 전류\n- `btbt-gidl` — 접합 BTBT, GIDL, $\\varphi_{\\mathrm{GIDL}}$\n- `channel` — 고정된 저 $V_D$ 채널 fit과 확장 항\n- `bjt-transport` — 전류가 흐르는 준중성 베이스의 수송, SRH, 손실, 접근 저항, $V_D$\n- `charge-balance` — Eq. 1, 정상상태, fold, 전하 지형\n- `photo` — 광생성과 광세기 변환\n- `parameters` — p[0]…p[25] 표\n- `numerics` — 격자, 이분법, 가드, FPT 창\n- 확률 모델과 회로: `stochastic-events`, `first-passage`, `local-states`, `sweep-mc`, `circuit-element`, `design-map`, `open-problems`, `validation`",
        en: "- `electrostatics` — $V_T$, $V_{\\mathrm{bi}}$, injection level $\\delta(u)$, depletion widths, neutral length $L_n$, $C_{\\mathrm{ox}}$\n- `impact-ionization` — van Overstraeten–de Man coefficients, local-field multiplication $M(r)$, II currents\n- `btbt-gidl` — junction BTBT, GIDL, $\\varphi_{\\mathrm{GIDL}}$\n- `channel` — frozen low-$V_D$ channel fit and extension terms\n- `bjt-transport` — transport in the current-carrying quasi-neutral base, SRH, losses, access resistance, $V_D$\n- `charge-balance` — Eq. 1, steady state, folds, charge landscape\n- `photo` — photogeneration and optical-power conversion\n- `parameters` — table of p[0]…p[25]\n- `numerics` — grids, bisection, guards, FPT window\n- Stochastic model and circuit: `stochastic-events`, `first-passage`, `local-states`, `sweep-mc`, `circuit-element`, `design-map`, `open-problems`, `validation`",
      },
    },
  ],
  related: [
    "electrostatics",
    "impact-ionization",
    "bjt-transport",
    "charge-balance",
    "stochastic-events",
    "first-passage",
    "local-states",
  ],
  codeRefs: [
    "photo_extension/photo_mean.py",
    "model/janus_calibration_20260920/idvd_model/mean_model.py",
    "docs/MODEL_SPEC.md",
  ],
};

export default topic;
