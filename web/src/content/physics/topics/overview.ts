// Physics topic "overview" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "overview",
  title: { ko: "STL 개요: 원리, 소자, 상태변수", en: "STL overview: principle, device, state variables" },
  summary: {
    ko: String.raw`단일 트랜지스터 래치(STL)는 floating-body SOI n-MOSFET에서 기생 n-p-n 바이폴라 작용과 drain 접합의 impact ionization(충돌 이온화)이 만드는 양의 되먹임 때문에 S자형 I–V와 히스테리시스(latch-up $V_{\mathrm{LU}}$, latch-down $V_{\mathrm{LD}}$)를 보인다. 모델은 두 내부 상태 $u$, $r$로 모든 전류와 $V_D$를 계산한다.`,
    en: String.raw`The single-transistor latch (STL) is a floating-body SOI n-MOSFET whose parasitic n-p-n bipolar action and drain-junction impact ionisation form a positive feedback loop, giving an S-shaped I–V with hysteresis between latch-up $V_{\mathrm{LU}}$ and latch-down $V_{\mathrm{LD}}$. The model computes every current and $V_D$ from two internal states $u$ and $r$.`,
  },
  tags: ["Eq. 1", "deterministic", "stochastic"],
  sections: [
    {
      heading: { ko: "동작 원리: 양의 되먹임", en: "Operating principle: positive feedback" },
      body: {
        ko: String.raw`게이트를 음으로 바이어스해 채널을 끈 상태에서 $V_D$를 올리면 다음 고리가 닫힌다.

- body에 정공이 쌓이면 source–body 접합이 $u$만큼 순방향 바이어스된다.
- source(emitter)가 주입한 전자가 중성 body(base)를 지나 drain 공핍층(collector)에 도달한다: BJT seed 전류 $I_{\mathrm{seed}}$.
- drain 공핍층에서 전자가 증배되어 $(M-1)I_{\mathrm{seed}}$의 정공이 body로 돌아온다.
- GIDL, 접합 BTBT, 광생성 $I_{\mathrm{PH}}$가 정공을 더하고, emitter 확산, 접합 SRH, body SRH 재결합이 정공을 뺀다.

생성이 손실을 이기면 $u$가 커지고 seed가 지수적으로 늘어 저저항 상태(LRS)로 넘어간다. 정상상태 해의 궤적 $V_D(u)$는 극대($V_{\mathrm{LU}}$)와 극소($V_{\mathrm{LD}}$)를 갖는 S자 곡선이며, 그 사이가 히스테리시스 창이다.`,
        en: String.raw`With the gate biased negative (channel off), raising $V_D$ closes the following loop.

- Holes stored in the body forward-bias the source–body junction by $u$.
- Electrons injected by the source (emitter) cross the neutral body (base) and reach the drain depletion region (collector): the BJT seed current $I_{\mathrm{seed}}$.
- In the drain depletion region the electrons multiply and return $(M-1)I_{\mathrm{seed}}$ holes to the body.
- GIDL, junction BTBT and photogeneration $I_{\mathrm{PH}}$ add holes; emitter diffusion, junction SRH and body SRH recombination remove them.

When generation beats loss, $u$ grows, the seed grows exponentially and the device jumps to the low-resistance state (LRS). The steady-state locus $V_D(u)$ is S-shaped with a maximum ($V_{\mathrm{LU}}$) and a minimum ($V_{\mathrm{LD}}$); the hysteresis window lies between them.`,
      },
      equations: [
        {
          id: "eq-ov-eq1",
          label: { ko: "Eq. 1 — body 정공 전하 보존", en: "Eq. 1 — body hole-charge balance" },
          tex: String.raw`\frac{dQ_B}{dt} = I_{\mathrm{II}} + I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{PH}} - I_{\mathrm{REC}} - I_{\mathrm{DIFF}} \equiv F(u, r;\, V_G, I_{\mathrm{PH}})`,
          note: {
            ko: "각 항의 코드 대응은 `charge-balance` 주제에 있다. 정상상태 $F=0$이 I–V 가지를 준다.",
            en: "The code mapping of every term is in the `charge-balance` topic. The steady state $F=0$ gives the I–V branches.",
          },
          code: "photo_mean.py · components() → net (z[2])",
        },
        {
          id: "eq-ov-folds",
          label: { ko: "Fold 전압", en: "Fold voltages" },
          tex: String.raw`\begin{aligned} V_{\mathrm{LU}} &= V_D(u_i),\quad i = \text{first local maximum of } V_D(u)\\ V_{\mathrm{LD}} &= V_D(u_j),\quad j = \text{last local minimum after } i \end{aligned}`,
          note: {
            ko: String.raw`논문 소자($V_G=-2$ V, 암조건) 계산값: $V_{\mathrm{LU}} = 3.7037$ V, $V_{\mathrm{LD}} = 2.5979$ V.`,
            en: String.raw`Computed for the paper device ($V_G=-2$ V, dark): $V_{\mathrm{LU}} = 3.7037$ V, $V_{\mathrm{LD}} = 2.5979$ V.`,
          },
          code: "photo_mean.py · FastModel.classify()",
        },
      ],
    },
    {
      heading: { ko: "소자와 상수", en: "Device and constants" },
      body: {
        ko: "코드는 기하 상수를 m 단위(`LENGTH_M` 등)로 두고 정전기·수송 계산에서 ×100 하여 cm 단위로 쓴다. 전류는 A, 전위는 V, 전계는 V/cm, 밀도는 cm⁻³이다. 아래 값은 논문(paper) 소자 보정값이며, 광조사(photo) 소자는 같은 보정값에 local-state 중심과 body 결합 $\\gamma$만 다시 맞춘 것이다.",
        en: "The code keeps geometry in metres (`LENGTH_M`, …) and multiplies by 100 to work in cm for electrostatics and transport. Currents are in A, potentials in V, fields in V/cm, densities in cm⁻³. The values below are the paper-device calibration; the photo device reuses it and only re-fits the local-state centre and the body coupling $\\gamma$.",
      },
      variables: [
        {
          symbol: "L",
          name: { ko: "채널(body) 길이", en: "Channel (body) length" },
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
          name: { ko: "body 억셉터 농도 (보정)", en: "Body acceptor density (calibrated)" },
          value: "2.2958 × 10¹⁷",
          unit: "cm⁻³",
          code: "refit_3.json · NA_cm3",
        },
        {
          symbol: "N_D",
          name: { ko: "source/drain 도너 농도 (가정)", en: "Source/drain donor density (assumed)" },
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
          name: { ko: "접합(emitter/collector) 단면적", en: "Junction (emitter/collector) cross-section" },
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
          name: { ko: "base 전자/정공 이동도 (상수)", en: "Base electron/hole mobility (constant)" },
          value: "450 / 150",
          unit: "cm²/(V·s)",
          code: "DN = 450·VT, DP = 150·VT",
        },
      ],
    },
    {
      heading: { ko: "상태변수와 단자 관계", en: "State variables and terminal relation" },
      body: {
        ko: "$u$는 source–body 접합의 quasi-Fermi 분리(순방향 바이어스), $r$은 drain 접합의 역바이어스다. $(u, r)$이 주어지면 `components(u, r, p, …)`가 19개 출력(V_D, I_D, F, seed, 손실, BTBT, GIDL, 주입 수준, 길이, $R_{\\mathrm{acc}}$, 전하, 공핍폭, 채널, 정공 강하, $I_{\\mathrm{PH}}$)을 계산한다. 정상상태 모델에서 $V_G$는 채널 전류와 GIDL 전계(및 선택적 국소 경로 $\\kappa_F$ 항)로만 들어간다.",
        en: "$u$ is the quasi-Fermi splitting (forward bias) of the source–body junction and $r$ the reverse bias of the drain junction. Given $(u, r)$, `components(u, r, p, …)` returns 19 outputs (V_D, I_D, F, seed, losses, BTBT, GIDL, injection level, length, $R_{\\mathrm{acc}}$, charge, depletion widths, channel, hole drop, $I_{\\mathrm{PH}}$). In the steady-state model $V_G$ enters only through the channel current and the GIDL field (and the optional local-path $\\kappa_F$ term).",
      },
      equations: [
        {
          id: "eq-ov-terminal",
          label: { ko: "단자 전압", en: "Terminal voltage" },
          tex: String.raw`V_D = u + r + V_T\,\Delta_p + (R_c + R_{\mathrm{acc}})\,I_D`,
          note: {
            ko: "$V_T\\Delta_p$: 중성 base의 정공 quasi-Fermi 강하(전계 적분이 아님), $R_c$ = p[3], $R_{\\mathrm{acc}}$: 과잉 캐리어로 변조되는 access 저항. 유도는 `bjt-transport`.",
            en: "$V_T\\Delta_p$: hole quasi-Fermi drop across the neutral base (not an electrostatic field integral), $R_c$ = p[3], $R_{\\mathrm{acc}}$: access resistance modulated by the excess carriers. Derivation in `bjt-transport`.",
          },
          code: "photo_mean.py · components() (vd)",
        },
        {
          id: "eq-ov-id",
          label: { ko: "drain 전류", en: "Drain current" },
          tex: String.raw`I_D = I_{\mathrm{seed}} + (M-1)\,I_{\mathrm{seed}} + I_{\mathrm{BTBT}} + I_{\mathrm{GIDL}} + I_{\mathrm{ch}} + I_{\mathrm{II,ch}} + I_{\mathrm{PH}} + I_{\mathrm{II,PH}} + I_{\mathrm{loc}}`,
          code: "photo_mean.py · components() (drain)",
        },
      ],
      variables: [
        {
          symbol: "u",
          name: { ko: "source–body quasi-Fermi 분리", en: "Source–body quasi-Fermi splitting" },
          unit: "V",
          code: "branch[:,17]",
        },
        {
          symbol: "r",
          name: { ko: "drain 접합 역바이어스", en: "Drain-junction reverse bias" },
          unit: "V",
          code: "branch[:,18]",
        },
        {
          symbol: String.raw`V_T\Delta_p`,
          name: { ko: "base 정공 quasi-Fermi 강하", en: "Base hole quasi-Fermi drop" },
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
          name: { ko: "access 저항", en: "Access resistance" },
          value: "4.395 × 10⁵ (n̄ = 0)",
          unit: "Ω",
          code: "z[12]",
        },
        {
          symbol: "V_G",
          name: { ko: "게이트 전압", en: "Gate voltage" },
          value: "−2 (paper), −1.8 (photo)",
          unit: "V",
          code: "p[11]",
        },
      ],
      notes: [
        {
          ko: "$u$는 quasi-Fermi/주입 대리변수이며 풀어낸 정전위가 아니다(`mean_model.py` docstring). 정전기적 body 전위는 $\\psi = u - V_T\\ln(1+\\delta/N_A)$로 따로 정의된다(`electrostatics`).",
          en: "$u$ is a quasi-Fermi/injection proxy, not a solved electrostatic potential (`mean_model.py` docstring). The electrostatic body potential is defined separately as $\\psi = u - V_T\\ln(1+\\delta/N_A)$ (`electrostatics`).",
        },
        {
          ko: String.raw`2D Poisson 해는 없다. 두 접합은 1D abrupt-junction 저주입 정전기로 다루고, 게이트는 채널 전류·GIDL·전하 좌표($C_{\mathrm{ox}}$)로만 결합된다.`,
          en: String.raw`There is no 2D Poisson solution. Both junctions use 1D abrupt-junction low-injection electrostatics; the gate couples only through the channel current, GIDL and the charge coordinate ($C_{\mathrm{ox}}$).`,
        },
      ],
    },
    {
      heading: { ko: "Deterministic과 Stochastic의 의미", en: "What deterministic and stochastic mean here" },
      body: {
        ko: "**Deterministic(결정론)**: Eq. 1의 정상상태 $F = 0$과 Kirchhoff 관계로 I–V 가지(HRS/불안정/LRS)와 fold 전압을 구한다. 준정적 스윕은 fold에서 정확히 점프한다(`charge-balance`).\n\n**Stochastic(확률론)**: 정공이 이산 사건으로 들어오고 나간다(Eq. 2, $\\Delta Q_B = q\\sum_i s_i\\,\\Delta N_i$). GIDL·BTBT·광생성·재결합·확산은 Poisson unit 사건, II는 클러스터 사건이다(`stochastic-events`). 그래서 저전류 상태가 fold 이전에 탈출(first passage)할 수 있어 $V_{\\mathrm{LU}}$는 fold보다 낮은 분포를 갖는다(`first-passage`). 여기에 drain-edge/source-edge local state $\\delta\\varphi_G$, $\\delta\\varphi_E$가 cycle마다(frozen) 또는 시간에 따라(OU) 요동해 cycle-to-cycle 산포를 만든다(`local-states`, `sweep-mc`).",
        en: "**Deterministic**: the steady state $F = 0$ of Eq. 1 plus Kirchhoff's law gives the I–V branches (HRS/unstable/LRS) and the fold voltages; a quasi-static sweep jumps exactly at the folds (`charge-balance`).\n\n**Stochastic**: holes arrive and leave as discrete events (Eq. 2, $\\Delta Q_B = q\\sum_i s_i\\,\\Delta N_i$). GIDL, BTBT, photogeneration, recombination and diffusion are Poisson unit events; impact ionisation produces clusters (`stochastic-events`). The low-current state can therefore escape before the fold (first passage), so $V_{\\mathrm{LU}}$ is distributed below the fold (`first-passage`). On top, the drain-edge/source-edge local states $\\delta\\varphi_G$, $\\delta\\varphi_E$ fluctuate per cycle (frozen) or in time (OU), giving the cycle-to-cycle spread (`local-states`, `sweep-mc`).",
      },
      notes: [
        {
          ko: "`VALIDATION.md` 기준: $V_G=-2$ V에서 fold 3.7037 V, 중심 상태의 FPT 평균 ≈ 3.644 V(SD ≈ 8 mV), 진화하는 local state를 넣은 100-sweep MC 평균 ≈ 3.63 V(SD ≈ 120 mV).",
          en: "Per `VALIDATION.md`: at $V_G=-2$ V the fold is 3.7037 V, the FPT mean at centre states ≈ 3.644 V (SD ≈ 8 mV), and the 100-sweep MC with evolving local states gives mean ≈ 3.63 V (SD ≈ 120 mV).",
        },
      ],
    },
    {
      heading: { ko: "주제 안내", en: "Topic map" },
      body: {
        ko: "- `electrostatics` — $V_T$, $V_{\\mathrm{bi}}$, 주입 수준 $\\delta(u)$, 공핍폭, 중성 길이 $L_n$, $C_{\\mathrm{ox}}$\n- `impact-ionization` — van Overstraeten–de Man 계수, 국소장 증배 $M(r)$, II 전류\n- `btbt-gidl` — 접합 BTBT, GIDL, $\\varphi_{\\mathrm{GIDL}}$\n- `channel` — 저 $V_D$ 채널 fit과 확장항\n- `bjt-transport` — 전류운반 준중성 base 수송, SRH, 손실, access 저항, $V_D$\n- `charge-balance` — Eq. 1, 정상상태, fold, 전하 지형\n- `photo` — 광생성과 광세기 변환\n- `parameters` — p[0]…p[25] 표\n- `numerics` — 격자, 이분법, 가드, FPT 창\n- 확률·회로: `stochastic-events`, `first-passage`, `local-states`, `sweep-mc`, `circuit-element`, `design-map`, `open-problems`, `validation`",
        en: "- `electrostatics` — $V_T$, $V_{\\mathrm{bi}}$, injection level $\\delta(u)$, depletion widths, neutral length $L_n$, $C_{\\mathrm{ox}}$\n- `impact-ionization` — van Overstraeten–de Man coefficients, local-field multiplication $M(r)$, II currents\n- `btbt-gidl` — junction BTBT, GIDL, $\\varphi_{\\mathrm{GIDL}}$\n- `channel` — frozen low-$V_D$ channel fit and extension terms\n- `bjt-transport` — current-carrying quasi-neutral base transport, SRH, losses, access resistance, $V_D$\n- `charge-balance` — Eq. 1, steady state, folds, charge landscape\n- `photo` — photogeneration and power conversion\n- `parameters` — table of p[0]…p[25]\n- `numerics` — grids, bisection, guards, FPT window\n- Stochastic and circuit: `stochastic-events`, `first-passage`, `local-states`, `sweep-mc`, `circuit-element`, `design-map`, `open-problems`, `validation`",
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
