// Physics topic "parameters" (physics-content A). Generated from a reviewed source;
// every equation mirrors the engine code named in its `code` field (engine/ is the reference).
import type { PhysicsTopic } from "../types";

const topic: PhysicsTopic = {
  id: "parameters",
  title: { ko: "파라미터 표 p[0]…p[25]", en: "Parameter table p[0]…p[25]" },
  summary: {
    ko: "`components()`가 받는 26개 파라미터 벡터의 의미, 보정값(JSON 원본 값), 단위, 사용 주제; local-state 진폭·동역학과 광조사 소자 보정값.",
    en: "Meaning, calibrated value (exact from the JSON sources), unit and topic of each of the 26 entries of the parameter vector passed to `components()`; plus local-state amplitudes, kinetics and the photo-device calibration.",
  },
  tags: ["p[0..25]", "calibration"],
  sections: [
    {
      heading: { ko: "보정된 평균 파라미터 p[0]–p[12]", en: "Calibrated mean parameters p[0]–p[12]" },
      body: {
        ko: "p[0]–p[8]은 `refit_3.json`(= `data/tables/parameters.json`), p[9], p[10]은 `gate_dynamic_calibration.json`에서 온다. 이름 뒤 괄호는 그 파라미터를 쓰는 주제, 코드 열은 인덱스와 웹 JSON 키다.",
        en: "p[0]–p[8] come from `refit_3.json` (= `data/tables/parameters.json`), p[9], p[10] from `gate_dynamic_calibration.json`. The topic using each parameter is given in brackets; the code column shows the index and the web JSON key.",
      },
      variables: [
        {
          symbol: String.raw`\beta`,
          name: {
            ko: "확산비: L_ref에서의 저주입 I_e,inj/I_h,out (bjt-transport)",
            en: "Diffusion ratio: low-injection I_e,inj/I_h,out at L_ref (bjt-transport)",
          },
          value: "7.166501",
          unit: "–",
          code: "p[0] · calib.beta",
        },
        {
          symbol: String.raw`\tau_{\mathrm{bulk}} = \tau_n`,
          name: { ko: "body SRH 전자 수명 (bjt-transport)", en: "Body SRH electron lifetime (bjt-transport)" },
          value: "9.266284 × 10⁻⁷ (0.9266 µs)",
          unit: "s",
          code: "p[1] · calib.tau_bulk_s",
        },
        {
          symbol: String.raw`\tau_j`,
          name: { ko: "source 접합 SRH 수명 (bjt-transport)", en: "Source-junction SRH lifetime (bjt-transport)" },
          value: "5.358373 × 10⁻⁹ (5.358 ns)",
          unit: "s",
          code: "p[2] · calib.tau_junction_s",
        },
        {
          symbol: "R_c",
          name: { ko: "접촉 저항 (bjt-transport)", en: "Contact resistance (bjt-transport)" },
          value: "1.000000",
          unit: "Ω",
          code: "p[3] · calib.r_contact_ohm",
        },
        {
          symbol: String.raw`l_{\mathrm{GIDL}}`,
          name: { ko: "유효 GIDL 전계 길이 (btbt-gidl)", en: "Effective GIDL field length (btbt-gidl)" },
          value: "28.75439",
          unit: "nm",
          code: "p[4] · calib.l_gidl_nm",
        },
        {
          symbol: String.raw`t_{\mathrm{acc}}`,
          name: { ko: "access slab 두께 (bjt-transport)", en: "Access-slab thickness (bjt-transport)" },
          value: "3.312684",
          unit: "nm",
          code: "p[5] · calib.t_access_nm",
        },
        {
          symbol: String.raw`N_{A,\mathrm{acc}}`,
          name: { ko: "access 억셉터 농도 (bjt-transport)", en: "Access acceptor density (bjt-transport)" },
          value: "1.000240 × 10¹⁷",
          unit: "cm⁻³",
          code: "p[6] · calib.na_access_cm3",
        },
        {
          symbol: String.raw`L_{\mathrm{acc}}`,
          name: { ko: "access 길이 (bjt-transport)", en: "Access length (bjt-transport)" },
          value: "70.0",
          unit: "nm",
          code: "p[7] · calib.l_access_nm",
        },
        {
          symbol: String.raw`\tau_p/\tau_n`,
          name: { ko: "SRH 수명비 (bjt-transport)", en: "SRH lifetime ratio (bjt-transport)" },
          value: "117.4247",
          unit: "–",
          code: "p[8] · calib.tau_ratio",
        },
        {
          symbol: String.raw`\varphi_{\mathrm{GIDL}}`,
          name: {
            ko: "drain-edge 상태 평균 오프셋 (btbt-gidl, local-states)",
            en: "Drain-edge state mean offset (btbt-gidl, local-states)",
          },
          value: "2.721230 × 10⁻⁴ (0.2721 mV)",
          unit: "V",
          code: "p[9] · calib.phi_gidl0_V (+ state + δφ_G)",
        },
        {
          symbol: String.raw`\varphi_E`,
          name: {
            ko: "emitter 상태 평균 오프셋 (bjt-transport, local-states)",
            en: "Emitter state mean offset (bjt-transport, local-states)",
          },
          value: "5.908461 × 10⁻⁵ (0.05908 mV)",
          unit: "V",
          code: "p[10] · calib.phi_emitter0_V (+ state + δφ_E)",
        },
        {
          symbol: "V_G",
          name: { ko: "게이트 전압 (channel, btbt-gidl)", en: "Gate voltage (channel, btbt-gidl)" },
          value: "−2.0 paper / −1.8 photo",
          unit: "V",
          code: "p[11] · vg",
        },
        {
          symbol: "p_{12}",
          name: { ko: "채널 II 배율 (impact-ionization)", en: "Channel-II scale (impact-ionization)" },
          value: "1.0",
          unit: "–",
          code: "p[12] · calib.channel_ii_scale",
        },
      ],
    },
    {
      heading: { ko: "확장 파라미터 p[13]–p[25]", en: "Extension parameters p[13]–p[25]" },
      body: {
        ko: "모든 확장이 중립값(아래 기본값)이면 논문 모델(`gate_mean.py`)과 1e-12 V 이내로 같다(`VALIDATION.md`). UI에서 pA로 입력하는 값은 ×10⁻¹² 하여 A로 넣는다.",
        en: "With every extension at its neutral value (defaults below) the model equals the paper model (`gate_mean.py`) to within 1e-12 V (`VALIDATION.md`). Values entered in pA in the UI are multiplied by 10⁻¹² to give A.",
      },
      variables: [
        {
          symbol: String.raw`I_{\mathrm{PH}}`,
          name: { ko: "광생성 전류 (photo)", en: "Photogeneration current (photo)" },
          value: "0 (photo preset: R·P)",
          unit: "A",
          code: "p[13] · light",
        },
        {
          symbol: String.raw`\eta`,
          name: { ko: "DIBL (channel)", en: "DIBL (channel)" },
          value: "0",
          unit: "V/V",
          code: "p[14] · ext.dibl",
        },
        {
          symbol: String.raw`\gamma`,
          name: { ko: "body→채널 결합 (channel)", en: "Body-to-channel coupling (channel)" },
          value: "0 (photo preset 0.2794239)",
          unit: "V/V",
          code: "p[15] · ext.gamma",
        },
        {
          symbol: String.raw`\kappa`,
          name: { ko: "기울기 감쇠 (channel)", en: "Slope degradation (channel)" },
          value: "0",
          unit: "1/V",
          code: "p[16] · ext.kappa",
        },
        {
          symbol: "I_p",
          name: { ko: "고 V_D 채널 seed, V_G = −1.8 V 기준 (channel)", en: "High-V_D channel seed at V_G = −1.8 V (channel)" },
          value: "0 (option 1.33 pA)",
          unit: "A",
          code: "p[17] · ext.seed_ip_pA",
        },
        {
          symbol: "S",
          name: { ko: "seed 기울기 (channel)", en: "Seed slope (channel)" },
          value: "1.0 (option 0.8)",
          unit: "V/dec",
          code: "p[18] · ext.seed_S",
        },
        {
          symbol: String.raw`\delta\varphi_J`,
          name: {
            ko: "접합 전위 오프셋 (impact-ionization, btbt-gidl)",
            en: "Junction potential offset (impact-ionization, btbt-gidl)",
          },
          value: "0",
          unit: "V",
          code: "p[19] · ext.dj",
        },
        {
          symbol: String.raw`\ln s_M`,
          name: { ko: "(M−1)의 로그 스케일 (impact-ionization)", en: "Log scale of (M−1) (impact-ionization)" },
          value: "0",
          unit: "–",
          code: "p[20] · ext.dm",
        },
        {
          symbol: String.raw`a_{\mathrm{loc}}`,
          name: { ko: "국소 avalanche 세기 (impact-ionization)", en: "Local avalanche strength (impact-ionization)" },
          value: "0",
          unit: "–",
          code: "p[21] · ext.aloc",
        },
        {
          symbol: String.raw`I_{\mathrm{sat}}`,
          name: { ko: "국소 경로 포화 (impact-ionization)", en: "Local-path saturation (impact-ionization)" },
          value: "2 × 10⁻¹¹ (20 pA)",
          unit: "A",
          code: "p[22] · ext.isat_pA",
        },
        {
          symbol: String.raw`d_{\mathrm{loc}}`,
          name: {
            ko: "국소 경로 로그 요동 (impact-ionization, local-states)",
            en: "Local-path log fluctuation (impact-ionization, local-states)",
          },
          value: "0",
          unit: "–",
          code: "p[23] · ext.dloc",
        },
        {
          symbol: String.raw`c_{\mathrm{loc}}`,
          name: {
            ko: "국소 경로 캐리어 정의 0/1/2 (impact-ionization)",
            en: "Local-path carrier definition 0/1/2 (impact-ionization)",
          },
          value: "0",
          unit: "–",
          code: "p[24] · ext.loc_carriers",
        },
        {
          symbol: String.raw`\kappa_F`,
          name: { ko: "국소 경로 전계 의존 (impact-ionization)", en: "Local-path field dependence (impact-ionization)" },
          value: "0",
          unit: "1/V",
          code: "p[25] · ext.kappaF",
        },
      ],
      notes: [
        {
          ko: "p[24] = 2('채널 제외 edge')는 코드에서 p[24] = 1(bulk)과 똑같이 동작한다: `bulk = p[24] > 0.5`가 먼저 참이 되어 `edge_only` 분기에 도달하지 않는다(`impact-ionization`).",
          en: "p[24] = 2 ('edge excluding channel') behaves exactly like p[24] = 1 (bulk) in the code: `bulk = p[24] > 0.5` is already true, so the `edge_only` branch is never reached (`impact-ionization`).",
        },
        {
          ko: String.raw`p[17] > 0이면 p[18]로 나누므로 $S \ne 0$이어야 한다.`,
          en: String.raw`When p[17] > 0 the code divides by p[18], so $S \ne 0$ is required.`,
        },
      ],
    },
    {
      heading: { ko: "코드에 고정된 상수", en: "Hard-coded constants" },
      variables: [
        {
          symbol: "N_A",
          name: { ko: "body 도핑 (refit)", en: "Body doping (refit)" },
          value: "2.295773 × 10¹⁷",
          unit: "cm⁻³",
          code: "refit_3.json · NA_cm3",
        },
        {
          symbol: String.raw`V_{T0},\ n,\ \beta_0,\ \theta`,
          name: { ko: "채널 fit (channel)", en: "Channel fit (channel)" },
          value: "−0.4903252, 1.778668, 7.521352 × 10⁻⁵ A/V², 0.6335606 V⁻¹",
          code: "photo_mean.py",
        },
        {
          symbol: String.raw`A_{\mathrm{BB}},\ B_{\mathrm{BB}}`,
          name: { ko: "BTBT 계수 (btbt-gidl)", en: "BTBT coefficients (btbt-gidl)" },
          value: "4 × 10¹⁴, 1.9 × 10⁷ V/cm",
          code: "BB_A, BB_B",
        },
        {
          symbol: String.raw`\mu_n,\ \mu_p`,
          name: { ko: "base 이동도 (bjt-transport)", en: "Base mobilities (bjt-transport)" },
          value: "450, 150",
          unit: "cm²/(V·s)",
          code: "DN, DP",
        },
        {
          symbol: String.raw`0.3,\ 1.12`,
          name: { ko: "GIDL 전계 오프셋, 밴드갭 (btbt-gidl)", en: "GIDL field offset, band gap (btbt-gidl)" },
          value: "0.3, 1.12",
          unit: "V",
          code: "components() eg",
        },
        {
          symbol: String.raw`7\times10^{19},\ 5\ \mathrm{nm}`,
          name: { ko: "GIDL 깊이 도핑, 활성 길이 (btbt-gidl)", en: "GIDL depth doping, active length (btbt-gidl)" },
          value: "7 × 10¹⁹ cm⁻³, 5 nm",
          code: "components() depth, volume",
        },
        {
          symbol: "5.6",
          name: { ko: "κ_F 기준 V_GD (impact-ionization)", en: "κ_F reference V_GD (impact-ionization)" },
          value: "5.6",
          unit: "V",
          code: "components() fdep",
        },
      ],
    },
    {
      heading: { ko: "Local-state 진폭과 동역학 (논문 소자)", en: "Local-state amplitudes and kinetics (paper device)" },
      variables: [
        {
          symbol: String.raw`\sigma_{\varphi G}`,
          name: { ko: "drain-edge 상태 SD (p[9]에 작용)", en: "Drain-edge state SD (acts on p[9])" },
          value: "0.1533904",
          unit: "V",
          code: "SD_GIDL_phi_V · sigma_phi_G_V",
        },
        {
          symbol: String.raw`\sigma_{\varphi E}`,
          name: { ko: "source-edge(emitter) 상태 SD (p[10]에 작용)", en: "Source-edge (emitter) state SD (acts on p[10])" },
          value: "4.369612 × 10⁻⁴ (0.4370 mV)",
          unit: "V",
          code: "SD_emitter_phi_mV · sigma_phi_E_V",
        },
        {
          symbol: String.raw`\tau_E`,
          name: { ko: "emitter OU 상관시간 (fast)", en: "Emitter OU correlation time (fast)" },
          value: "1.620556",
          unit: "s",
          code: "kinetic_fit.tau_fast_s",
        },
        {
          symbol: String.raw`\tau_G`,
          name: { ko: "drain-edge OU, 스윕 내 잔차", en: "Drain-edge OU, within-sweep residual" },
          value: "5.0",
          unit: "s",
          code: "kinetic_fit.up_residual_tau_s",
        },
        {
          symbol: String.raw`\tau_{\mathrm{slow}},\ f_{\mathrm{slow}}`,
          name: { ko: "느린 성분 (사실상 0)", en: "Slow component (effectively zero)" },
          value: "1000 s, 5.56 × 10⁻¹⁴",
          code: "kinetic_fit.slow_tau_s, fraction_slow",
        },
        {
          symbol: String.raw`dV_{\mathrm{LD}}/d\varphi_E`,
          name: { ko: "LD 사건 민감도", en: "LD event sensitivity" },
          value: "−41.17 (JSON); −39.8 (computed ±0.1 mV)",
          unit: "V/V",
          code: "kinetic_fit.LD_event_derivatives[1]",
        },
        {
          symbol: String.raw`dV_{\mathrm{LU}}/d\varphi_G`,
          name: { ko: "fold 민감도, V_G = −2 V", en: "Fold sensitivity, V_G = −2 V" },
          value: "−0.795 (computed ±10 mV)",
          unit: "V/V",
          code: "FastModel.classify",
        },
      ],
    },
    {
      heading: { ko: "광조사 소자 보정 (V_G = −1.8 V 암조건)", en: "Photo-device calibration (V_G = −1.8 V, dark)" },
      body: {
        ko: String.raw`목표: 400 cycle, 평균 3.8061 V, SD 173.2 mV, lag-1 0.199. 확인(계산값): photo preset의 −1.1 V 암조건 fold 3.4079 V가 측정 평균 3.408 V와 맞는다($\gamma$의 역할).`,
        en: String.raw`Target: 400 cycles, mean 3.8061 V, SD 173.2 mV, lag-1 0.199. Check (computed): the photo-preset fold at −1.1 V dark, 3.4079 V, matches the measured mean 3.408 V (the role of $\gamma$).`,
      },
      equations: [
        {
          id: "eq-par-c2c",
          label: { ko: "fold 모멘트 맞춤 (frozen 상태)", en: "Fold-moment matching (frozen states)" },
          tex: "\\begin{aligned} \\mathbb{E}_\\xi\\big[V_{\\mathrm{fold}}(\\delta\\varphi_{G0} + \\sigma\\xi)\\big] &= 3.8061\\ \\mathrm{V}\\\\ \\mathrm{SD}_\\xi\\big[V_{\\mathrm{fold}}(\\delta\\varphi_{G0} + \\sigma\\xi)\\big] &= 173.2\\ \\mathrm{mV},\\qquad \\xi \\sim \\mathcal{N}(0, 1) \\end{aligned}",
          note: {
            ko: "fold 표 $\\delta\\varphi \\in [-1.5, 2.0]$ V(0.05 V 간격, PCHIP), 40점 Gauss–Hermite, 래치 범위 밖 노드는 버리고 가중치 재정규화, `fsolve`. 1200 V/s 램프가 상태 동역학보다 훨씬 빠르므로 frozen 상태에서 $V_{\\mathrm{LU}} \\approx$ fold.",
            en: "Fold table $\\delta\\varphi \\in [-1.5, 2.0]$ V (0.05 V steps, PCHIP), 40-node Gauss–Hermite, nodes outside the latch range dropped and weights renormalised, `fsolve`. The 1200 V/s ramp is much faster than the state kinetics, so with frozen states $V_{\\mathrm{LU}} \\approx$ fold.",
          },
          code: "photo_extension/calibrate.py",
        },
      ],
      variables: [
        {
          symbol: String.raw`\delta\varphi_{G0}`,
          name: { ko: "drain-edge 상태 중심", en: "Drain-edge state centre" },
          value: "+0.07443209",
          unit: "V",
          code: "c2c_calibration_m18_dark.json · delta_phi_G0_V",
        },
        {
          symbol: String.raw`\sigma`,
          name: { ko: "drain-edge 상태 SD (photo)", en: "Drain-edge state SD (photo)" },
          value: "0.2153646",
          unit: "V",
          code: "c2c_calibration_m18_dark.json · sigma_phi_V",
        },
        {
          symbol: String.raw`\gamma`,
          name: { ko: "body 결합 (−1.1 V 암 평균으로 설정)", en: "Body coupling (set by the −1.1 V dark mean)" },
          value: "0.2794239",
          unit: "V/V",
          code: "gamma_probe.txt",
        },
        {
          symbol: "R",
          name: { ko: "광 변환 (photo)", en: "Optical conversion (photo)" },
          value: "0.75",
          unit: "pA/mW",
          code: "photo_conversion_fit.json",
        },
      ],
    },
    {
      heading: { ko: "UI 입력 → p 벡터", en: "UI inputs → p vector" },
      equations: [
        {
          id: "eq-par-state",
          label: { ko: "상태 오프셋의 합성", en: "Composition of the state offsets" },
          tex: String.raw`p_9 = \varphi_{\mathrm{GIDL},0} + \delta\varphi_{G0} + \delta\varphi_G,\qquad p_{10} = \varphi_{E,0} + \delta\varphi_{E0} + \delta\varphi_E,\qquad p_{13} = 10^{-12}\,R\,P\ \ (\text{power mode})`,
          note: {
            ko: "`device.calib`의 키 → p[0..8], p[12]; `calib.phi_gidl0_V` + `state.delta_phi_G0_V` + 요동 `dg` → p[9]; `calib.phi_emitter0_V` + `state.delta_phi_E0_V` + 요동 `de` → p[10]; `vg` → p[11]; `light` → p[13]; `device.ext`의 키 → p[14..25].",
            en: "`device.calib` keys → p[0..8], p[12]; `calib.phi_gidl0_V` + `state.delta_phi_G0_V` + deviation `dg` → p[9]; `calib.phi_emitter0_V` + `state.delta_phi_E0_V` + deviation `de` → p[10]; `vg` → p[11]; `light` → p[13]; `device.ext` keys → p[14..25].",
          },
          code: "server/params.py · build_p(), iph_A()",
        },
      ],
      notes: [
        {
          ko: "p[0..8]은 C2C가 큰 한 소자에 대한 조건부 fit이다(`MODEL_PARAMETERS.json` scope).",
          en: "p[0..8] is a conditional fit for one high-C2C device (`MODEL_PARAMETERS.json` scope).",
        },
      ],
    },
  ],
  related: ["bjt-transport", "btbt-gidl", "channel", "impact-ionization", "local-states", "photo", "open-problems"],
  codeRefs: [
    "data/tables/parameters.json",
    "model/MODEL_PARAMETERS.json",
    "photo_extension/setup_photo.py",
    "photo_extension/calibrate.py",
    "photo_extension/c2c_calibration_m18_dark.json",
    "server/params.py",
  ],
};

export default topic;
