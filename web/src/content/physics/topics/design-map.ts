import type { PhysicsTopic } from "../types";
import { L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "design-map",
  title: L("설계 지도 (hotspot 크기 × 깊이)", "Design map (hotspot size × depth)"),
  summary: L(
    "`design_map_filled.npz`는 drain-edge hotspot의 크기 $L_0$와 깊이 비율 $d$에 대해 국소 전위 산포 $\\sigma_\\phi$와 그로 인한 $\\sigma_{V_{LU}}$, latch 비율을 준다. 생성 스크립트는 인수인계에 없으므로 아래 관계는 배열에서 수치로 역산한 것(추정)이며, 모두 기계 정밀도로 일치한다.",
    "`design_map_filled.npz` gives the local-potential spread $\\sigma_\\phi$ of a drain-edge hotspot of size $L_0$ and depth fraction $d$, the resulting $\\sigma_{V_{LU}}$ and the latched fraction. The generating script is not in the handoff, so the relations below are inferred numerically from the arrays (all agree to machine precision).",
  ),
  tags: ["design", "inferred", "local state"],
  sections: [
    {
      heading: L("파일 내용 (주어진 값)", "File contents (given)"),
      variables: [
        { symbol: r`L_0`, name: L("hotspot 길이, 81점 로그 간격", "hotspot length, 81 log-spaced points"), value: "3 … 100", unit: "nm", code: "length_nm" },
        { symbol: r`d`, name: L("깊이 비율, 61점 선형", "depth fraction, 61 linear points"), value: "0 … 0.85", code: "depth_fraction" },
        { symbol: r`\sigma_\phi`, name: L("국소 전위 SD (61×81)", "local-potential SD (61×81)"), value: "4.9 … 1090", unit: "mV", code: "sigma_phi_mV" },
        { symbol: r`\sigma_{V_{LU}}`, name: L("V_LU SD, 0→4 V 스윕", "V_LU SD, 0→4 V sweep"), value: "10.6 … 425", unit: "mV", code: "sigma_VLU_mV" },
        { symbol: r`f_{\mathrm{latch}}`, name: L("4 V 안에 latch된 비율", "fraction latched within 4 V"), value: "0.817 … 1", code: "latched_fraction" },
        { symbol: r`\sigma_{V_{LU}}^{5.2}`, name: L("V_LU SD, 5.2 V 스윕 (검열 없음)", "V_LU SD, 5.2 V sweep (no censoring)"), value: "11.1 … 614", unit: "mV", code: "sigma_VLU_sweep5p2V_mV" },
        { symbol: r`\bar N`, name: L("기대 trap 수", "expected trap count"), value: "0.09 … 100", code: "expected_trap_count" },
        { symbol: r`N_t`, name: L("trap 면밀도(지도 기준)", "trap areal density (map reference)"), value: "1e12", unit: "cm⁻²", code: "Nt_cm2" },
        { symbol: r`\sigma_{\phi,\mathrm{dev}}`, name: L("논문 소자 σ_φG", "paper device σ_φG"), value: "153.39", unit: "mV", code: "device_sigma_phi_mV" },
        { symbol: r`\phi_{50}`, name: L("σ_VLU = 50 mV가 되는 σ_φ", "σ_φ giving σ_VLU = 50 mV"), value: "65.82", unit: "mV", code: "phi_50mV" },
        { symbol: r`L_0^{\mathrm{dev}}`, name: L("σ_φ = 153.4 mV 등고선, N_t = 1e11, 5e11, 1e12, 5e12 cm⁻²", "σ_φ = 153.4 mV iso-line at N_t = 1e11, 5e11, 1e12, 5e12 cm⁻²"), value: "6.7, 15.1, 21.3, 47.7", unit: "nm", code: "line_Nt, line_L0_device" },
        { symbol: r`L_0^{50}`, name: L("σ_VLU = 50 mV 등고선, 같은 N_t", "σ_VLU = 50 mV iso-line, same N_t"), value: "15.7, 35.1, 49.7, 111", unit: "nm", code: "line_L0_50" },
      ],
    },
    {
      heading: L("추정 관계 (수치 검증)", "Inferred relations (numerically verified)"),
      body: L(
        "해석: $L_0\\times L_0$ hotspot의 trap 수가 Poisson 요동하고(표준편차 $\\sqrt{\\bar N}$), 한 trap의 전하 $q$가 산화막 용량 $C'_{ox}L_0^2$을 통해 drain-edge 전위를 바꾼다. 깊이에 따른 결합 계수 $(1-d)/2$는 배열에서 읽은 것이며 물리적 유도는 인수인계에 없다.",
        "Interpretation: the trap count in an $L_0\\times L_0$ hotspot fluctuates as Poisson (SD $\\sqrt{\\bar N}$), and each trap charge $q$ shifts the drain-edge potential through the oxide capacitance $C'_{ox}L_0^2$. The depth coupling $(1-d)/2$ is read off the arrays; its physical derivation is not in the handoff.",
      ),
      equations: [
        {
          id: "eq-dm-traps",
          label: L("기대 trap 수 (추정, 정확)", "expected trap count (inferred, exact)"),
          tex: r`\bar N = N_t\,L_0^2`,
          note: L("$N_t=10^{12}$ cm⁻² = 0.01 nm⁻²; 깊이와 무관.", "$N_t=10^{12}$ cm⁻² = 0.01 nm⁻²; independent of depth."),
          code: "design_map_filled.npz (inferred)",
        },
        {
          id: "eq-dm-sigma-phi",
          label: L("국소 전위 산포 (추정, 비 0.5000000000000001)", "local-potential spread (inferred, ratio 0.5000000000000001)"),
          tex: r`\begin{aligned} &\sigma_\phi = \frac{1-d}{2}\,\frac{q\sqrt{\bar N}}{C'_{ox}L_0^2} = \frac{(1-d)\,q\sqrt{N_t}}{2\,C'_{ox}\,L_0}\\ &C'_{ox} = \frac{3.9\,\varepsilon_0}{\mathrm{EOT}} = 2.449\times10^{-3}\ \mathrm{F/m^2}\end{aligned}`,
          note: L(
            "$\\sigma_\\phi L_0 = 3271.04\\,(1-d)$ mV·nm가 모든 $L_0$에서 상수. 논문 소자 값 153.39 mV는 $d=0$, $N_t=10^{12}$에서 $L_0=21.3$ nm에 해당.",
            "$\\sigma_\\phi L_0 = 3271.04\\,(1-d)$ mV·nm is constant over $L_0$. The paper device's 153.39 mV corresponds to $L_0=21.3$ nm at $d=0$, $N_t=10^{12}$.",
          ),
          code: "design_map_filled.npz (inferred); mean_model.py EOT_M, EPS0",
        },
        {
          id: "eq-dm-response",
          label: L("응답 곡선 (추정, 정확)", "response curves (inferred, exact)"),
          tex: r`\sigma_{V_{LU}} = \mathcal P_{\sigma}\big(\sigma_\phi\big),\qquad f_{\mathrm{latch}} = \mathcal P_{f}\big(\sigma_\phi\big),\qquad \mathcal P_{\sigma}(\phi_{50}) = 50\ \mathrm{mV}`,
          note: L(
            "$\\mathcal P$: `fill_response_sweep4V.json`(σ_φ 0…1100 mV, 20점, 각 1000 cycle, 4 V 스윕)에 대한 PCHIP; $f=$ switched/1000. $\\phi_{50}$은 PCHIP의 근(선형 보간은 65.805로 불일치).",
            "$\\mathcal P$: PCHIP through `fill_response_sweep4V.json` (σ_φ 0…1100 mV, 20 points, 1000 cycles each, 4 V sweep); $f=$ switched/1000. $\\phi_{50}$ is the PCHIP root (linear interpolation would give 65.805, which does not match).",
          ),
          code: "data/fill_response_sweep4V.json (inferred use)",
        },
        {
          id: "eq-dm-isolines",
          label: L("등고선 (d = 0, 추정, 정확)", "iso-lines (d = 0, inferred, exact)"),
          tex: r`L_0^{\mathrm{dev}}(N_t) = \frac{q\sqrt{N_t}}{2\,C'_{ox}\,\sigma_{\phi,\mathrm{dev}}},\qquad L_0^{50}(N_t) = \frac{q\sqrt{N_t}}{2\,C'_{ox}\,\phi_{50}}`,
          code: "line_L0_device, line_L0_50 (inferred)",
        },
      ],
      notes: [
        L(
          "주어진 값(재구성 불가): `sigma_VLU_sweep5p2V_mV`는 σ_φ의 단조 함수이지만 그 응답 곡선은 없다. 채움 응답 곡선 자체도 생성 코드가 없다; `gate_dynamic_compare.simulate(amplitude_scale)`로는 재현되지 않는다(σ_φ = 153.4 mV에서 111.5 mV 대 120–126 mV).",
          "Given (not reconstructible): `sigma_VLU_sweep5p2V_mV` is a monotone function of σ_φ but its response curve is not included. The fill response itself has no generating code; `gate_dynamic_compare.simulate(amplitude_scale)` does not reproduce it (111.5 mV vs 120–126 mV at σ_φ = 153.4 mV).",
        ),
        L(
          "σ_φ = 0에서의 8.74 mV는 10 mV 판독이 포함된 캐리어 잡음이다(연속 FPT 8.03 mV).",
          "The 8.74 mV at σ_φ = 0 is carrier noise including the 10 mV readout (continuous FPT: 8.03 mV).",
        ),
      ],
    },
    {
      heading: L("지도 읽는 법", "How to read the map"),
      body: L(
        "- $(L_0,d)$ → $\\sigma_\\phi$ → $\\sigma_{V_{LU}}$: 작은 hotspot과 얕은 깊이일수록 산포가 크다.\n- $\\sigma_{V_{LU}}$가 커지면 4 V 스윕 안에서 latch되지 않는 사이클(검열)이 늘어 4 V 지도는 포화된다(최대 425 mV). 5.2 V 지도는 검열이 없다.\n- 논문 소자(153.4 mV)와 50 mV 목표선은 $N_t$마다 다른 $L_0$ 등고선이다: 목표 $\\sigma_{V_{LU}}\\le50$ mV에는 $d=0$에서 $L_0\\ge 49.7$ nm ($N_t=10^{12}$)가 필요하다.",
        "- $(L_0,d)$ → $\\sigma_\\phi$ → $\\sigma_{V_{LU}}$: smaller and shallower hotspots give larger spreads.\n- As $\\sigma_{V_{LU}}$ grows, more cycles fail to latch within the 4 V sweep (censoring) and the 4 V map saturates (max 425 mV); the 5.2 V map has no censoring.\n- The paper device (153.4 mV) and the 50 mV target are $L_0$ iso-lines that move with $N_t$: reaching $\\sigma_{V_{LU}}\\le50$ mV needs $L_0\\ge 49.7$ nm at $d=0$ ($N_t=10^{12}$).",
      ),
      notes: [
        L(
          "지도는 σ_φ를 주는 가설적 trap 통계 모델 위에 논문 소자의 응답을 얹은 것이다(σ_φ = 0의 평균 3.646 V가 V_G = −2 V, 0.4 V/s FPT와 일치하므로 그 조건으로 추정); 다른 V_G나 빛에서는 응답 곡선이 달라진다.",
          "The map combines a hypothetical trap-statistics model for σ_φ with the paper device's response (inferred to be V_G = −2 V, 0.4 V/s, since the σ_φ = 0 mean 3.646 V matches that FPT node); the response curve differs at other V_G or illumination.",
        ),
      ],
    },
  ],
  related: ["local-states", "sweep-mc", "validation", "parameters"],
  codeRefs: ["data/design_map_filled.npz", "data/fill_response_sweep4V.json", "server/compute/data.py"],
};

export default topic;
