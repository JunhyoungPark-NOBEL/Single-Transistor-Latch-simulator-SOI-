import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "open-problems",
  title: L("미해결 문제와 선택 옵션", "Open problems and options"),
  summary: L(
    "아직 답이 정해지지 않은 세 가지 문제 — 광조사 보정에서 −1.1 V의 채널 시드, 국소 상태의 작용점, 스윕 사이에 남는 정공 — 는 코드가 답을 강요하지 않도록 모두 선택 옵션으로 둔다. 기본값은 확장 없는 기본 모델이다.",
    "Three questions that are still open — the channel seed at −1.1 V in the illumination calibration, the action point of the local state, and residual holes between sweeps — are all exposed as options, so the code never forces an answer. The default is the base model without extensions.",
  ),
  tags: ["open problem", "options", "extensions"],
  sections: [
    {
      heading: L("1. −1.1 V의 채널 시드", "1. Channel seed at −1.1 V"),
      body: L(
        "기본 모델의 채널(V_D = 0.05 V의 I_D–V_G에 맞춘 fit)만으로는 광조사 보정의 V_G = −1.1 V 암조건 fold가 4.356 V로, 측정 평균 3.408 V보다 훨씬 높다(δφ_G0 = +0.0744 V로 실행). 두 가지 확장이 이 차이를 메우며, 어느 쪽이 맞는지는 아직 정해지지 않았다.\n\n- **none** (기본 모델)\n- **바디 결합** γ = p[15] = 0.2794 (−1.1 V 암조건 평균으로 설정, 광조사 보정 프리셋의 기본값): fold 3.408 V, −1.8 V에는 영향 없음(3.804 V)\n- **고 V_D 채널 시드** p[17] = 1.33 pA, p[18] = 0.8 V/dec: fold 3.419 V, −1.8 V의 fold는 3.750 V로 이동",
        "With the base-model channel alone (fitted to I_D–V_G at V_D = 0.05 V), the V_G = −1.1 V dark fold of the illumination calibration lies at 4.356 V, far above the measured mean of 3.408 V (run at δφ_G0 = +0.0744 V). Two extensions close the gap; which one is right is not yet settled.\n\n- **none** (base model)\n- **body coupling** γ = p[15] = 0.2794 (set by the −1.1 V dark mean; default of the illumination-calibration preset): fold 3.408 V, no effect at −1.8 V (3.804 V)\n- **high-V_D channel seed** p[17] = 1.33 pA, p[18] = 0.8 V/dec: fold 3.419 V; moves the −1.8 V fold to 3.750 V",
      ),
      equations: [
        {
          id: "eq-op-channel",
          label: L("채널 확장 (p[14]–p[16])", "Channel extensions (p[14]–p[16])"),
          tex: r`V_{ov} = V_G - V_{T0} + p_{14}\,(u+r) + p_{15}\,u,\qquad n = n_0\,\big(1 + p_{16}\,(u+r)\big)`,
          note: L(
            "$V_{T0}=-0.4903$ V, $n_0=1.7787$. $p_{14}$는 DIBL, $p_{15}$는 바디 결합, $p_{16}$은 기울기 감쇠다(모두 0이면 기본 모델).",
            "$V_{T0}=-0.4903$ V, $n_0=1.7787$. $p_{14}$: DIBL, $p_{15}$: body coupling, $p_{16}$: slope degradation (all 0 = base model).",
          ),
          code: "photo_mean.py · components(): n, ov",
        },
        {
          id: "eq-op-seed",
          label: L("고 V_D 채널 시드 (p[17], p[18])", "High-V_D channel seed (p[17], p[18])"),
          tex: r`I_{\mathrm{ch}} \leftarrow I_{\mathrm{ch}} + p_{17}\,10^{(V_G+1.8)/p_{18}}\qquad (p_{17}>0)`,
          note: L(
            "드레인에서 채널 전자에 더해지므로 증배되어 II 정공을 만든다(플로팅 바디 DIBL 가설).",
            "It is added to the channel electrons at the drain, so it is multiplied and creates II holes (floating-body DIBL hypothesis).",
          ),
          code: "photo_mean.py · components(): ch",
        },
      ],
      notes: [
        L(
          "UI: 파라미터 그룹 '모델 확장 · 미해결 문제'의 채널 시드 선택(none | body coupling | high-V_D seed), DIBL p[14], κ p[16] (`params.CHANNEL_SEED_OPTIONS`).",
          "UI: the channel-seed selector in the 'Model extensions · open problems' group (none | body coupling | high-V_D seed), DIBL p[14], κ p[16] (`params.CHANNEL_SEED_OPTIONS`).",
        ),
      ],
    },
    {
      heading: L("2. 국소 상태의 작용점", "2. Action point of the local state"),
      body: L(
        "기본 모델에서는 드레인 가장자리 상태가 GIDL 전계(p[9])에 작용한다. 광조사 측정 기록에서는 빛을 쪼여도 σ가 유지되는데(−1.8 V: 0/1.15/2.55/3.51 mW에서 173/177/119/134 mV), 이를 설명하려면 국소 애벌랜치 경로(p[21]–p[25])가 필요했다. `hypotheses.py`는 −1.8 V 암조건에서 각 레버를 보정한 뒤 8개 조건을 예측한다: H_G(GIDL), H_J(접합 전위 p[19]), H_M((M−1) 스케일 p[20]), H_LOC(시드 전류에 작용하는 국소 경로, 포화 20 pA), H_LOCB(벌크 캐리어에 작용, a = 0.3, 포화 10 pA). `verify_hloc.py`는 −1.1 V 암조건의 SD 54 mV로 $\\kappa_F$(p[25])를 정한다.",
        "In the base model the drain-edge state acts on the GIDL field (p[9]). In the illumination records σ persists under light (−1.8 V: 173/177/119/134 mV at 0/1.15/2.55/3.51 mW), which required a local avalanche path (p[21]–p[25]). `hypotheses.py` calibrates each lever at −1.8 V dark and predicts the eight conditions: H_G (GIDL), H_J (junction potential p[19]), H_M ((M−1) scale p[20]), H_LOC (local path acting on the seed currents, 20 pA saturation) and H_LOCB (acting on the bulk carriers, a = 0.3, 10 pA saturation). `verify_hloc.py` sets $\\kappa_F$ (p[25]) from the −1.1 V dark SD of 54 mV.",
      ),
      notes: [
        L(
          "UI: 확률 계산 그룹의 `action` = gidl(기본값) | local_avalanche(p[23]; p[21] ≤ 0이면 서버가 p[21] = 1.0을 씀) | junction(p[19]) | multiplication(p[20]). GIDL 이외의 선택에는 '실험적 가설' 경고가 붙는다. 식은 local-states에 있다.",
          "UI: `action` in the stochastic group = gidl (default) | local_avalanche (p[23]; the server uses p[21] = 1.0 when it is ≤ 0) | junction (p[19]) | multiplication (p[20]). Non-GIDL choices carry an 'experimental hypothesis' warning. The equations are in local-states.",
        ),
        L(
          "주의: p[24] = 2('채널을 뺀 가장자리 정의')는 코드에서 p[24] = 1(벌크)과 똑같이 동작한다(local-states 참조).",
          "Caveat: p[24] = 2 ('edge definition without the channel') behaves exactly like p[24] = 1 (bulk) in the code (see local-states).",
        ),
      ],
    },
    {
      heading: L("3. 잔류 정공 (바디 기억)", "3. Residual holes (body memory)"),
      body: L(
        "소자 시뮬레이터의 스윕 MC와 FPT는 모든 스윕이 비워진 바디(HRS 우물)에서 시작한다고 가정한다(`independent_cycles_only`, `trap_memory_included = False`). 사이클 간 상관은 국소 상태(OU 과정, 추세)를 통해서만 들어온다.\n\n회로 시뮬레이터는 $Q_B$를 끊김 없이 적분하므로, 펄스나 스윕 사이에 남은 정공이 다음 사건에 자동으로 반영된다(펄스 간격 기억). 따로 켜는 옵션은 없으며, 펄스 벤치의 간격을 바꿔 가며 확인할 수 있다.",
        "The device simulator's sweep MC and FPT assume that every sweep starts from an emptied body (HRS well) (`independent_cycles_only`, `trap_memory_included = False`). Correlation between cycles enters only through the local states (OU processes, trend).\n\nThe circuit simulator integrates $Q_B$ continuously, so holes left over between pulses or sweeps automatically carry into the next event (pulse-interval memory). There is no separate switch; vary the interval in the pulse bench to see the effect.",
      ),
      equations: [
        {
          id: "eq-op-memory",
          label: L("회로에서의 연속 적분", "Continuous integration in the circuit"),
          tex: r`Q_B(t_{k+1}) = Q_B(t_k) + \int_{t_k}^{t_{k+1}} F\,dt + \Delta Q_{\mathrm{noise}}\qquad \text{(no reset between pulses)}`,
          code: "server/compute/circuit/ (see circuit-element)",
        },
      ],
    },
  ],
  related: ["local-states", "channel", "photo", "circuit-element", "parameters", "sweep-mc"],
  codeRefs: [CODE.photoMean, CODE.params, CODE.hypotheses, CODE.verifyHloc, CODE.mcCycles, CODE.luFpt, CODE.ldFpt],
};

export default topic;
