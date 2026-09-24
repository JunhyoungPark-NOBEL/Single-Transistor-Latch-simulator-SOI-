import type { PhysicsTopic } from "../types";
import { CODE, L } from "./_sharedB";

const r = String.raw;

const topic: PhysicsTopic = {
  id: "open-problems",
  title: L("열린 문제와 UI 옵션", "Open problems and UI options"),
  summary: L(
    "인수인계에서 답이 정해지지 않은 세 가지 — 광조사 소자의 −1.1 V 채널 seed, 국소 상태의 작용점, 스윕 사이의 잔류 정공 — 는 코드가 답을 강요하지 않도록 모두 선택 옵션으로 둔다. 기본값은 논문 모델이다.",
    "Three questions the handoff leaves open — the channel seed of the photo device at −1.1 V, the action point of the local state, and residual holes between sweeps — are all exposed as options so the code never forces an answer. The default is the paper model.",
  ),
  tags: ["open problem", "options", "extensions"],
  sections: [
    {
      heading: L("1. −1.1 V 채널 seed", "1. Channel seed at −1.1 V"),
      body: L(
        "논문 채널(V_D = 0.05 V I_D–V_G 적합)만으로는 광조사 소자의 V_G = −1.1 V 암조건 fold가 4.356 V로, 측정 평균 3.408 V보다 훨씬 높다(δφ_G0 = +0.0744 V에서 실행). 두 확장이 이를 메운다; 값은 미확정이다.\n\n- **none** (논문 모델)\n- **body 결합** γ = p[15] = 0.2794 (−1.1 V 암조건 평균으로 설정, photo 프리셋 기본값): fold 3.408 V, −1.8 V에서는 영향 없음(3.804 V)\n- **고 V_D 채널 seed** p[17] = 1.33 pA, p[18] = 0.8 V/dec: fold 3.419 V, −1.8 V fold는 3.750 V로 이동",
        "The paper channel alone (fitted to I_D–V_G at V_D = 0.05 V) puts the photo device's V_G = −1.1 V dark fold at 4.356 V, far above the measured mean 3.408 V (run at δφ_G0 = +0.0744 V). Two extensions close the gap; their values are not settled.\n\n- **none** (paper model)\n- **body coupling** γ = p[15] = 0.2794 (set by the −1.1 V dark mean, photo preset default): fold 3.408 V, no effect at −1.8 V (3.804 V)\n- **high-V_D channel seed** p[17] = 1.33 pA, p[18] = 0.8 V/dec: fold 3.419 V, moves the −1.8 V fold to 3.750 V",
      ),
      equations: [
        {
          id: "eq-op-channel",
          label: L("채널 확장 (p[14]–p[16])", "channel extensions (p[14]–p[16])"),
          tex: r`V_{ov} = V_G - V_{T0} + p_{14}\,(u+r) + p_{15}\,u,\qquad n = n_0\,\big(1 + p_{16}\,(u+r)\big)`,
          note: L(
            "$V_{T0}=-0.4903$ V, $n_0=1.7787$. $p_{14}$: DIBL, $p_{15}$: body 결합, $p_{16}$: 기울기 열화(모두 0 = 논문).",
            "$V_{T0}=-0.4903$ V, $n_0=1.7787$. $p_{14}$: DIBL, $p_{15}$: body coupling, $p_{16}$: slope degradation (all 0 = paper).",
          ),
          code: "photo_mean.py · components(): n, ov",
        },
        {
          id: "eq-op-seed",
          label: L("고 V_D 채널 seed (p[17], p[18])", "high-V_D channel seed (p[17], p[18])"),
          tex: r`I_{\mathrm{ch}} \leftarrow I_{\mathrm{ch}} + p_{17}\,10^{(V_G+1.8)/p_{18}}\qquad (p_{17}>0)`,
          note: L(
            "드레인에서 채널 전자에 더해지므로 증배되어 II 정공을 만든다(floating-body DIBL 가설).",
            "Added to the channel electrons at the drain, so it is multiplied and creates II holes (floating-body DIBL hypothesis).",
          ),
          code: "photo_mean.py · components(): ch",
        },
      ],
      notes: [
        L(
          "UI: 파라미터 그룹 '모델 확장 / 열린 문제'의 채널 seed 선택(none | body coupling | high-V_D seed), DIBL p[14], κ p[16] (`params.CHANNEL_SEED_OPTIONS`).",
          "UI: channel-seed selector in the 'Model extensions / open problems' group (none | body coupling | high-V_D seed), DIBL p[14], κ p[16] (`params.CHANNEL_SEED_OPTIONS`).",
        ),
      ],
    },
    {
      heading: L("2. 국소 상태의 작용점", "2. Action point of the local state"),
      body: L(
        "논문은 drain-edge 상태가 GIDL 전계에 작용한다고 본다(p[9]). 광조사 소자에서는 빛 아래에서도 σ가 유지되며(−1.8 V: 173/177/119/134 mV, 0/1.15/2.55/3.51 mW), 이를 위해 국소 애벌랜치 경로(p[21]–p[25])가 필요했다. `hypotheses.py`는 −1.8 V 암조건에서 각 레버를 보정한 뒤 8개 조건을 예측한다: H_G(GIDL), H_J(접합 전위 p[19]), H_M((M−1) 스케일 p[20]), H_LOC(seed 위 국소 경로, 포화 20 pA), H_LOCB(벌크 캐리어 위, a = 0.3, 포화 10 pA). `verify_hloc.py`는 −1.1 V 암조건 SD 54 mV로 $\\kappa_F$ (p[25])를 정한다.",
        "The paper lets the drain-edge state act on the GIDL field (p[9]). On the photo device σ persists under light (−1.8 V: 173/177/119/134 mV at 0/1.15/2.55/3.51 mW), which required a local avalanche path (p[21]–p[25]). `hypotheses.py` calibrates each lever at −1.8 V dark and predicts the eight conditions: H_G (GIDL), H_J (junction potential p[19]), H_M ((M−1) scale p[20]), H_LOC (local path on the seeds, 20 pA saturation), H_LOCB (on bulk carriers, a = 0.3, 10 pA saturation). `verify_hloc.py` sets $\\kappa_F$ (p[25]) from the −1.1 V dark SD of 54 mV.",
      ),
      notes: [
        L(
          "UI: 확률 그룹의 `action` = gidl (기본) | local_avalanche (p[23], p[21] ≤ 0이면 서버가 1.0 사용) | junction (p[19]) | multiplication (p[20]). 비-GIDL 선택에는 '실험적 가설' 경고가 붙는다. 식은 local-states 참조.",
          "UI: `action` in the stochastic group = gidl (default) | local_avalanche (p[23]; the server uses p[21] = 1.0 when it is ≤ 0) | junction (p[19]) | multiplication (p[20]). Non-GIDL choices carry an 'experimental hypothesis' warning. Equations in local-states.",
        ),
        L(
          "주의: p[24] = 2 ('채널 제외 edge')는 코드상 p[24] = 1(벌크)과 같다(local-states 참조).",
          "Caveat: p[24] = 2 ('edge without channel') behaves exactly like p[24] = 1 (bulk) in the code (see local-states).",
        ),
      ],
    },
    {
      heading: L("3. 잔류 정공 (body 기억)", "3. Residual holes (body memory)"),
      body: L(
        "소자 시뮬레이터의 스윕 MC와 FPT는 각 스윕이 비워진 body(HRS well)에서 시작한다고 가정한다(`independent_cycles_only`, `trap_memory_included = False`). 사이클 간 상관은 국소 상태(OU, 추세)로만 들어온다.\n\n회로 시뮬레이터는 $Q_B$를 연속으로 적분하므로 펄스·스윕 사이에 남은 정공이 다음 사건에 자동으로 반영된다(펄스 간격 기억). 따로 켜는 옵션은 없고, 펄스 벤치의 간격을 바꿔 확인한다.",
        "The device simulator's sweep MC and FPT assume every sweep starts from an emptied body (HRS well) (`independent_cycles_only`, `trap_memory_included = False`). Cycle-to-cycle correlation enters only through the local states (OU, trend).\n\nThe circuit simulator integrates $Q_B$ continuously, so holes left between pulses or sweeps carry into the next event automatically (pulse-interval memory). There is no separate switch; vary the interval in the pulse bench to see it.",
      ),
      equations: [
        {
          id: "eq-op-memory",
          label: L("회로에서의 연속 적분", "continuous integration in the circuit"),
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
