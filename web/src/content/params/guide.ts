// Parameter guide shown at the top of every parameter's help: a very short, intuitive picture plus a
// three-line summary of what happens to V_LU / V_LD when the parameter is INCREASED.
//
// Every number and direction comes from ./sensitivity.json (scripts/param_sensitivity.py, real engine):
// the reference calibration ("ref": V_G = −2 V, dark, 0.4 V/s) unless the line says otherwise; the
// illumination calibration ("photo": V_G = −1.8 V, 1200 V/s) is mentioned only where it behaves differently.
// Rounding: mV to 2 significant digits, V to 2 decimals. |Δ| < 1 mV (or, for Monte-Carlo means, below
// 3 standard errors) is written as "→". guide.test.ts checks the arrows and magnitudes against the JSON.
//
// Picture used throughout: the floating body is a bucket that fills with holes. Hole sources (GIDL,
// light, impact ionization) are taps; recombination and back-diffusion are leaks. V_LU = the drain
// voltage where the bucket overflows on the way up (latch-up); V_LD = where the on-state can no longer
// hold on the way down (latch-down); window = V_LU − V_LD.
import type { L10n } from "../physics/types";

export interface ParamGuide {
  key: string;
  /** 1–2 very short sentences with an everyday picture; no equations. */
  intuitive: L10n;
  /** If you INCREASE this parameter: [V_LU change, V_LD change, one-line reason / window]. */
  effect: [L10n, L10n, L10n];
  /** Only when needed: hypothesis lever, limited range, window vanishes, non-monotonic, engine quirks. */
  caveat?: L10n;
  /** Step and preset the numbers come from (sensitivity.json). */
  basis?: string;
}

/** Legend for the guide box (how to read the arrows, and the bucket picture). */
export const GUIDE_LEGEND: { arrows: L10n; picture: L10n } = {
  arrows: {
    ko: "화살표는 이 값을 키울 때의 변화이고, 괄호는 키운 폭입니다(기준 보정: V_G = −2 V, 암조건). V_LU = 래치업 전압, V_LD = 래치다운 전압, 창 = V_LU − V_LD(히스테리시스 창).",
    en: "Arrows show the change when the value is INCREASED; brackets give the step (reference calibration: V_G = −2 V, dark). V_LU = latch-up voltage, V_LD = latch-down voltage, window = V_LU − V_LD (hysteresis window).",
  },
  picture: {
    ko: "바디는 정공이 차오르는 물통입니다. GIDL·빛·충돌 이온화는 수도꼭지, 재결합·확산은 새는 구멍이고, 물통이 넘치면 래치업됩니다.",
    en: "The body is a bucket that fills with holes: GIDL, light and impact ionization are taps; recombination and diffusion are leaks. When it overflows, the device latches up.",
  },
};

type Row = [ko: string, en: string];
type V = "V_LU" | "V_LD";

const L = ([ko, en]: Row): L10n => ({ ko, en });
const same = (v: V): Row => [`${v} → 거의 그대로 (<1 mV)`, `${v} → almost unchanged (<1 mV)`];
const zero = (v: V): Row => [`${v} → 변화 없음 (0 mV)`, `${v} → no change (0 mV)`];
const foldKept = (v: V): Row => [`${v} → 소자 fold는 그대로 (수치 설정)`, `${v} → device fold unchanged (numerics)`];

function g(key: string, intuitive: Row, effect: [Row, Row, Row], opts: { caveat?: Row; basis?: string } = {}): ParamGuide {
  const out: ParamGuide = { key, intuitive: L(intuitive), effect: [L(effect[0]), L(effect[1]), L(effect[2])] };
  if (opts.caveat) out.caveat = L(opts.caveat);
  if (opts.basis) out.basis = opts.basis;
  return out;
}

const ENTRIES: ParamGuide[] = [
  // ------------------------------------------------------------------ bias / sweep
  g(
    "vg",
    [
      "게이트는 드레인 가장자리 '정공 수도꼭지'(GIDL, 게이트 유도 드레인 누설: 게이트–드레인 전계에 의한 밴드 간 터널링)의 손잡이입니다. V_G를 0 V 쪽으로 올리면 꼭지가 잠겨 바디가 더 천천히 찹니다.",
      "The gate is the handle of the hole tap at the drain edge (GIDL, gate-induced drain leakage: band-to-band tunneling in the gate–drain field). Raising V_G toward 0 V closes the tap, so the body fills more slowly.",
    ],
    [
      ["V_LU ↑ 약 80 mV (+0.1 V)", "V_LU ↑ ≈ 80 mV (+0.1 V)"],
      same("V_LD"),
      ["GIDL 정공 공급 ↓ → 래치업이 늦어짐 · 창 1.11 → 1.19 V", "Less GIDL supply → later latch-up · window 1.11 → 1.19 V"],
    ],
    {
      caveat: [
        "V_D = 3 V로 고정하면 HRS의 GIDL은 V_G +0.1 V마다 약 0.6배로 줄지만, fold 위치가 함께 옮겨 가므로 fold에서의 GIDL은 약 2 pA로 거의 같습니다. 채널 전류도 +0.1 V마다 약 9배 늘지만 V_G = −2 V에서는 10⁻²¹ A 수준이라 그래프에서 보이지 않습니다. V_G가 약 −1.1 V보다 높으면 채널이 켜져 V_LU가 다시 내려가고(비단조), 래치 창(fold 쌍이 있는 V_G 범위)은 −3.88 ~ −0.81 V입니다.",
        "At a fixed V_D = 3 V the HRS GIDL falls ≈ 0.6× per +0.1 V of V_G, but at the fold it stays near 2 pA because the fold itself moves. The channel current also grows ≈ 9× per +0.1 V, but at V_G = −2 V it is ~10⁻²¹ A, far below the plot range. Above V_G ≈ −1.1 V the channel turns on and V_LU falls again (non-monotonic); the latch window (V_G range with a fold pair) is −3.88 to −0.81 V.",
      ],
      basis: "ref ±0.1 V at V_G = −2 V (extra points −3 … −0.9 V)",
    },
  ),
  g(
    "vd_max",
    [
      "드레인 스윕을 어디까지 올릴지 정하는 '천장'입니다. 천장이 V_LU보다 낮으면 물통이 넘치기 전에 되돌아가 래치업이 일어나지 않습니다.",
      "The 'ceiling' of the drain sweep. If it is below V_LU, the sweep turns back before the bucket overflows and the device never latches up.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["래치업되는 사이클 비율만 ↑ (3.66 V 52 % → 4 V 100 %)", "Only the latch-up fraction rises (3.66 V 52 % → 4 V 100 %)"],
    ],
    {
      caveat: [
        "확률 모드에서는 V_LU가 사이클마다 달라서, 천장이 V_LU 근처이면 일부 사이클만 래치업됩니다. 평균에는 래치업된(V_LU가 낮은) 사이클만 들어가 평균 V_LU가 낮게 보이고, 천장을 올리면 평균이 올라갑니다.",
        "In stochastic mode V_LU differs from cycle to cycle, so with a ceiling near V_LU only some cycles latch up. Only those (low-V_LU) cycles enter the mean, so the displayed mean V_LU is biased low; it rises as the ceiling is raised.",
      ],
      basis: "ref MC n = 200: latched fraction vs V_D,max",
    },
  ),
  g(
    "rate",
    [
      "드레인 전압을 올리고 내리는 속도입니다. 천천히 스윕하면 우연한 요동으로 물통이 먼저 넘칠 시간이 늘어나 조금 일찍 스위칭합니다.",
      "How fast the drain voltage is swept. A slow sweep gives a random fluctuation more time to tip the bucket early, so the device switches a little sooner.",
    ],
    [
      ["V_LU 평균 ↑ 약 16 mV (×10)", "V_LU mean ↑ ≈ 16 mV (×10)"],
      ["V_LD 평균 ↓ 약 13 mV", "V_LD mean ↓ ≈ 13 mV"],
      ["조기 탈출할 시간 ↓ → fold 가까이서 스위칭 (fold는 그대로)", "Less time to escape early → switching moves toward the fold"],
    ],
    {
      caveat: [
        "광조사 보정(1200 V/s)에서는 이미 사이클의 91 %가 fold에서 스위칭하므로 ×10에도 V_LU는 +1.3 mV만 움직입니다(V_LD −27 mV).",
        "In the illumination calibration (1200 V/s) 91 % of cycles already switch at the fold, so ×10 moves V_LU by only +1.3 mV (V_LD −27 mV).",
      ],
      basis: "ref ×10 / ×0.1 (0.4 V/s), carrier-noise hazard",
    },
  ),
  g(
    "dv",
    [
      "스윕을 몇 mV 간격으로 잘라 계산할지입니다. 간격이 촘촘할수록 정확하지만 느려집니다.",
      "The voltage step the sweep is sliced into. A finer step is more precise but slower.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 (<1 mV, 스텝 1 → 5 mV)", "V_LU mean → almost unchanged (<1 mV, step 1 → 5 mV)"],
      ["V_LD 평균 → 통계 잡음 수준 (약 3 mV)", "V_LD mean → within statistical noise (≈ 3 mV)"],
      ["수치 설정: 정확도와 속도만 바뀌고 물리는 그대로", "Numerics only: changes accuracy and speed, not the physics"],
    ],
    { basis: "ref MC n = 200: 5 mV / 1 mV step" },
  ),

  // ------------------------------------------------------------------ light
  g(
    "light",
    [
      "빛은 바디에 정공을 직접 부어 주는 '추가 수도꼭지'입니다. 빛이 강할수록 물통이 빨리 차서 더 낮은 전압에서 래치업됩니다.",
      "Light is an extra tap that pours holes straight into the body. Brighter light fills the bucket faster, so the device latches up at a lower voltage.",
    ],
    [
      ["V_LU ↓ 약 320 mV (0 → 2.55 mW)", "V_LU ↓ ≈ 320 mV (0 → 2.55 mW)"],
      ["V_LD ↓ 약 1.2 mV (거의 그대로)", "V_LD ↓ ≈ 1.2 mV (almost unchanged)"],
      ["광생성 정공(+ 광전자 증배분)이 GIDL에 더해짐 · 창 1.11 → 0.79 V", "Photo-holes (+ multiplication) add to GIDL · window 1.11 → 0.79 V"],
    ],
    {
      caveat: [
        "광조사 보정에서는 같은 빛에 V_LU가 약 370 mV 내려갑니다. I_PH가 약 20.7 pA(≈ 27.6 mW)를 넘으면 V_LU가 V_LD까지 내려와 히스테리시스 창이 닫힙니다.",
        "In the illumination calibration the same light lowers V_LU by ≈ 370 mV. Above I_PH ≈ 20.7 pA (≈ 27.6 mW) V_LU falls to V_LD and the hysteresis window closes.",
      ],
      basis: "ref 0 → 2.55 mW (I_PH 1.91 pA, R = 0.75 pA/mW)",
    },
  ),
  g(
    "iph_pA",
    [
      "광전류 I_PH는 빛이 바디에 부어 주는 정공의 양(유량)입니다. 많이 부을수록 물통이 빨리 찹니다.",
      "The photocurrent I_PH is the rate at which light pours holes into the body. More flow fills the bucket faster.",
    ],
    [
      ["V_LU ↓ 약 320 mV (0 → 1.9 pA)", "V_LU ↓ ≈ 320 mV (0 → 1.9 pA)"],
      ["V_LD ↓ 약 1.2 mV (거의 그대로)", "V_LD ↓ ≈ 1.2 mV (almost unchanged)"],
      ["+0.1 pA마다 V_LU −13 ~ −17 mV · 창 1.11 → 0.79 V", "V_LU ≈ −13 to −17 mV per +0.1 pA · window 1.11 → 0.79 V"],
    ],
    {
      caveat: ["약 20.7 pA를 넘으면 V_LU가 V_LD까지 내려와 히스테리시스 창이 닫힙니다.", "Above ≈ 20.7 pA V_LU falls to V_LD and the hysteresis window closes."],
      basis: "ref 0 → 1.91 pA; slope per +0.1 pA",
    },
  ),
  g(
    "power_mW",
    [
      "입사광의 세기입니다. 이 소자에서는 1 mW당 약 0.75 pA의 정공이 바디에 들어갑니다(응답도).",
      "The incident optical power. In this device each mW delivers ≈ 0.75 pA of holes into the body (the responsivity).",
    ],
    [
      ["V_LU ↓ 약 320 mV (0 → 2.55 mW)", "V_LU ↓ ≈ 320 mV (0 → 2.55 mW)"],
      ["V_LD ↓ 약 1.2 mV (거의 그대로)", "V_LD ↓ ≈ 1.2 mV (almost unchanged)"],
      ["광생성 정공·증배 ↑ → 물통이 빨리 차오름 · 창 1.11 → 0.79 V", "More photo-holes (+ multiplication) · window 1.11 → 0.79 V"],
    ],
    {
      caveat: [
        "약 27.6 mW(20.7 pA)를 넘으면 히스테리시스 창이 닫힙니다. 광조사 보정에서는 같은 2.55 mW에 V_LU가 약 370 mV 내려갑니다.",
        "Above ≈ 27.6 mW (20.7 pA) the hysteresis window closes. In the illumination calibration 2.55 mW lowers V_LU by ≈ 370 mV.",
      ],
      basis: "ref 0 → 2.55 mW (measured condition)",
    },
  ),
  g(
    "resp",
    [
      "응답도는 빛 1 mW가 몇 pA의 정공으로 바뀌는지 정하는 '환율'입니다. 광 파워 모드에서만 쓰입니다.",
      "Responsivity is the 'exchange rate' from mW of light to pA of hole current. It is used only in optical-power mode.",
    ],
    [
      ["V_LU ↓ 약 370 mV (×2, 2.55 mW 광조사 보정)", "V_LU ↓ ≈ 370 mV (×2 at 2.55 mW, illumination cal.)"],
      ["V_LD ↓ 약 1.2 mV (거의 그대로)", "V_LD ↓ ≈ 1.2 mV (almost unchanged)"],
      ["같은 파워에서 I_PH를 2배로 한 것과 같음 · 창 0.84 → 0.47 V", "Like doubling I_PH at the same power · window 0.84 → 0.47 V"],
    ],
    {
      caveat: ["I_PH를 직접 입력하는 모드에서는 효과가 없습니다.", "No effect in direct I_PH mode."],
      basis: "photo at 2.55 mW: I_PH ×2 (≙ R ×2)",
    },
  ),

  // ------------------------------------------------------------------ local-state means
  g(
    "dphiG0",
    [
      "드레인 가장자리 국소 상태(결함·전하)의 평균 이동량입니다. 올리면 GIDL 수도꼭지가 조금 더 열립니다.",
      "The mean shift of the local state (traps, charge) at the drain edge. Raising it opens the GIDL tap a little more.",
    ],
    [
      ["V_LU ↓ 약 8.0 mV (+10 mV)", "V_LU ↓ ≈ 8.0 mV (+10 mV)"],
      same("V_LD"),
      ["GIDL 전계 ↑ (V_G를 10 mV 낮춘 것과 같음) → 정공 공급 ↑", "Stronger GIDL field, as if V_G were 10 mV lower"],
    ],
    {
      caveat: [
        "이 값이 사이클마다 흔들리는 것이 V_LU 퍼짐의 주원인입니다 (상태 표준편차 참고).",
        "Cycle-to-cycle jitter of this value is the main source of the V_LU spread (see State SD).",
      ],
      basis: "ref ±10 mV",
    },
  ),
  g(
    "dphiE0",
    [
      "소스 가장자리 국소 상태의 평균 이동량입니다. 올리면 정공이 소스로 빠지는 '배수구'가 살짝 좁아져 켜진 상태가 더 낮은 V_D까지 버팁니다.",
      "The mean shift of the source-edge local state. Raising it narrows the outlet through which holes escape to the source, so the on-state holds to a lower V_D.",
    ],
    [
      same("V_LU"),
      ["V_LD ↓ 약 4.0 mV (+0.1 mV)", "V_LD ↓ ≈ 4.0 mV (+0.1 mV)"],
      ["소스로 새는 정공 ↓ → 켜진 상태 유지 · 1 mV당 약 40 mV", "Less hole leak to the source · very sensitive (≈ 40 mV/mV)"],
    ],
    { basis: "ref ±0.1 mV" },
  ),

  // ------------------------------------------------------------------ calibration
  g(
    "beta",
    [
      "β는 기생 BJT의 전류 이득(소스가 바디로 보내는 전자 수 ÷ 소스로 새는 바디 정공 수)입니다. 클수록 소스 쪽으로 새는 구멍이 작아 정공이 바디에 잘 남습니다.",
      "β is the parasitic BJT current gain: electrons the source injects per body hole that leaks back into the source. A larger β means a smaller leak toward the source, so holes stay in the body.",
    ],
    [
      ["V_LU ↓ 약 49 mV (×2)", "V_LU ↓ ≈ 49 mV (×2)"],
      ["V_LD ↓ 약 610 mV", "V_LD ↓ ≈ 610 mV"],
      ["켜진 상태의 주 손실(소스 역확산 ∝ 1/β) ↓ · 창 1.11 → 1.67 V", "Less back-diffusion loss (∝ 1/β) · window 1.11 → 1.67 V"],
    ],
    {
      caveat: ["약 ×0.3 아래로 줄이면 V_LD가 V_LU(약 4.1 V)까지 올라와 히스테리시스 창이 닫힙니다.", "Below ≈ ×0.3 V_LD rises to V_LU (≈ 4.1 V) and the hysteresis window closes."],
      basis: "ref ×2 / ×0.5",
    },
  ),
  g(
    "tau_bulk",
    [
      "바디 한가운데서 여분의 전자–정공 쌍이 재결합해 사라지기까지의 평균 시간(전자 수명 τ_n)입니다. 길수록 물통 바닥의 틈이 좁아지지만, 원래 아주 작은 틈입니다.",
      "The average time before an extra electron–hole pair in the middle of the body recombines (electron lifetime τ_n). A longer lifetime narrows a crack in the bucket's floor, but that crack is already tiny.",
    ],
    [
      same("V_LU"),
      same("V_LD"),
      ["fold에서 벌크 재결합은 다른 손실의 수천분의 1 이하", "At the fold, bulk recombination is >1000× below other losses"],
    ],
    { basis: "ref ×2 / ×0.5" },
  ),
  g(
    "tau_junction",
    [
      "소스 접합 근처(공핍 영역)에서 정공이 재결합해 사라지기까지의 시간입니다. 꺼진 상태에서 가장 큰 '새는 구멍'이라, 길어지면 물통이 훨씬 쉽게 찹니다.",
      "How long holes survive in the source-junction depletion region. Recombination there is the biggest off-state leak, so a longer lifetime lets the bucket fill much more easily.",
    ],
    [
      ["V_LU ↓ 약 280 mV (×2)", "V_LU ↓ ≈ 280 mV (×2)"],
      ["V_LD ↓ 약 49 mV", "V_LD ↓ ≈ 49 mV"],
      ["꺼진 상태의 주 손실(접합 재결합) ↓ → 창 1.11 → 0.87 V", "Main off-state leak shrinks · window 1.11 → 0.87 V"],
    ],
    {
      caveat: [
        "×0.5로 줄이면 V_LU(4.03 V)가 기준 스윕 최대 4 V를 넘어 래치업이 보이지 않습니다. ×0.037 아래에서는 V_LU가 약 5.7 V를 넘어 모델이 fold를 계산하지 못합니다.",
        "At ×0.5 V_LU (4.03 V) exceeds the 4 V reference sweep, so no latch-up is seen; below ×0.037 V_LU exceeds ≈ 5.7 V and the model can no longer trace the fold.",
      ],
      basis: "ref ×2 / ×0.5",
    },
  ),
  g(
    "r_contact",
    [
      "측정 단자와 소자 사이 '연결선'의 저항입니다. 전류가 흐르면 여기서 전압이 조금 떨어져 소자에 걸리는 전압이 줄어듭니다.",
      "The resistance of the 'wire' between the probe and the device. Current through it drops some voltage, so the device sees a little less.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["기본값 약 1 Ω이라 ×2는 무시 가능 · 1 MΩ이면 V_LD ↑ 13 mV", "Default ≈ 1 Ω, so ×2 is negligible · 1 MΩ: V_LD ↑ ≈ 13 mV"],
    ],
    {
      caveat: [
        "큰 켜진 상태 전류(약 17 nA)가 흐를 때만 전압 강하가 생겨 V_LD만 움직입니다.",
        "Only the large on-state current (≈ 17 nA) makes a noticeable drop, so only V_LD moves.",
      ],
      basis: "ref ×2 (≈ 1 Ω); 10 kΩ … 1 MΩ absolute",
    },
  ),
  g(
    "l_gidl",
    [
      "GIDL 전계가 퍼져 있는 '거리'입니다. 같은 전압을 더 긴 거리에 나누면 전계가 약해져 GIDL 수도꼭지가 크게 잠깁니다.",
      "The 'distance' the GIDL field is spread over. The same voltage over a longer distance makes a weaker field, which closes the GIDL tap sharply.",
    ],
    [
      ["V_LU ↑ 약 340 mV (+10 %)", "V_LU ↑ ≈ 340 mV (+10 %)"],
      same("V_LD"),
      ["GIDL이 지수적으로 ↓ → 창 1.11 → 1.45 V (+10 %)", "GIDL drops exponentially → window 1.11 → 1.45 V (+10 %)"],
    ],
    {
      caveat: [
        "가장 민감한 파라미터 중 하나입니다. +10 %만으로 V_LU(4.04 V)가 기준 스윕 4 V를 넘습니다. ×0.63 아래에서는 V_LU가 V_LD까지 내려와 히스테리시스 창이 닫히고, ×1.55 위에서는 V_LU가 약 5.6 V를 넘어 모델이 fold를 계산하지 못합니다(×2: 계산 불가, ×0.5: 창 없음).",
        "One of the most sensitive parameters: +10 % already pushes V_LU (4.04 V) past the 4 V reference sweep. Below ×0.63 V_LU falls to V_LD and the hysteresis window closes; above ×1.55 V_LU exceeds ≈ 5.6 V and the model can no longer trace the fold (×2: not computable, ×0.5: no window).",
      ],
      basis: "ref ×1.1 step (extra point; ±5 % slope 325 mV per 10 %); ×2 / ×0.5 outside the traceable window",
    },
  ),
  g(
    "t_access",
    [
      "소스·드레인으로 이어지는 '입구 통로'(접근 영역)의 두께입니다. 통로가 두꺼울수록 저항이 작아 전압 손실이 줄어듭니다.",
      "The thickness of the 'entrance corridor' (access region) to source and drain. A thicker corridor has less resistance, so less voltage is lost.",
    ],
    [
      same("V_LU"),
      ["V_LD ↓ 약 3.6 mV (×2)", "V_LD ↓ ≈ 3.6 mV (×2)"],
      ["직렬 저항 ↓ → 켜진 상태 전류의 전압 강하 ↓ (V_LD만)", "Lower series R → less IR drop in the on-state (V_LD only)"],
    ],
    { basis: "ref ×2 / ×0.5" },
  ),
  g(
    "na_access",
    [
      "입구 통로에 넣은 불순물(도핑) 농도입니다. 많을수록 캐리어가 많아져 통로 저항이 작아집니다.",
      "The impurity (doping) level in the entrance corridor. More doping means more carriers and a lower corridor resistance.",
    ],
    [
      same("V_LU"),
      ["V_LD ↓ 약 3.5 mV (×2)", "V_LD ↓ ≈ 3.5 mV (×2)"],
      ["직렬 저항 ↓ → 켜진 상태 전류의 전압 강하 ↓ (V_LD만)", "Lower series R → less IR drop in the on-state (V_LD only)"],
    ],
    { basis: "ref ×2 / ×0.5" },
  ),
  g(
    "l_access",
    [
      "입구 통로의 길이입니다. 길수록 저항이 커져 전압 손실이 늘어납니다.",
      "The length of the entrance corridor. A longer corridor has more resistance and loses more voltage.",
    ],
    [
      same("V_LU"),
      ["V_LD ↑ 약 6.3 mV (×2)", "V_LD ↑ ≈ 6.3 mV (×2)"],
      ["직렬 저항 ↑ → 켜진 상태 전류의 전압 강하 ↑ (V_LD만)", "Higher series R → more IR drop in the on-state (V_LD only)"],
    ],
    { basis: "ref ×2 / ×0.5" },
  ),
  g(
    "tau_ratio",
    [
      "정공과 전자 수명의 비(τ_p/τ_n)입니다. 바디에 캐리어가 아주 많을 때(고주입)만 재결합을 느리게 만듭니다.",
      "The hole-to-electron lifetime ratio τ_p/τ_n. It only matters when the body is flooded with carriers (high injection), where it slows recombination.",
    ],
    [
      same("V_LU"),
      same("V_LD"),
      ["벌크 재결합은 fold에서 아주 작은 손실이라 영향 없음", "Bulk recombination is a tiny loss at the fold → no effect"],
    ],
    { basis: "ref ×2 / ×0.5" },
  ),
  g(
    "phi_gidl0",
    [
      "GIDL 수도꼭지의 '영점' 보정값(드레인 가장자리 상태의 평균)입니다. 올리면 같은 V_G에서도 꼭지가 더 열립니다.",
      "The calibrated 'zero point' of the GIDL tap (the mean drain-edge state). Raising it opens the tap more at the same V_G.",
    ],
    [
      ["V_LU ↓ 약 8.0 mV (+10 mV)", "V_LU ↓ ≈ 8.0 mV (+10 mV)"],
      same("V_LD"),
      ["GIDL 전계 ↑ (V_G를 10 mV 낮춘 것과 같음) → 정공 공급 ↑", "Stronger GIDL field, as if V_G were 10 mV lower"],
    ],
    {
      caveat: [
        "'드레인 쪽 이동'(δφ_G0)과 같은 항에 더해지므로 효과가 똑같습니다.",
        "It enters the same term as the drain shift δφ_G0, so the effect is identical.",
      ],
      basis: "ref ±10 mV",
    },
  ),
  g(
    "phi_emitter0",
    [
      "정공이 소스로 빠지는 '배수구' 크기의 보정값(소스 가장자리 상태의 평균)입니다. 올리면 배수구가 좁아집니다.",
      "The calibrated setting of the outlet to the source (the mean source-edge state). Raising it narrows the outlet.",
    ],
    [
      same("V_LU"),
      ["V_LD ↓ 약 4.0 mV (+0.1 mV)", "V_LD ↓ ≈ 4.0 mV (+0.1 mV)"],
      ["소스로 새는 정공 ↓ → 켜진 상태 유지 · 1 mV당 약 40 mV", "Less hole leak to the source · very sensitive (≈ 40 mV/mV)"],
    ],
    {
      caveat: [
        "'소스 쪽 이동'(δφ_E0)과 같은 항에 더해지므로 효과가 똑같습니다.",
        "It enters the same term as the source shift δφ_E0, so the effect is identical.",
      ],
      basis: "ref ±0.1 mV",
    },
  ),
  g(
    "ch_ii",
    [
      "채널을 지나는 전자가 드레인 근처에서 부딪혀 새 정공을 만드는 효율(충돌 이온화)의 배율입니다. 채널 전류가 거의 없으면 키워도 소용없습니다.",
      "Scales impact ionization by channel electrons, which crash near the drain and knock loose new holes. With almost no channel current, raising it does nothing.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["fold의 채널 전류(약 10⁻²¹ A)가 GIDL보다 약 10⁹배 작음", "Channel current at the fold (≈ 10⁻²¹ A) is ~10⁹× below GIDL"],
    ],
    {
      caveat: [
        "채널이 켜지는 V_G = −1.1 V에서는 중요해집니다: ×2에 V_LU가 약 31 mV(기준) / 430 mV(광조사) 내려갑니다.",
        "It matters once the channel turns on, e.g. at V_G = −1.1 V: ×2 lowers V_LU by ≈ 31 mV (ref) / 430 mV (illumination).",
      ],
      basis: "ref & photo ×2 / ×0.5 (also 0 … 100)",
    },
  ),

  // ------------------------------------------------------------------ extensions (channel, hypothesis levers, local path)
  g(
    "dibl",
    [
      "드레인 전압이 채널 입구의 '문턱(장벽)'을 낮춰 주는 정도입니다. 클수록 드레인 전압만으로도 채널 문이 조금 열립니다.",
      "How much the drain voltage lowers the source barrier of the channel. The larger it is, the more the drain alone cracks the channel open.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["V_G = −2 V에선 채널이 깊이 꺼져 +0.05도 효과 없음", "Channel is deeply off at V_G = −2 V; even +0.05 does nothing"],
    ],
    {
      caveat: [
        "V_G = −1.1 V에서는 +0.01만으로 V_LU가 약 50 mV(기준) / 400 mV(광조사) 내려갑니다.",
        "At V_G = −1.1 V, +0.01 already lowers V_LU by ≈ 50 mV (ref) / 400 mV (illumination).",
      ],
      basis: "ref & photo +0.01 (also +0.03, +0.05)",
    },
  ),
  g(
    "gamma",
    [
      "바디에 정공이 차오를 때 채널 문이 함께 열리는 정도(바디–채널 결합)입니다. 클수록 물통이 차면서 채널이 빨리 켜집니다.",
      "How much a filling body also opens the channel door (body–channel coupling). The larger it is, the sooner the channel turns on as the bucket fills.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["V_G = −2 / −1.8 V에선 채널이 꺼져 있어 효과 없음", "Channel is off at V_G = −2 / −1.8 V → no effect"],
    ],
    {
      caveat: [
        "광조사 보정의 γ = 0.2794도 이 조건에서는 효과가 없습니다. V_G = −1.1 V(광조사 보정)에서는 ±0.05에 V_LU가 약 −400 / +430 mV로 크게 움직입니다.",
        "Even the illumination γ = 0.2794 has no effect here. At V_G = −1.1 V (illumination) ±0.05 moves V_LU by ≈ −400 / +430 mV.",
      ],
      basis: "ref & photo ±0.05",
    },
  ),
  g(
    "kappa",
    [
      "꺼진 채널 전류의 기울기가 바디·드레인 전압에 따라 완만해지는 정도입니다. 클수록 꺼진 채널에서도 전류가 조금 더 샙니다.",
      "How much the channel's subthreshold slope flattens with body and drain voltage. The larger it is, the more current leaks through the 'off' channel.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["채널 전류가 GIDL보다 훨씬 작아 효과 없음", "The channel current is far below GIDL, so no effect"],
    ],
    {
      caveat: [
        "V_G = −1.1 V에서는 +0.05에 V_LU가 약 380 mV(기준) / 660 mV(광조사) 내려갑니다.",
        "At V_G = −1.1 V, +0.05 lowers V_LU by ≈ 380 mV (ref) / 660 mV (illumination).",
      ],
      basis: "ref & photo ±0.05 1/V",
    },
  ),
  g(
    "seed_ip",
    [
      "채널로 새어 드는 '시드(씨앗)' 전자 전류입니다(모델에서는 V_D와 무관). 이 전자들이 드레인에서 증배되어(V_D가 클수록 더 많이) 정공을 만들어 물통을 채웁니다.",
      "A constant 'seed' electron current through the channel (independent of V_D in the model). Its electrons are multiplied at the drain — more so at higher V_D — creating holes that fill the bucket.",
    ],
    [
      ["V_LU ↓ 약 29 mV (0 → 1.33 pA)", "V_LU ↓ ≈ 29 mV (0 → 1.33 pA)"],
      same("V_LD"),
      ["시드 전자의 충돌 이온화로 정공 ↑ · 창 1.11 → 1.08 V", "Seed electrons add holes via II · window 1.11 → 1.08 V"],
    ],
    {
      caveat: [
        "프리셋 값은 0(끔)입니다. 광조사 보정에서는 V_LU가 약 55 mV, V_G = −1.1 V에서는 약 770 mV(기준 보정) 내려갑니다.",
        "The preset value is 0 (off). In the illumination calibration V_LU drops by ≈ 55 mV, and at V_G = −1.1 V by ≈ 770 mV (ref).",
      ],
      basis: "ref 0 → 1.33 pA ('high-V_D seed' option)",
    },
  ),
  g(
    "seed_S",
    [
      "시드 전류가 V_G에 따라 변하는 기울기(10배 바뀌는 데 필요한 V_G 폭)입니다. 클수록 V_G에 덜 민감합니다.",
      "How steeply the seed current changes with V_G (the V_G needed for a 10× change). A larger S makes it less sensitive to V_G.",
    ],
    [
      ["V_LU ↓ 약 1.7 mV (+0.1 V/dec)", "V_LU ↓ ≈ 1.7 mV (+0.1 V/dec)"],
      same("V_LD"),
      ["V_G = −2 V에서 시드 전류 약간 ↑ (−1.8 V에선 효과 없음)", "Seed current rises slightly at V_G = −2 V (none at −1.8 V)"],
    ],
    {
      caveat: [
        "시드 전류(I_p)가 0인 프리셋에서는 효과가 없습니다. V_G = −1.1 V에서는 방향이 바뀌어 V_LU가 약 130 mV(기준 보정) 올라갑니다.",
        "No effect while the seed current I_p is 0 (presets). At V_G = −1.1 V the sign flips: V_LU rises by ≈ 130 mV (ref).",
      ],
      basis: "ref ±0.1 V/dec with I_p = 1.33 pA, S = 0.8 V/dec",
    },
  ),
  g(
    "dj",
    [
      "드레인–바디 접합에 걸린 전압을 가상으로 조금 더해 보는 '실험용 손잡이'입니다. 올리면 전자가 부딪혀 정공을 만드는 증배(충돌 이온화)가 강해집니다.",
      "An 'experimental knob' that pretends a bit more voltage sits across the drain–body junction. Raising it strengthens multiplication — electrons creating extra holes.",
    ],
    [
      ["V_LU ↓ 약 9.1 mV (+0.05 V)", "V_LU ↓ ≈ 9.1 mV (+0.05 V)"],
      ["V_LD ↓ 약 46 mV", "V_LD ↓ ≈ 46 mV"],
      ["증배(M−1) ↑ → 충돌 이온화 ↑ · 창 1.11 → 1.14 V", "More multiplication (M−1) · window 1.11 → 1.14 V"],
    ],
    {
      caveat: ["보정값이 아닌 가설 검증용 레버입니다 (기본 0).", "A hypothesis-testing lever, not a calibrated value (default 0)."],
      basis: "ref ±0.05 V",
    },
  ),
  g(
    "dm",
    [
      "전자 하나가 부딪혀 만드는 추가 정공 수(증배 M−1)를 통째로 키우는 '볼륨 손잡이'입니다 (로그 단위, +1 ≈ 2.7배).",
      "A volume knob for the extra holes each electron creates by impact ionization (M−1), in log units (+1 ≈ ×2.7).",
    ],
    [
      ["V_LU ↓ 약 90 mV (+0.3)", "V_LU ↓ ≈ 90 mV (+0.3)"],
      ["V_LD ↓ 약 310 mV", "V_LD ↓ ≈ 310 mV"],
      ["충돌 이온화 전체가 약 1.35배 · 창 1.11 → 1.33 V", "All impact ionization ≈ ×1.35 · window 1.11 → 1.33 V"],
    ],
    {
      caveat: [
        "가설 검증용 레버입니다. −1.38 아래에서는 V_LU와 V_LD가 약 4.56 V에서 만나 히스테리시스 창이 닫히고, +3.0 위에서는 V_LD가 약 0.7 V 아래로 내려가 모델이 계산하지 못합니다.",
        "A hypothesis lever. Below −1.38 V_LU and V_LD meet at ≈ 4.56 V and the hysteresis window closes; above +3.0 V_LD drops below ≈ 0.7 V and the model can no longer trace it.",
      ],
      basis: "ref ±0.3",
    },
  ),
  g(
    "aloc",
    [
      "드레인 근처 작은 '약한 지점'에서 일어나는 국소 애벌랜치(전자 눈사태)의 세기입니다. 0이면 꺼져 있고, 키울수록 정공이 추가로 쏟아집니다.",
      "The strength of a local avalanche — a snowballing electron cascade — at a tiny weak spot near the drain. At 0 it is off; the larger it is, the more extra holes pour in.",
    ],
    [
      ["V_LU ↓ 약 64 mV (0 → 1)", "V_LU ↓ ≈ 64 mV (0 → 1)"],
      same("V_LD"),
      ["국소 경로가 꺼진 상태에 정공을 보탬 · 창 1.11 → 1.04 V", "The local path adds off-state holes · window 1.11 → 1.04 V"],
    ],
    {
      caveat: [
        "광조사에서도 σ가 유지되는 이유를 시험하는 가설 레버입니다 (프리셋 0). 10으로 키우면 V_LU가 약 290 mV 내려갑니다.",
        "A hypothesis lever for why σ survives illumination (preset 0). At 10, V_LU drops by ≈ 290 mV.",
      ],
      basis: "ref 0 → 1 (also 0.5, 2, 10)",
    },
  ),
  g(
    "isat",
    [
      "국소 애벌랜치 경로가 낼 수 있는 최대 전류(천장)입니다. 경로 전류가 천장에 닿기 전에는 아무 영향이 없습니다.",
      "The ceiling current of the local avalanche path. Until the path current reaches the ceiling it has no effect.",
    ],
    [
      zero("V_LU"),
      zero("V_LD"),
      ["a_loc = 1에서도 경로 전류가 포화값(20 pA)에 못 미침", "At a_loc = 1 the path stays below saturation (20 pA)"],
    ],
    {
      caveat: ["a_loc > 0일 때만 의미가 있습니다.", "Relevant only when a_loc > 0."],
      basis: "ref ×2 / ×0.5 with a_loc = 1",
    },
  ),
  g(
    "dloc",
    [
      "국소 애벌랜치 경로의 세기를 로그 단위로 조절합니다. 올리면 그 경로가 더 세집니다.",
      "Shifts the strength of the local avalanche path in log units. Raising it makes the path stronger.",
    ],
    [
      ["V_LU ↓ 약 31 mV (+0.5)", "V_LU ↓ ≈ 31 mV (+0.5)"],
      same("V_LD"),
      ["국소 경로 세기 약 1.65배 → 꺼진 상태 정공 ↑", "Local path ≈ ×1.65 stronger → more holes in the off-state"],
    ],
    {
      caveat: ["a_loc > 0일 때만 효과가 있습니다 (프리셋 a_loc = 0에서는 0 mV).", "Acts only when a_loc > 0 (0 mV at the preset a_loc = 0)."],
      basis: "ref ±0.5 with a_loc = 1",
    },
  ),
  g(
    "loc_carriers",
    [
      "국소 애벌랜치가 어떤 전자를 증배할지 고릅니다. 0은 가장자리 전자, 1은 바디를 가로지르는 BJT 전자입니다.",
      "A switch choosing which electrons the local avalanche multiplies: 0 picks edge electrons, 1 picks electrons crossing the body (BJT).",
    ],
    [
      ["V_LU ↓ 약 130 mV (0 → 1)", "V_LU ↓ ≈ 130 mV (0 → 1)"],
      ["V_LD ↓ 약 14 mV", "V_LD ↓ ≈ 14 mV"],
      ["벌크(BJT) 전자가 더 많아 국소 경로의 정공 생성 ↑", "More bulk (BJT) electrons → the local path makes more holes"],
    ],
    {
      caveat: ["a_loc > 0일 때만 효과가 있고, 2는 현재 엔진에서 1과 같습니다.", "Acts only when a_loc > 0; option 2 currently behaves like 1."],
      basis: "ref 0 → 1 with a_loc = 1",
    },
  ),
  g(
    "kappaF",
    [
      "국소 애벌랜치가 게이트–드레인 전압에 얼마나 민감한지(기울기)입니다.",
      "How sensitive the local avalanche is to the gate–drain voltage (its slope).",
    ],
    [
      ["V_LU ↓ 약 1.0 mV (+0.5 1/V)", "V_LU ↓ ≈ 1.0 mV (+0.5 1/V)"],
      same("V_LD"),
      ["fold가 기준점(V_GD = 5.6 V) 근처라 효과가 아주 작음", "The fold lies near the V_GD reference (5.6 V) → tiny effect"],
    ],
    {
      caveat: [
        "a_loc > 0일 때만 효과가 있고, 광조사 보정에서는 방향이 반대입니다(+1.7 mV).",
        "Acts only when a_loc > 0; in the illumination calibration the sign flips (+1.7 mV).",
      ],
      basis: "ref ±0.5 1/V with a_loc = 1",
    },
  ),

  // ------------------------------------------------------------------ Monte Carlo (device tab)
  g(
    "n_cycles",
    [
      "몬테카를로로 스윕을 몇 번 반복할지입니다. 많을수록 평균과 퍼짐이 더 정확해지지만 느려집니다.",
      "How many sweeps the Monte Carlo repeats. More cycles give a more accurate mean and spread, but take longer.",
    ],
    [
      ["V_LU 평균 → 물리 변화 없음 (9 mV 차이 = 통계 잡음)", "V_LU mean → no physical change (9 mV = sampling noise)"],
      ["V_LD 평균 → 물리 변화 없음 (<1 mV)", "V_LD mean → no physical change (<1 mV)"],
      ["평균의 오차만 ↓ (4배 → 절반) · 기준 보정 200회 약 ±19 mV", "Only the mean's error shrinks (×4 → half) · ≈ ±19 mV at 200"],
    ],
    {
      caveat: [
        "기준 보정은 진화 모드라 이웃 사이클끼리 상관(lag-1 0.67)이 있어, 평균의 오차가 σ/√N보다 약 2.2배 큽니다: 200회 약 ±19 mV, 2000회 약 ±6 mV.",
        "The reference calibration runs in evolving mode, so neighboring cycles are correlated (lag-1 0.67) and the error of the mean is ≈ 2.2× σ/√N: ≈ ±19 mV at 200 cycles, ±6 mV at 2000.",
      ],
      basis: "ref MC ×4 (800) / ×0.25 (50); SE = σ/√N · √((1+ρ)/(1−ρ)), lag-1 ρ = 0.67",
    },
  ),
  g(
    "seed",
    [
      "난수의 출발점입니다. 같은 시드는 언제나 같은 결과를 재현하고, 시드를 바꾸면 다른 표본을 뽑습니다.",
      "The starting point of the random numbers. The same seed always reproduces the same result; a new seed draws a different sample.",
    ],
    [
      ["V_LU 평균 → 물리 변화 없음 (시드 간 약 4.5 mV)", "V_LU mean → no physical change (≈ 4.5 mV between seeds)"],
      ["V_LD 평균 → 물리 변화 없음 (시드 간 약 2.9 mV)", "V_LD mean → no physical change (≈ 2.9 mV between seeds)"],
      ["표본만 바뀜 · 광조사 보정은 시드 간 V_LU 약 18 mV", "Only the sample changes · illumination: ≈ 18 mV apart"],
    ],
    { basis: "ref & photo, 4 seeds, n = 200" },
  ),
  g(
    "carrier_noise",
    [
      "정공이 하나씩 또는 한꺼번에 무작위로 생기는 '캐리어 잡음'(알갱이 잡음)입니다. 켜면 fold에 닿기 전에 물통이 우연히 먼저 넘칠 수 있습니다.",
      "Carrier noise: holes arrive one at a time or in random bursts, like grains. When on, the bucket can overflow by chance before the sweep reaches the fold.",
    ],
    [
      ["V_LU 평균 ↓ 약 59 mV (끔 → 켬)", "V_LU mean ↓ ≈ 59 mV (off → on)"],
      ["V_LD → 변화 없음 (0 mV, 하향은 별도 옵션)", "V_LD → no change (0 mV; the down sweep has its own option)"],
      ["우연한 조기 탈출 → 평균이 fold(3.70 V)보다 낮아짐", "Chance early escapes → mean below the fold (3.70 V)"],
    ],
    {
      caveat: [
        "광조사 보정(1200 V/s)은 램프가 빨라 차이가 약 1.5 mV뿐입니다.",
        "In the illumination calibration (1200 V/s) the ramp is fast, so the difference is only ≈ 1.5 mV.",
      ],
      basis: "ref general engine, MC n = 200, off vs on",
    },
  ),
  g(
    "ld_carrier_noise",
    [
      "내려가는 스윕에서도 캐리어 잡음 때문에 먼저 꺼지는 효과를 계산할지 정합니다. 켜면 더 정확하지만 느려집니다.",
      "Whether carrier noise is also applied on the down sweep, where it can switch the device off early. On is more accurate but slower.",
    ],
    [
      zero("V_LU"),
      ["V_LD 평균 ↑ 약 100 mV (끔 → 켬)", "V_LD mean ↑ ≈ 100 mV (off → on)"],
      ["하향 스윕에서 우연히 먼저 꺼짐 → V_LD가 fold(2.60 V)보다 위로", "Chance early turn-off → V_LD above the fold (2.60 V)"],
    ],
    {
      caveat: [
        "램프가 느릴수록 차이가 큽니다 (광조사 보정 1200 V/s: +48 mV). 기준 보정의 자동 엔진은 이 효과를 항상 포함합니다.",
        "The slower the ramp, the larger the shift (illumination, 1200 V/s: +48 mV). The reference calibration's automatic engine always includes it.",
      ],
      basis: "ref general engine, MC n = 200, off vs on",
    },
  ),
  g(
    "engine",
    [
      "MC 계산 방식입니다. '보정 lookup 표'는 기준 측정에 맞춘 빠른 표이고, '일반'은 모든 설정을 그대로 반영해 계산합니다.",
      "Chooses the MC method: 'calibrated lookup' is a fast table fitted to the reference measurement; 'general' honors every setting.",
    ],
    [
      ["V_LU → fold는 그대로 (계산 방식만 다름)", "V_LU → the fold is unchanged (method only)"],
      ["V_LD → fold는 그대로", "V_LD → the fold is unchanged"],
      ["기준 보정: 두 엔진의 평균 V_LU 차이 약 2.6 mV, σ도 비슷", "Reference: mean V_LU differs by ≈ 2.6 mV; similar σ"],
    ],
    {
      caveat: [
        "보정 lookup 표는 이미터 상태 SD를 상태 표준편차에 묶어 둡니다. 이미터 상태 SD나 이미터 상관 시간을 따로 바꾸려면 '일반'을 고르세요.",
        "The calibrated lookup ties the Emitter SD to the State SD; pick 'general' to set the Emitter SD or emitter correlation time separately.",
      ],
      basis: "ref frozen state, MC n = 200: lookup vs general",
    },
  ),
  g(
    "n_traces",
    [
      "I–V 그림에 몇 개의 사이클 곡선을 그릴지입니다. 그림만 바뀝니다.",
      "How many cycle curves to draw in the I–V plot. Only the picture changes.",
    ],
    [
      ["V_LU → 변화 없음 (표시 설정)", "V_LU → no change (display setting)"],
      ["V_LD → 변화 없음 (표시 설정)", "V_LD → no change (display setting)"],
      ["그리는 곡선 수만 바뀜 · 통계는 모든 사이클로 계산", "Only the drawn curves change; statistics use every cycle"],
    ],
    { basis: "display setting" },
  ),

  // ------------------------------------------------------------------ local state (device tab)
  g(
    "ls_mode",
    [
      "국소 상태(가장자리의 미세한 요동)를 흔드는 방식입니다. 없음, 사이클마다 한 번 뽑기(고정), 시간에 따라 천천히 변하기(진화) 중에서 고릅니다.",
      "Chooses how the local state (tiny edge fluctuations) varies: none, one draw per cycle (frozen), or drifting slowly in time (evolving).",
    ],
    [
      ["V_LU 평균 → 거의 그대로 · '없음'이면 σ 110 → 8.4 mV", "V_LU mean → almost unchanged · 'none' cuts σ 110 → 8.4 mV"],
      ["V_LD 평균 → 거의 그대로 · '없음'이면 σ 20 → 7.4 mV", "V_LD mean → almost unchanged · 'none' cuts σ 20 → 7.4 mV"],
      ["퍼짐의 대부분은 국소 상태에서 생김 · '진화'는 사이클 간 상관도 만듦", "Most spread comes from here · 'evolving' adds correlation"],
    ],
    { basis: "ref MC n = 200: frozen vs none (ls_sigma auto_engine_4V, preset view)" },
  ),
  g(
    "ls_action",
    [
      "국소 상태의 흔들림이 어느 '손잡이'에 걸리는지 고릅니다. 기본은 GIDL(드레인 가장자리)이고, 나머지는 검증되지 않은 가설입니다.",
      "Chooses which 'knob' the local-state jitter acts on. GIDL (drain edge) is the default; the others are unvalidated hypotheses.",
    ],
    [
      ["V_LU → 중심 fold는 그대로 · 퍼짐의 원인만 바뀜", "V_LU → fold unchanged · only the spread's source changes"],
      ["V_LD → 중심 fold는 그대로", "V_LD → fold unchanged"],
      ["GIDL 작용점: 상태가 1 mV 흔들리면 V_LU가 약 0.8 mV 흔들림", "GIDL: 1 mV of state jitter → ≈ 0.8 mV of V_LU jitter"],
    ],
    {
      caveat: [
        "GIDL 외 작용점은 보정되지 않아 수치가 실제 소자와 다를 수 있습니다.",
        "Action points other than GIDL are uncalibrated; numbers may not match the real device.",
      ],
      basis: "dV_LU/dφ_G from dphiG0 (ref ±10 mV)",
    },
  ),
  g(
    "ls_sigma",
    [
      "드레인 가장자리 국소 상태가 사이클마다 흔들리는 크기(표준편차)입니다. 클수록 스위칭 전압이 더 넓게 퍼집니다.",
      "How strongly the drain-edge local state jitters from cycle to cycle (its standard deviation). More jitter makes the switching voltages scatter more widely.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 · σ 약 2배 (120 → 230 mV, ×2)", "V_LU mean → about the same · σ doubles (120 → 230 mV, ×2)"],
      ["V_LD 평균 → 거의 그대로 · σ ↑ 20 → 46 mV (자동 엔진)", "V_LD mean → about the same · σ ↑ 20 → 46 mV (auto engine)"],
      ["사이클마다 GIDL 꼭지가 더 크게 흔들림 → 스위칭 전압이 넓게 퍼짐", "The GIDL tap jitters more each cycle → wider spread"],
    ],
    {
      caveat: [
        "기본 4 V 스윕(자동 엔진)에서는 ×2일 때 200사이클 중 10개(5 %)가 래치업하지 못해(중도절단) 통계에서 빠지므로, 평균이 약 27 mV 낮게, σ가 110 → 200 mV로 작게 보입니다. 6 V까지 스윕하면 평균 변화는 −6 mV(잡음 수준)입니다(광조사 보정: −7 mV). 자동 엔진은 이미터 상태 SD도 같이 2배로 만들어 V_LD도 퍼집니다(평균 약 −7.5 mV).",
        "In the default 4 V sweep (automatic engine), at ×2 10 of 200 cycles (5 %) do not latch up (censored) and drop out of the statistics, so the mean looks ≈ 27 mV lower and σ only 110 → 200 mV. Sweeping to 6 V gives a mean shift of −6 mV (noise level; illumination: −7 mV). The automatic engine also doubles the Emitter SD, so V_LD spreads too (mean ≈ −7.5 mV).",
      ],
      basis: "ref MC n = 200, frozen, ×2 / ×0.5: general engine 0 → 6 V (uncensored); σ(V_LD) and the 4 V view from auto_engine_4V",
    },
  ),
  g(
    "ls_tau",
    [
      "국소 상태가 얼마나 오래 '기억'되는지(상관 시간)입니다. 길수록 이웃한 사이클의 스위칭 전압이 서로 비슷해집니다.",
      "How long the local state is 'remembered' (correlation time). The longer it is, the closer the switching voltages of neighboring cycles.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 (+10 mV, 통계 잡음 수준)", "V_LU mean → almost unchanged (+10 mV, within noise)"],
      zero("V_LD"),
      ["이웃 사이클 상관 ↑ (lag-1 0.67 → 0.92, ×10)", "Neighboring cycles more alike (lag-1 0.67 → 0.92, ×10)"],
    ],
    {
      caveat: [
        "기록 전체가 τ보다 짧으면(광조사 보정: 200사이클 ≈ 1.7 s, τ = 5 s) 상태가 거의 움직이지 않아 σ(V_LU)가 작게 보입니다: ×10에 47 → 17 mV.",
        "If the whole record is shorter than τ (illumination: 200 cycles ≈ 1.7 s vs τ = 5 s) the state barely moves and σ(V_LU) looks small: ×10 gives 47 → 17 mV.",
      ],
      basis: "ref & photo MC n = 200, ×10 / ×0.1 (evolving)",
    },
  ),
  g(
    "ls_sigmaE",
    [
      "소스 가장자리 국소 상태가 사이클마다 흔들리는 크기입니다. 소스 쪽 '배수구'가 매번 달라져 V_LD만 퍼집니다.",
      "How much the source-edge local state jitters each cycle. The source-side outlet changes every time, so only V_LD spreads.",
    ],
    [
      same("V_LU"),
      ["V_LD 평균 → 거의 그대로 · σ ↑ 18 → 35 mV (×2)", "V_LD mean → almost unchanged · σ ↑ 18 → 35 mV (×2)"],
      ["V_LD는 소스 쪽 상태에 매우 민감 (1 mV당 약 40 mV)", "V_LD is very sensitive to the source-edge state (≈ 40 mV/mV)"],
    ],
    {
      caveat: [
        "기준 보정의 자동 엔진(보정 lookup 표)에서는 이 값이 무시되고 상태 표준편차에 묶입니다. 엔진을 '일반'으로 바꾸세요. 광조사 보정의 기본값은 0(끔)입니다.",
        "With the reference calibration's automatic engine (calibrated lookup) this value is ignored and tied to the State SD; switch the engine to 'general'. The illumination default is 0 (off).",
      ],
      basis: "ref general engine, MC n = 200, ×2 / ×0.5",
    },
  ),
  g(
    "ls_tauE",
    [
      "소스 쪽(이미터) 상태가 얼마나 오래 기억되는지(상관 시간)입니다.",
      "How long the source-side (emitter) state is remembered (correlation time).",
    ],
    [
      ["V_LU → 영향 없음 (소스 쪽 상태)", "V_LU → no effect (source-side state)"],
      ["V_LD 평균 → 거의 그대로 · 이웃 사이클 상관 ↑", "V_LD mean → almost unchanged · neighboring cycles more alike"],
      ["상태 상관 시간과 같은 원리이고 V_LD에만 작용", "Same idea as the correlation time, acting on V_LD only"],
    ],
    {
      caveat: [
        "진화 모드이면서 이미터 상태 SD > 0이고 '일반' 엔진일 때만 쓰입니다(보정 lookup 표는 보정값을 고정해 씀).",
        "Used only in evolving mode with emitter SD > 0 and the 'general' engine (the calibrated lookup keeps its own value).",
      ],
      basis: "no sweep; by analogy with ls_tau",
    },
  ),
  g(
    "ls_trend",
    [
      "기준 측정 기록(암조건, 100회 스윕)에서 V_LU가 천천히 흘러가던 추세를 넣습니다. 그 기록을 재현할 때만 켜세요.",
      "Adds the slow V_LU drift seen in the reference record (dark, 100 sweeps). Turn it on only to reproduce that record.",
    ],
    [
      ["V_LU → 중심 fold는 그대로 · 사이클 순서에 따른 느린 흐름 추가", "V_LU → fold unchanged · adds a slow drift across cycles"],
      ["V_LD → 영향 없음 (상향 스윕 추세)", "V_LD → no effect (up-sweep trend)"],
      ["이웃 사이클 상관 ↑ (기준 보정 lag-1 0.67의 주원인)", "Neighboring cycles more alike (main cause of lag-1 0.67)"],
    ],
    {
      caveat: [
        "보정 lookup 엔진의 진화 모드에서만 작동합니다 (일반 엔진에서는 무시).",
        "Works only in evolving mode with the calibrated lookup engine (ignored by the general engine).",
      ],
      basis: "ref MC: lag-1 0.67 with trend (ls_tau block)",
    },
  ),

  // ------------------------------------------------------------------ numerics (device tab)
  g(
    "grid",
    [
      "fold(스위칭 지점)를 찾을 때 쓰는 계산 눈금의 수입니다. 촘촘할수록 정밀하지만 느려집니다.",
      "The number of grid points used to find the folds (switching points). A finer grid is more precise but slower.",
    ],
    [
      ["V_LU → 거의 그대로 (<1 mV, 601 → 2001)", "V_LU → almost unchanged (<1 mV, 601 → 2001)"],
      same("V_LD"),
      ["수치 설정: 차이 0.05 mV 미만, 계산만 느려짐", "Numerics only: < 0.05 mV difference, just slower"],
    ],
    { basis: "ref 601 → 2001 / 201" },
  ),
  g(
    "fold_nodes",
    [
      "상태별 fold 표를 몇 개의 점으로 만들지입니다 (사이는 보간). 많을수록 정확하지만 느려집니다.",
      "How many points the fold-vs-state table uses (interpolated in between). More points are more accurate but slower.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 (<1 mV)", "V_LU mean → almost unchanged (<1 mV)"],
      ["V_LD 평균 → 거의 그대로 (<1 mV)", "V_LD mean → almost unchanged (<1 mV)"],
      ["수치 설정: 정확도·속도만 바뀜 · 보정 lookup 엔진은 사용 안 함", "Numerics only · not used by the calibrated lookup"],
    ],
    { basis: "ref general engine, 49 / 13 nodes" },
  ),
  g(
    "hazard_nodes",
    [
      "캐리어 잡음에 의한 탈출 확률을 계산할 때 쓰는 적분 점의 수입니다. 많을수록 정확하지만 느려집니다.",
      "The number of integration points for the carrier-noise escape probability. More points are more accurate but slower.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 (<1 mV)", "V_LU mean → almost unchanged (<1 mV)"],
      ["V_LD 평균 → 거의 그대로 (<1 mV)", "V_LD mean → almost unchanged (<1 mV)"],
      ["수치 설정: 변화 0.05 mV 미만, 계산만 느려짐", "Numerics only: < 0.05 mV change, just slower"],
    ],
    { basis: "ref general engine, 7 / 3 nodes" },
  ),

  // ------------------------------------------------------------------ circuit solver
  g(
    "method",
    [
      "회로의 시간 적분 방식입니다. BE는 안정적이고, TRAP은 더 정확하지만 급변하는 곳에서 출렁일(링잉) 수 있습니다.",
      "The circuit time-integration rule. BE is robust and stable; TRAP is more accurate but can ring at sharp jumps.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["파형의 정확도·속도만 바뀜", "Only waveform accuracy and speed change"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "dt_min",
    [
      "허용하는 가장 작은 시간 스텝입니다. 너무 크면 급격한 스위칭을 놓칠 수 있습니다.",
      "The smallest time step allowed. Too large and a sharp switching event can be missed.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["급변 구간의 정확도만 바뀜 · 비워 두면 자동", "Only accuracy at sharp transitions · leave empty for auto"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "dt_max",
    [
      "시간 스텝의 상한입니다. 작을수록 파형이 매끄럽지만 느려집니다.",
      "The largest time step allowed. A smaller cap gives smoother waveforms but slower runs.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["파형 해상도·속도만 바뀜 · 비워 두면 자동", "Only waveform resolution and speed · leave empty for auto"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "reltol",
    [
      "한 스텝에서 허용하는 변화의 크기입니다. 작을수록 정확하지만 스텝이 많아져 느려집니다.",
      "How much change one step may make. A smaller value is more accurate but needs more steps and time.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["키우면 빨라지지만 파형이 거칠어짐", "Larger = faster but coarser waveforms"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "max_steps",
    [
      "계산이 멈추기 전까지 허용하는 최대 시간 스텝 수입니다(안전장치).",
      "The maximum number of time steps before the run stops (a safety cap).",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["너무 작으면 시뮬레이션이 끝까지 못 가고 멈춤", "Too small and the run stops before the end"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "c_ith",
    [
      "드레인 전류가 이 값을 넘어 올라가면 래치업, 아래로 내려가면 래치다운으로 판정하는 기준선입니다.",
      "The drain-current line that marks a switch: crossing it upward counts as latch-up, downward as latch-down.",
    ],
    [
      ["V_LU → 소자 물리는 그대로 · 판정 시점만 바뀜", "V_LU → physics unchanged · only the detection point moves"],
      ["V_LD → 소자 물리는 그대로 · 판정 시점만 바뀜", "V_LD → physics unchanged · only the detection point moves"],
      ["HRS(약 15 pA)와 LRS(약 17 nA) fold 전류 사이에 둘 것", "Set between the fold currents: HRS ≈ 15 pA, LRS ≈ 17 nA"],
    ],
    { basis: "ref fold currents (sensitivity baseline)" },
  ),
  g(
    "tau_frac",
    [
      "확률 과도해석에서 한 스텝을 소자 반응 시간의 몇 분의 1로 제한할지입니다. 작을수록 잡음이 정확하지만 느립니다.",
      "Limits each stochastic step to a fraction of the device's response time. A smaller fraction gives more faithful noise but runs slower.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["잡음 해상도·속도만 바뀜", "Only noise resolution and speed change"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "max_ev",
    [
      "한 스텝에서 기대되는 무작위 사건 수의 상한입니다. 넘지 않도록 스텝을 줄입니다.",
      "The cap on expected random events per step; the step shrinks to stay below it.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["잡음 정확도·속도만 바뀜", "Only noise accuracy and speed change"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "noise_dt_min",
    [
      "이보다 짧은 시간 규모에서는 잡음을 하나하나 계산하지 않고 평균 흐름(드리프트)만 계산합니다.",
      "Below this time scale the noise is not resolved; only the average drift is computed.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["키우면 빨라지지만 빠른 잡음이 빠짐", "Larger = faster, but fast noise is dropped"]],
    { basis: "circuit numerics (no device effect)" },
  ),
  g(
    "gauss_th",
    [
      "평균 사건 수가 이보다 많으면 하나씩 세지 않고 종 모양(가우스) 분포로 한 번에 뽑습니다. 빠르지만 근사입니다.",
      "Above this many events, the count comes from a bell-curve (Gaussian) approximation instead of counting one by one. Faster, but approximate.",
    ],
    [foldKept("V_LU"), foldKept("V_LD"), ["잡음 계산 방식만 바뀜 (큰 사건 수에서는 차이 미미)", "Only how noise is drawn changes (negligible at large counts)"]],
    { basis: "circuit numerics (no device effect)" },
  ),

  // ------------------------------------------------------------------ circuit stochastic
  g(
    "c_runs",
    [
      "회로 과도해석을 독립적으로 몇 번 반복할지입니다. 많을수록 통계가 정확하지만 느립니다.",
      "How many independent circuit transients to run. More runs give better statistics but take longer.",
    ],
    [
      ["V_LU 평균 → 물리 변화 없음 (통계 정밀도만)", "V_LU mean → no physical change (statistics only)"],
      ["V_LD 평균 → 물리 변화 없음", "V_LD mean → no physical change"],
      ["실행 4배 → 평균 오차 절반 · 파형은 처음 8개만", "×4 runs halve the error · waveforms for the first 8 only"],
    ],
    { basis: "statistics only" },
  ),
  g(
    "c_seed",
    [
      "회로 확률 계산의 난수 출발점입니다. 같은 시드는 같은 결과를 재현합니다.",
      "The random-number starting point for circuit runs. The same seed reproduces the same result.",
    ],
    [
      ["V_LU 평균 → 물리 변화 없음 (표본만 바뀜)", "V_LU mean → no physical change (sample only)"],
      ["V_LD 평균 → 물리 변화 없음 (표본만 바뀜)", "V_LD mean → no physical change (sample only)"],
      ["다른 시드 = 다른 무작위 표본 (소자 탭: 시드 간 4–18 mV)", "New seed = new random sample (device tab: 4–18 mV apart)"],
    ],
    { basis: "device-tab seed block (ref & photo)" },
  ),
  g(
    "c_noise",
    [
      "회로 과도해석에서 정공이 무작위로 생기는 캐리어 잡음(알갱이 잡음)을 켭니다. 켜면 스위칭 시점이 매번 달라집니다.",
      "Turns on carrier noise — the random, grain-like generation of holes — in the circuit transient. When on, the switching time differs every run.",
    ],
    [
      ["V_LU 평균 ↓ (소자 탭 기준 보정: 약 59 mV)", "V_LU mean ↓ (device tab, reference: ≈ 59 mV)"],
      ["V_LD 평균 ↑ (소자 탭 기준 보정: 약 100 mV)", "V_LD mean ↑ (device tab, reference: ≈ 100 mV)"],
      ["펄스마다 달라지는 무작위 발화(p-bit)의 주원인", "The main source of pulse-to-pulse random firing (p-bit)"],
    ],
    {
      caveat: [
        "회로에서는 상향·하향 모두에 작용합니다. 수치는 느린 스윕(0.4 V/s) 기준이며, 빠른 펄스에서는 차이가 작아집니다.",
        "In the circuit it acts on both switching directions. The numbers are for a slow sweep (0.4 V/s); fast pulses shrink the shift.",
      ],
      basis: "device-tab carrier_noise / ld_carrier_noise (ref)",
    },
  ),
  g(
    "c_ls_mode",
    [
      "회로 실행마다 국소 상태를 어떻게 흔들지 고릅니다: 없음, 실행마다 한 번 뽑기(고정), 시간에 따라 변하기(진화).",
      "Chooses how the local state varies in circuit runs: none, one draw per run (frozen), or drifting in time (evolving).",
    ],
    [
      ["V_LU 평균 → 거의 그대로 · '없음'이면 퍼짐이 크게 ↓", "V_LU mean → almost unchanged · 'none' cuts the spread a lot"],
      ["V_LD 평균 → 거의 그대로 · '없음'이면 퍼짐 ↓", "V_LD mean → almost unchanged · 'none' shrinks the spread"],
      ["소자 탭 기준 보정: '없음'이면 σ(V_LU) 110 → 8.4 mV", "Device tab, reference: 'none' cuts σ(V_LU) 110 → 8.4 mV"],
    ],
    { basis: "device-tab ls_sigma auto_engine_4V (frozen vs none)" },
  ),
  g(
    "c_ls_action",
    [
      "회로 실행에서 국소 상태의 흔들림이 어느 '손잡이'에 걸리는지 고릅니다. 기본은 GIDL이고, 나머지는 가설입니다.",
      "Chooses which 'knob' the local-state jitter acts on in circuit runs. GIDL is the default; the others are hypotheses.",
    ],
    [
      ["V_LU → 중심 fold는 그대로 · 퍼짐의 원인만 바뀜", "V_LU → fold unchanged · only the spread's source changes"],
      ["V_LD → 중심 fold는 그대로", "V_LD → fold unchanged"],
      ["GIDL 작용점: 상태가 1 mV 흔들리면 V_LU가 약 0.8 mV 흔들림", "GIDL: 1 mV of state jitter → ≈ 0.8 mV of V_LU jitter"],
    ],
    {
      caveat: ["GIDL 외 작용점은 보정되지 않은 가설입니다.", "Action points other than GIDL are uncalibrated hypotheses."],
      basis: "dV_LU/dφ_G from dphiG0 (ref ±10 mV)",
    },
  ),
  g(
    "c_ls_sigma",
    [
      "회로 실행마다 드레인 가장자리 상태가 얼마나 크게 흔들리는지입니다. 클수록 스위칭 시점과 전압이 더 넓게 퍼집니다.",
      "How strongly the drain-edge state jitters between circuit runs. The larger it is, the more switching times and voltages scatter.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 · σ 약 2배 ↑ (×2)", "V_LU mean → almost unchanged · σ roughly doubles (×2)"],
      ["V_LD 평균 → 거의 그대로 (GIDL 작용점)", "V_LD mean → almost unchanged (GIDL action point)"],
      ["상태가 더 크게 흔들림 → 발화 문턱이 매번 더 달라짐", "More state jitter → the firing threshold varies more"],
    ],
    { basis: "device-tab ls_sigma block (ref ×2, 0 → 6 V uncensored)" },
  ),
  g(
    "c_ls_tau",
    [
      "회로 실행에서 국소 상태가 얼마나 오래 기억되는지(상관 시간)입니다. 길수록 이웃한 펄스의 반응이 비슷해집니다.",
      "How long the local state is remembered in circuit runs (correlation time). The longer it is, the more alike neighboring pulses respond.",
    ],
    [
      ["V_LU 평균 → 거의 그대로 (상관만 바뀜)", "V_LU mean → almost unchanged (only correlation changes)"],
      ["V_LD → 영향 없음 (GIDL 작용점)", "V_LD → no effect (GIDL action point)"],
      ["이웃 펄스끼리 상관 ↑ (×10: lag-1 0.67 → 0.92)", "Neighboring pulses more alike (×10: lag-1 0.67 → 0.92)"],
    ],
    { basis: "device-tab ls_tau block (ref ×10)" },
  ),
];

function assemble(entries: ParamGuide[]): Record<string, ParamGuide> {
  const out: Record<string, ParamGuide> = {};
  for (const e of entries) {
    if (out[e.key]) throw new Error(`duplicate parameter guide key: ${e.key}`);
    out[e.key] = e;
  }
  return out;
}

/** Guide per parameter key (FieldDef.key in params/schema.ts, the light-block keys, and the sensitivity keys). */
export const PARAM_GUIDE: Record<string, ParamGuide> = assemble(ENTRIES);
