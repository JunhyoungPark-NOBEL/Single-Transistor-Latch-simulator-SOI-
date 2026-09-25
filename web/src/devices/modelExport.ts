// Commercial behavioral exports deliberately preserve the web solver's quasi-static branch interpolation.
// It is not a port of the engine's SRH transport / body-charge transient solver.
import type { BranchesResult, DeviceBlock, DeviceGeometry, SweepBlock, XY } from "../api/types";
import { isReferenceGeometry, resolveGeometry } from "../params/geometry";
import { geometryLine } from "./library";

export type Point = readonly [voltage: number, current: number];
export interface ExportSelection {
  device: DeviceBlock; sweep: SweepBlock; name?: string;
  /** Embed the calibration descriptors and the engine vector as comments (off by default: the file is shared). */
  includeCalibration?: boolean;
}
export type ExportFormat = "ltspice" | "verilog-a" | "sentaurus";
export interface VerilogAExport {
  filename: string;
  source: string;
  readme: string;
  modelName: string;
  maxVoltage: number;
}
export interface LtspiceExport {
  filename: string;
  circuit: string;
  readme: string;
  subcircuit: string;
  modelName: string;
  maxVoltage: number;
}
export type ExportErrorCode = "invalid" | "range" | "noData";
export class ModelExportError extends Error {
  constructor(public code: ExportErrorCode) { super(code); }
}

const MIN_DV = 1e-9;
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const num = (v: number) => v.toExponential(16);
const asciiName = (s: string) => `STL_${s.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48) || "device"}`;

/** Same first-occurrence monotonic selection used by deterministic._monotone. */
function points(curve: XY): Point[] {
  if (curve.vd.length !== curve.id.length) throw new ModelExportError("invalid");
  const out: Point[] = [];
  for (let i = 0; i < curve.vd.length; i++) {
    const v = curve.vd[i], current = curve.id[i];
    if (!finite(v) || !finite(current) || v < 0 || current < 0) throw new ModelExportError("noData");
    // Points closer than 1 nV to the previous one are numerical noise of the u-parameterised locus
    // (the reference HRS has ~90 below 1 nV, down to 1e-58 V); they carry < 1e-25 A and break table parsers.
    if (!out.length || v > out[out.length - 1][0] + MIN_DV) out.push([v, current]);
  }
  if (out.length < 2) throw new ModelExportError("noData");
  return out;
}

/** Current law used by the generated exp(table()) expression, with exact equilibrium at VDS = 0. */
export function interpolatedCurrent(part: readonly Point[], voltage: number): number {
  if (voltage <= 0) return 0;
  let i = 0;
  while (i < part.length - 2 && part[i + 1][0] < voltage) i++;
  const a = part[i], b = part[i + 1];
  const f = Math.max(0, Math.min(1, (voltage - a[0]) / (b[0] - a[0])));
  return Math.exp((1 - f) * Math.log(Math.max(a[1], 1e-300)) + f * Math.log(Math.max(b[1], 1e-300)));
}

function table(name: string, part: readonly Point[]): string {
  // One pair per continuation line: no parser-dependent long physical lines.
  const rows = part.map(([v, current]) => `+ ${num(v)}, ${num(Math.log(Math.max(current, 1e-300)))}`);
  return `.func ${name}(v) {if(v<=0,0,exp(table(v,\n${rows.join(",\n")}\n+ )))}`;
}

function jsonComments(value: unknown): string {
  return JSON.stringify(value, null, 2).split("\n").map((line) => `* ${line}`).join("\n");
}

/** Only call with a live engine result for this exact selection; never with UI demo/snapshot data. */
function prepareModelExport(selection: ExportSelection, result: BranchesResult) {
  const { device, sweep } = selection;
  if (!finite(sweep.vd_max_V) || sweep.vd_max_V <= 0 || !finite(sweep.rate_V_per_s) || sweep.rate_V_per_s <= 0 ||
      !finite(sweep.dv_V) || sweep.dv_V <= 0 || !finite(device.vg) || !finite(result.iph_A) ||
      !Array.isArray(result.p) || ![26, 32, 33].includes(result.p.length) || !result.p.every(finite) ||
      Math.abs(result.p[11] - device.vg) > 1e-10) throw new ModelExportError("invalid");

  const backGate = device.vbg ?? 0;
  const computedBackGate = result.vbg ?? 0;
  const packedBackGate = result.p.length === 33 ? result.p[32] : 0;
  if (!finite(backGate) || !finite(computedBackGate) || Math.abs(computedBackGate - backGate) > 1e-10 ||
      Math.abs(packedBackGate - backGate) > 1e-10) throw new ModelExportError("invalid");

  const geometry = resolveGeometry(device.geometry);
  // Older results only represent the calibrated reference. Never label those tables as a resized device.
  if (!result.geometry) {
    if (!isReferenceGeometry(geometry)) throw new ModelExportError("invalid");
  } else if ((Object.keys(geometry) as (keyof DeviceGeometry)[]).some((k) =>
    !finite(result.geometry![k]) || Math.abs(result.geometry![k] / geometry[k] - 1) > 1e-12)) {
    throw new ModelExportError("invalid");
  }

  const { V_LU: lu, V_LD: ld, I_LU: ilu, I_LD: ild } = result.folds;
  let hrs: Point[], lrs: Point[] = [];
  if (result.latch) {
    if (![lu, ld, ilu, ild].every(finite) || lu! <= ld! || ld! <= 0 || ilu! <= 0 || ild! <= 0) throw new ModelExportError("invalid");
    hrs = points(result.HRS);
    lrs = points(result.LRS);
    // Close both stable segments with the same refined fold values as double_sweep().
    if (lu! > hrs[hrs.length - 1][0] + MIN_DV) hrs.push([lu!, ilu!]);
    else if (lu! > hrs[hrs.length - 1][0]) hrs[hrs.length - 1] = [lu!, ilu!];
    if (ld! < lrs[0][0] - MIN_DV) lrs.unshift([ld!, ild!]);
    else if (ld! < lrs[0][0]) lrs[0] = [ld!, ild!];
    if (hrs[0][0] !== 0 || hrs[hrs.length - 1][0] < lu! || lrs[0][0] > ld! ||
        lrs[lrs.length - 1][0] < sweep.vd_max_V || sweep.vd_max_V <= lu!) throw new ModelExportError("range");
  } else {
    // The server's no-latch sweep already excludes any untraceable locus after its first gap.
    // Reject missing cells instead of silently bridging/extrapolating them.
    hrs = points(result.double_sweep.up);
    if (hrs[0][0] !== 0 || hrs[hrs.length - 1][0] < sweep.vd_max_V) throw new ModelExportError("range");
  }

  return { hrs, lrs, lu, ld };
}

/** Only call with a live engine result for this exact selection; never with UI demo/snapshot data. */
export function buildLtspiceExport(selection: ExportSelection, result: BranchesResult): LtspiceExport {
  const { device, sweep } = selection;
  const { hrs, lrs, lu, ld } = prepareModelExport(selection, result);
  const modelName = asciiName(selection.name ?? "device");
  const maxVoltage = sweep.vd_max_V;
  const tPeak = maxVoltage / sweep.rate_V_per_s;
  const tStop = 2 * tPeak;
  const maxStep = Math.min(sweep.dv_V / sweep.rate_V_per_s, tPeak / 2000);
  const fixed = `VGS=${num(result.p[11])} V, VBG=${num(device.vbg ?? 0)} V, IPH=${num(result.iph_A)} A`;
  const metadata = {
    format: "stl-ltspice-quasistatic-v1",
    device_name: selection.name ?? "STL device",
    approximation: "Deterministic fixed-bias stable-branch lookup; instantaneous ideal hysteresis; no body-charge dynamics or noise",
    terminal_order: ["D", "S"],
    valid_VDS_V: [0, maxVoltage],
    fixed_VGS_V: result.p[11],
    fixed_VBG_V: device.vbg ?? 0,
    fixed_IPH_A: result.iph_A,
    fixed_geometry: resolveGeometry(device.geometry),
    geometry_model: result.geometry_model ?? null,
    calibration_included: !!selection.includeCalibration,
    ...(selection.includeCalibration ? { effective_engine_p: result.p, submitted_device: device } : {}),
    sweep,
    folds: result.folds,
    engine_grid: result.grid ?? device.numerics.grid,
    engine_warnings: result.warnings,
  };
  const lines = [
    `* ${modelName}: deterministic quasi-static ID-VD behavioral model`,
    "* Pins: D S. Gate/light/local-state means are fixed at export.",
    `* ${fixed}`,
    `* Fixed geometry: ${geometryLine(resolveGeometry(device.geometry))}`,
    `* Supported V(D,S): 0 to ${num(maxVoltage)} V. No extrapolation is validated.`,
    "* Table endpoints clamp outside their range; VDS<=0 returns zero.",
    "* No physical body-charge transient, noise, temperature or gate sweep model.",
    "* Switching is ideal/instantaneous; do not infer device switching time or oscillator frequency.",
    `.subckt ${modelName} D S`,
    table("Ihrs", hrs),
  ];
  if (result.latch) {
    lines.push(
      table("Ilrs", lrs),
      "* Internal Schmitt state is electrically isolated from the drain.",
      "Vlogic logic S 1",
      "Smemory logic state D S latch_memory OFF",
      "Rmemory state S 1e6",
      `.model latch_memory SW(Ron=1 Roff=1e12 Vt=${num((lu! + ld!) / 2)} Vh=${num((lu! - ld!) / 2)})`,
      "Bdrain D S I={if(V(state,S)>0.5,Ilrs(V(D,S)),Ihrs(V(D,S)))}",
    );
  } else lines.push("Bdrain D S I={Ihrs(V(D,S))}");
  lines.push(`.ends ${modelName}`, "");
  const subcircuit = [...lines,
    "* Export metadata:",
    jsonComments(metadata), ""].join("\n");
  const circuit = [
    `${modelName} - quasi-static triangular ID-VD example`,
    "* Open this .cir in LTspice, Run, then plot -I(Vdrive) versus V(d).",
    "* The .subckt block below can be copied into a separate .lib file.",
    "* Read the scope comments before reuse. No physical transient prediction.",
    `Vdrive d 0 PWL(0 0 ${num(tPeak)} ${num(maxVoltage)} ${num(tStop)} 0)`,
    `Xdevice d 0 ${modelName}`,
    `.tran 0 ${num(tStop)} 0 ${num(maxStep)}`,
    ".options plotwinsize=0 reltol=1e-4 abstol=1e-15",
    ".save V(d) I(Vdrive)",
    "",
    subcircuit,
    ".end",
    "",
  ].join("\n");
  const readme = `# ${modelName}: LTspice export\n\n` +
    `## 한국어\n\n` +
    `현재 보정의 **결정론적 준정적 ID–VD 동작 모델**입니다. 고정 조건: ${fixed}. 단자는 **D, S** 순서이며, 게이트 전압은 저장된 상수입니다.\n\n` +
    `형상: ${geometryLine(resolveGeometry(device.geometry))}. 이 형상에서 계산한 ID–VD를 저장합니다. 형상을 바꾸면 웹에서 다시 계산하여 내보내세요.\n\n` +
    `1. 내려받은 \`${modelName}.cir\` 파일을 LTspice에서 열고 Run을 실행합니다.\n` +
    `2. \`-I(Vdrive)\`를 표시하고 가로축을 \`V(d)\`로 바꾸면 상승·하강 ID–VD를 볼 수 있습니다.\n` +
    `3. 다른 회로에 넣을 때 \`.subckt\`부터 \`.ends\`까지를 \`${modelName}.lib\`로 저장합니다. 회로에는 \`.include ${modelName}.lib\`와 \`X1 drain source ${modelName}\`를 추가합니다.\n\n` +
    `전압 범위는 **0–${maxVoltage} V**입니다. 범위 밖 테이블은 끝값으로 고정되며 역방향 전압에서는 0 A입니다. 이 동작은 외삽에 대한 물리 모델이 아닙니다.\n\n` +
    `안정 branch의 전류는 웹 시뮬레이터와 같은 log(I) 선형 보간입니다. ${result.latch ? `이상적 히스테리시스 스위치가 VLU=${lu} V와 VLD=${ld} V에서 전환합니다.` : "이 조건에서 유효한 두 fold가 없어 단일 branch를 내보냈습니다."} 초기 상태는 HRS이며 먼저 0 V에서 시작합니다.\n\n` +
    `**지원 범위:** 고정 게이트·광전류·국소 상태 평균에서 결정론적 ID–VD와 정적 히스테리시스. 캐리어 잡음, 확률적 래칭, 바디 전하의 시간 적분, 물리적 스위칭 시간, 발진 주파수, 온도·형상·게이트 스윕은 포함하지 않습니다. 상승 속도는 예제 시간축만 정합니다. 빠른 펄스나 발진 예측에는 웹의 물리 회로 엔진을 사용합니다. LTspice 바이너리 실행 검증은 아직 수행하지 않았습니다.\n\n` +
    `**공유 전 확인:** 표에는 이 소자의 보정된 ID–VD 곡선이 그대로 담깁니다${selection.includeCalibration ? ". 보정값과 엔진 파라미터 벡터도 JSON 주석으로 들어 있습니다" : ""}. 공유 범위를 확인해 주세요.\n\n` +
    `## English\n\n` +
    `This is a **deterministic quasi-static ID–VD behavioral export**, not the full transport/body-charge solver. Fixed bias: ${fixed}. Pin order: **D S**. The gate has no external pin because its voltage is held constant.\n\n` +
    `Open \`${modelName}.cir\` in LTspice and Run. Plot \`-I(Vdrive)\` versus \`V(d)\`. For reuse, copy the complete \`.subckt … .ends\` block into \`${modelName}.lib\`, add \`.include ${modelName}.lib\`, then instantiate \`X1 drain source ${modelName}\`.\n\n` +
    `Valid drain-source range: **0–${maxVoltage} V**. Stable currents use the web solver's linear interpolation of log(I); the refined folds set a native positive-hysteresis switch. Start at 0 V in HRS. Table endpoints clamp out of range, and VDS ≤ 0 returns zero; neither behavior is a validated extrapolation.\n\n` +
    `Fixed geometry: ${geometryLine(resolveGeometry(device.geometry))}. The tables are computed at this geometry. The exported model has no tunable geometry parameters; regenerate it after changing any dimension or doping.\n\n` +
    `No carrier/local-state noise, body-charge ODE, physical switching delay, oscillator-frequency prediction, temperature scaling or variable gate bias is included. Transient time in the example only traverses the static curve; switching is instantaneous. Regenerate the export after changing calibration or fixed bias. The actual LTspice executable has not been run for validation.\n\n` +
    `Bias, geometry, folds and server warnings are embedded as JSON comments in the .cir${selection.includeCalibration ? ", together with the calibration descriptors and the effective engine parameter vector" : ""}. The tables themselves reproduce the calibrated ID–VD curve: share the file only where that is acceptable.\n\n` +
    `Native switch syntax and Vt/Vh conventions: https://www.analog.com/en/resources/analog-dialogue/articles/how-to-add-a-voltage-controlled-switch.html\n`;
  return { filename: `${modelName}.cir`, circuit, readme, subcircuit, modelName, maxVoltage };
}

/** Explicit functions avoid simulator-specific table file formats or nonstandard lookup helpers. */
function verilogFunction(name: string, part: readonly Point[]): string {
  const logI = (point: Point) => Math.log(Math.max(point[1], 1e-300));
  const lines = [
    `  analog function real ${name};`,
    "    input v;",
    "    real v;",
    "    begin",
    `      if (v <= 0.0) ${name} = 0.0;`,
    `      else if (v <= ${num(part[0][0])}) ${name} = exp(${num(logI(part[0]))});`,
  ];
  for (let i = 1; i < part.length; i++) {
    const a = part[i - 1], b = part[i];
    const slope = (logI(b) - logI(a)) / (b[0] - a[0]);
    lines.push(`      else if (v <= ${num(b[0])})`,
      `        ${name} = exp(${num(logI(a))} + (v - ${num(a[0])}) * ${num(slope)});`);
  }
  lines.push(`      else ${name} = exp(${num(logI(part[part.length - 1]))});`,
    "    end", "  endfunction");
  return lines.join("\n");
}

/** Standard Verilog-A behavioral model; a .va source is not a Sentaurus physical-device deck. */
export function buildVerilogAExport(selection: ExportSelection, result: BranchesResult): VerilogAExport {
  const { hrs, lrs, lu, ld } = prepareModelExport(selection, result);
  const modelName = asciiName(selection.name ?? "device");
  const maxVoltage = selection.sweep.vd_max_V;
  const metadata = {
    format: "stl-verilog-a-quasistatic-v1",
    device_name: selection.name ?? "STL device",
    approximation: "Deterministic fixed-bias stable-branch lookup; instantaneous ideal hysteresis; no body-charge dynamics or noise",
    terminal_order: ["D", "S"],
    valid_VDS_V: [0, maxVoltage],
    fixed_VGS_V: result.p[11],
    fixed_VBG_V: selection.device.vbg ?? 0,
    fixed_IPH_A: result.iph_A,
    fixed_geometry: resolveGeometry(selection.device.geometry),
    geometry_model: result.geometry_model ?? null,
    calibration_included: !!selection.includeCalibration,
    ...(selection.includeCalibration ? { effective_engine_p: result.p, submitted_device: selection.device } : {}),
    sweep: selection.sweep,
    folds: result.folds,
    engine_grid: result.grid ?? selection.device.numerics.grid,
    engine_warnings: result.warnings,
  };
  const lines = [
    `// ${modelName}: deterministic quasi-static ID-VD behavioral model`,
    "// Pins D S. Fixed gate and light; positive drain current flows D -> S.",
    `// VGS = ${num(result.p[11])} V; VBG = ${num(selection.device.vbg ?? 0)} V; IPH = ${num(result.iph_A)} A.`,
    `// Fixed geometry: ${geometryLine(resolveGeometry(selection.device.geometry))}`,
    `// Supported V(D,S): 0 to ${num(maxVoltage)} V.`,
    "// Use transient voltage sweep starting at V(D,S)=0; DC hysteresis is not validated.",
    ...(result.latch ? ["// Needs analog @(cross) event support (e.g. Spectre); OpenVAF-based flows such as ngspice OSDI cannot compile it."] : []),
    "// No physical body-charge transient, noise, temperature or variable gate bias.",
    "// Ideal instantaneous switching: do not infer CSVM frequency, Vtop or Vbottom.",
    "// Table endpoints clamp; VDS <= 0 gives zero. Out-of-range use is not validated.",
    '`include "disciplines.vams"',
    `module ${modelName}(D, S);`,
    "  inout D, S;",
    "  electrical D, S;",
  ];
  if (result.latch) lines.push(`  localparam real V_LU = ${num(lu!)};`, `  localparam real V_LD = ${num(ld!)};`, "  integer lrs_state;");
  lines.push("", verilogFunction("i_hrs", hrs));
  if (result.latch) lines.push("", verilogFunction("i_lrs", lrs));
  lines.push("", "  analog begin");
  if (result.latch) lines.push(
    "    // Initialize HRS in the hysteresis window; above V_LU initialize LRS.",
    "    @(initial_step) lrs_state = (V(D,S) >= V_LU) ? 1 : 0;",
    "    @(cross(V(D,S) - V_LU, +1)) lrs_state = 1;",
    "    @(cross(V(D,S) - V_LD, -1)) lrs_state = 0;",
    "    I(D,S) <+ (lrs_state == 1) ? i_lrs(V(D,S)) : i_hrs(V(D,S));",
  );
  else lines.push("    I(D,S) <+ i_hrs(V(D,S));");
  lines.push("  end", "endmodule", "", "// Export metadata:",
    ...JSON.stringify(metadata, null, 2).split("\n").map((line) => `// ${line}`), "");

  const readme = `# ${modelName}: Verilog-A export\n\n` +
    `## 한국어\n\n` +
    `현재 보정의 **결정론적 준정적 ID–VD 동작 모델**입니다. 단자는 **D, S**이며 고정 조건은 VG=${result.p[11]} V, VBG=${selection.device.vbg ?? 0} V, IPH=${result.iph_A} A입니다. 유효 VDS 범위는 0–${maxVoltage} V입니다.\n\n` +
    `형상: ${geometryLine(resolveGeometry(selection.device.geometry))}. 이 형상에서 계산한 ID–VD를 저장합니다. 형상을 바꾸면 웹에서 다시 계산하여 내보내세요.\n\n` +
    (result.latch ? `**시뮬레이터 요구 사항:** 히스테리시스에 @(cross) 이벤트를 사용하므로 Spectre처럼 아날로그 이벤트를 지원하는 시뮬레이터가 필요합니다. OpenVAF 기반 흐름(ngspice OSDI 등)에서는 컴파일되지 않습니다.\n\n` : "") +
    `Verilog-A를 지원하는 시뮬레이터에 \`${modelName}.va\`를 모델 소스로 등록합니다. \`disciplines.vams\`는 해당 시뮬레이터의 표준 include 경로에서 찾을 수 있어야 합니다. Spectre에서는 \`ahdl_include "${modelName}.va"\`와 \`X1 (d 0) ${modelName}\`로 등록·연결할 수 있습니다. 다른 시뮬레이터는 해당 제품의 Verilog-A 등록 절차를 따릅니다. LTspice는 별도 LTspice 형식으로 내보내세요.\n\n` +
    `0 V에서 시작하는 삼각파 전압원의 **과도해석**으로 상승·하강 ID–VD를 확인합니다. ${result.latch ? `VLU=${lu} V에서 LRS, VLD=${ld} V에서 HRS로 전환합니다. 초기 전압이 VLU 미만이면 HRS, 이상이면 LRS입니다.` : "이 조건은 단일 branch이며 히스테리시스 상태를 만들지 않습니다."} 전류는 안정 branch의 log(I) 선형 보간이며, 표 범위를 넘으면 끝값으로 고정됩니다. 역방향 VDS에서는 0 A입니다.\n\n` +
    `**포함하지 않는 항목:** 바디 전하 시간 적분, 캐리어 잡음, 물리적 스위칭 지연, CSVM의 Vtop·Vbottom·주파수, 온도·형상·게이트 스윕. 시간은 정적 곡선을 따라가는 용도이며 실제 발진을 예측하지 않습니다. DC sweep 히스테리시스와 AC/잡음 해석 용도로 검증하지 않았습니다. 상용 Verilog-A 시뮬레이터 실행 검증은 수행하지 않았습니다.\n\n` +
    `이 .va 파일은 Sentaurus의 물리 소자 TCAD 입력 파일이 아닙니다. Sentaurus 물리 해석에는 별도 구조·격자·도핑 분포·접촉·물리 모델 설정이 필요하며, compact model 연동에는 버전에 맞는 지원 인터페이스와 검증이 필요합니다.\n\n` +
    `**공유 전 확인:** 표에는 이 소자의 보정된 ID–VD 곡선이 그대로 담깁니다${selection.includeCalibration ? ". 보정값과 엔진 파라미터 벡터도 JSON 주석으로 들어 있습니다" : ""}. 공유 범위를 확인해 주세요.\n\n` +
    `## English\n\n` +
    `A standard Verilog-A **deterministic quasi-static behavioral model**, with fixed VG=${result.p[11]} V, VBG=${selection.device.vbg ?? 0} V and IPH=${result.iph_A} A. Pins: **D S**. Valid VDS: 0–${maxVoltage} V. Add the .va source through your simulator's Verilog-A integration and ensure its standard disciplines.vams is on the include path. Spectre example: \`ahdl_include "${modelName}.va"\`, then \`X1 (d 0) ${modelName}\`.\n\n` +
    (result.latch ? `**Simulator requirement:** the hysteresis uses @(cross) events, so the model needs a simulator with analog-event support such as Spectre. It does not compile in OpenVAF-based flows (e.g. ngspice with OSDI).\n\n` : "") +
    `Run a transient triangular voltage sweep from 0 V. Stable branches use linear interpolation of log(I), with ideal threshold events from initial_step/cross. There is no artificial switching delay. Endpoint clamping and zero reverse current are numerical boundaries, not validated extrapolation.\n\n` +
    `Fixed geometry: ${geometryLine(resolveGeometry(selection.device.geometry))}. The tables are computed at this geometry. Regenerate the export after changing any dimension or doping; the .va has no tunable geometry parameters.\n\n` +
    `No body-charge ODE, carrier noise, physical switching delay, CSVM Vtop/Vbottom/frequency, or variable bias/geometry/temperature is modeled. DC-sweep hysteresis and AC/noise operation are not validated. No commercial Verilog-A simulator was executed to validate this source. This is not a Sentaurus physical-device deck, nor a claim of direct .va import into Sentaurus.\n\n` +
    `Bias, geometry, folds and server warnings are embedded as comments in the .va${selection.includeCalibration ? ", together with the calibration descriptors and the effective engine parameter vector" : ""}. The tables themselves reproduce the calibrated ID–VD curve: share the file only where that is acceptable.\n\n` +
    `Language reference: https://www.accellera.org/images/downloads/standards/v-ams/VAMS-LRM-2023.pdf\n`;
  return { filename: `${modelName}.va`, source: lines.join("\n"), readme, modelName, maxVoltage };
}
