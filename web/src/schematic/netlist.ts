// Schematic → §6 request (netlist + .tran + stochastic + detect + probes) and a read-only SPICE-like
// text view of the same netlist (for transparency). Pure functions (unit-tested).
import type { CustomCircuitRequest, CustomElement, CustomTran } from "../api/circuitCustom";
import type { LocalStateBlock, Mode } from "../api/types";
import { clone } from "../utils/object";
import { elementNets } from "./erc";
import { CIRCUIT_KINDS, DEFAULT_CMP, type SchematicDoc, type SElement, type TranSettings } from "./model";
import type { Connectivity } from "./nets";
import { toSpice } from "./si";
import { waveSpice } from "./waves";

export const autoDtMax = (tStop: number) => tStop / 2000;
export const autoDtMin = (tStop: number) => Math.max(1e-15, 1e-13 * tStop);

export function effectiveTran(tr: TranSettings): CustomTran {
  return {
    t_stop_s: tr.t_stop_s,
    t_start_save_s: tr.t_start_save_s,
    dt_max_s: tr.dt_max_s ?? autoDtMax(tr.t_stop_s),
    dt_min_s: tr.dt_min_s ?? autoDtMin(tr.t_stop_s),
    method: tr.method,
    reltol: tr.reltol,
  };
}

const NONE_LOCAL: LocalStateBlock = { mode: "none", action: "gidl", sigma: 0, tau_s: 5, sigma_E_V: 0, tau_E_s: 1.62, acquisition_trend: false };

/** Local state sent in `stochastic.local_state`: the override, or the first STL's library setting. */
export function globalLocalState(doc: SchematicDoc): LocalStateBlock {
  if (doc.stoch.local_source === "override") return { ...clone(doc.stoch.local_state), acquisition_trend: false };
  const first = doc.elements.find((e) => e.kind === "STL" && e.stl?.local_state);
  return { ...clone(first?.stl?.local_state ?? NONE_LOCAL), acquisition_trend: false };
}

const nodeName = (n: { name: string } | undefined) => n?.name ?? "?";

export function toCustomElement(e: SElement, conn: Connectivity): CustomElement | null {
  const ns = elementNets(e, conn).map(nodeName);
  switch (e.kind) {
    case "R":
    case "C":
      return { type: e.kind, name: e.name, nodes: [ns[0], ns[1]], value: e.value ?? 0 };
    case "V":
    case "I":
      return { type: e.kind, name: e.name, nodes: [ns[0], ns[1]], wave: clone(e.wave ?? { kind: "dc", value: 0 }) };
    case "STL": {
      if (!e.stl) return null;
      const out: CustomElement = { type: "STL", name: e.name, nodes: { d: ns[0], g: ns[1], s: ns[2] }, device: clone(e.stl.device), light_pA: e.light ? clone(e.light) : null };
      if (e.stl.local_state) out.local_state = { ...clone(e.stl.local_state), acquisition_trend: false };
      return out;
    }
    case "CMP": {
      const c = e.cmp ?? DEFAULT_CMP;
      return { type: "CMP", name: e.name, nodes: { in: ns[0], out: ns[1] }, v_ref: c.v_ref, v_high: c.v_high, v_low: c.v_low, hysteresis: c.hysteresis };
    }
    default:
      return null;
  }
}

export function buildRequest(doc: SchematicDoc, conn: Connectivity, mode: Mode, probes: string[] | null): CustomCircuitRequest {
  const elements = doc.elements.filter((e) => CIRCUIT_KINDS.includes(e.kind)).map((e) => toCustomElement(e, conn)).filter((x): x is CustomElement => !!x);
  const req: CustomCircuitRequest = {
    bench: "custom",
    mode,
    netlist: { elements },
    tran: effectiveTran(doc.tran),
    detect: { i_threshold_A: doc.detect.i_threshold_A, hysteresis: doc.detect.hysteresis },
    probes: null,
  };
  if (!doc.save_all && probes?.length) {
    // the server rejects unknown probe keys (§6.3): keep only keys valid for this netlist
    const ok = validProbeKeys(elements);
    const keep = probes.filter((k) => ok.has(k));
    req.probes = keep.length ? keep : null;
  }
  if (mode === "stochastic")
    req.stochastic = {
      seed: Math.round(doc.stoch.seed),
      n_runs: Math.round(doc.stoch.n_runs),
      carrier_noise: doc.stoch.carrier_noise,
      ld_carrier_noise: doc.stoch.ld_carrier_noise,
      local_state: globalLocalState(doc),
      local_state_override: doc.stoch.local_source === "override",
    };
  return req;
}

/** Every probe key the netlist can answer: V(node), I(element), I(X.d|g|s), X.u/.r/.q_b. */
export function validProbeKeys(elements: CustomElement[]): Set<string> {
  const keys = new Set<string>(["V(0)"]);
  for (const e of elements) {
    const nodes = e.type === "STL" || e.type === "CMP" ? Object.values(e.nodes) : e.nodes;
    for (const n of nodes) keys.add(`V(${n})`);
    if (e.type === "STL") for (const k of [`I(${e.name}.d)`, `I(${e.name}.g)`, `I(${e.name}.s)`, `${e.name}.u`, `${e.name}.r`, `${e.name}.q_b`]) keys.add(k);
    else if (e.type === "CMP") for (const k of [`I(${e.name})`, `${e.name}.bit`]) keys.add(k);
    else keys.add(`I(${e.name})`);
  }
  return keys;
}

/** Node names used by the circuit (ground first). */
export function nodeList(conn: Connectivity): string[] {
  const names = conn.nets.filter((n) => n.pins.length > 0).map((n) => n.name);
  return [...new Set(names)].sort((a, b) => (a === "0" ? -1 : b === "0" ? 1 : a.localeCompare(b, "en", { numeric: true })));
}

const fmtV = (v: number) => toSpice(v, 5);

/** SPICE-like listing (read-only view). */
export function netlistText(doc: SchematicDoc, conn: Connectivity, mode: Mode): string {
  const lines: string[] = [`* ${doc.name || "untitled"} — STL circuit (${mode})`];
  const els = doc.elements.filter((e) => CIRCUIT_KINDS.includes(e.kind));
  const order: Record<string, number> = { V: 0, I: 1, R: 2, C: 3, STL: 4, CMP: 5 };
  for (const e of [...els].sort((a, b) => order[a.kind] - order[b.kind] || a.name.localeCompare(b.name, "en", { numeric: true }))) {
    const ns = elementNets(e, conn).map(nodeName);
    switch (e.kind) {
      case "R":
      case "C":
        lines.push(`${e.name} ${ns[0]} ${ns[1]} ${fmtV(e.value ?? 0)}`);
        break;
      case "V":
      case "I":
        lines.push(`${e.name} ${ns[0]} ${ns[1]} ${e.wave ? waveSpice(e.wave) : "DC 0"}`);
        break;
      case "STL": {
        const d = e.stl?.device;
        const parts = [`${e.name} ${ns[0]} ${ns[1]} ${ns[2]} STL`, `dev="${e.stl?.name ?? "?"}"`];
        if (d) parts.push(`preset=${d.preset}`);
        if (e.light) parts.push(`light_pA=${waveSpice(e.light)}`);
        else if (d) {
          const iph = d.light.mode === "power" ? d.light.power_mW * d.light.responsivity_pA_per_mW : d.light.iph_pA;
          if (iph) parts.push(`iph=${fmtV(iph)}p`);
        }
        if (e.stl?.local_state && e.stl.local_state.mode !== "none") parts.push(`local=${e.stl.local_state.mode}`);
        lines.push(parts.join(" "));
        break;
      }
      case "CMP": {
        const c = e.cmp ?? DEFAULT_CMP;
        lines.push(`${e.name} ${ns[0]} 0 ${ns[1]} CMP vref=${fmtV(c.v_ref)} vhigh=${fmtV(c.v_high)} vlow=${fmtV(c.v_low)}${c.hysteresis ? ` hyst=${fmtV(c.hysteresis)}` : ""}`);
        break;
      }
    }
  }
  const tr = effectiveTran(doc.tran);
  lines.push(`.tran 0 ${fmtV(tr.t_stop_s)} ${fmtV(tr.t_start_save_s)} ${fmtV(tr.dt_max_s)}`);
  lines.push(`.options method=${tr.method} reltol=${fmtV(tr.reltol)} dtmin=${fmtV(tr.dt_min_s)}`);
  lines.push(`.detect ith=${fmtV(doc.detect.i_threshold_A)} hysteresis=${doc.detect.hysteresis}`);
  if (mode === "stochastic") {
    const ls = globalLocalState(doc);
    lines.push(
      `.stochastic runs=${doc.stoch.n_runs} seed=${doc.stoch.seed} carrier_noise=${doc.stoch.carrier_noise ? "on" : "off"} ld_noise=${doc.stoch.ld_carrier_noise ? "on" : "off"} local=${ls.mode}${ls.mode !== "none" ? `(${ls.action}, σ=${fmtV(ls.sigma)})` : ""}`,
    );
  }
  lines.push(doc.save_all ? ".save all" : ".save <probed traces>");
  lines.push(".end");
  return lines.join("\n");
}
