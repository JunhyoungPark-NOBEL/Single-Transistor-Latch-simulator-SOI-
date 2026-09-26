// Electrical rule check before submit: ground reference, unconnected pins, floating nodes, DC path to
// ground, voltage-source loops, current sources with nothing in series, element values, names, limits.
// Messages are i18n keys ("schematic.erc.<code>") with variables; `elementIds`/`wireIds` let the UI
// highlight the offending parts.
import { isSupportedTechnology } from "../devices/library";
import type { Connectivity, NetInfo } from "./nets";
import { isCircuitNet, optionalPinConnected, pinId, UnionFind } from "./nets";
import { CIRCUIT_KINDS, isOptionalStlPin, pinsFor, type SchematicDoc, type SElement } from "./model";
import { waveIssues } from "./waves";

export interface ErcItem {
  level: "error" | "warning";
  code: string;
  vars?: Record<string, string | number>;
  elementIds: string[];
  wireIds?: string[];
}

export const LIMITS = { elements: 40, stl: 8, nodes: 30, pwlPoints: 2000 };
// §6.3: element and node names 1–32 characters (the editor is stricter: letters, digits, _)
export const NAME_RE = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
export const LABEL_RE = /^[A-Za-z0-9_+-]{1,32}$/;

const PIN_LABEL: Record<string, string> = { p: "+", n: "−", d: "D", g: "G", s: "S", bg: "BG", b: "B", o: "", i: "IN", q: "OUT" };

/** Nets of an element's pins in pin order (undefined when not found). */
export function elementNets(el: SElement, conn: Connectivity): (NetInfo | undefined)[] {
  const pins = pinsFor(el).map((p) => p.name);
  return pins.map((p) => conn.pinNet.get(pinId(el.id, p)));
}

export function runErc(doc: SchematicDoc, conn: Connectivity): ErcItem[] {
  const out: ErcItem[] = [];
  const els = doc.elements.filter((e) => CIRCUIT_KINDS.includes(e.kind));
  const err = (code: string, elementIds: string[], vars?: ErcItem["vars"], wireIds?: string[]) => out.push({ level: "error", code, elementIds, vars, wireIds });
  const warn = (code: string, elementIds: string[], vars?: ErcItem["vars"], wireIds?: string[]) => out.push({ level: "warning", code, elementIds, vars, wireIds });

  if (!els.length) {
    err("empty", []);
    return out;
  }
  // ---- limits
  if (els.length > LIMITS.elements) err("tooManyElements", [], { n: els.length, max: LIMITS.elements });
  const stls = els.filter((e) => e.kind === "STL");
  if (stls.length > LIMITS.stl) err("tooManyStl", stls.map((e) => e.id), { n: stls.length, max: LIMITS.stl });
  const circuitNets = conn.nets.filter((n) => isCircuitNet(n, conn));
  const nodeCount = circuitNets.filter((n) => !n.ground).length;
  if (nodeCount > LIMITS.nodes) err("tooManyNodes", [], { n: nodeCount, max: LIMITS.nodes });

  // ---- names
  const seen = new Map<string, SElement>();
  for (const e of els) {
    if (!NAME_RE.test(e.name)) err("badName", [e.id], { name: e.name });
    const k = e.name.toUpperCase();
    const prev = seen.get(k);
    if (prev) err("dupName", [prev.id, e.id], { name: e.name });
    else seen.set(k, e);
  }
  for (const l of doc.elements.filter((e) => e.kind === "LABEL")) {
    const s = (l.label ?? "").trim();
    if (!s) err("labelEmpty", [l.id]);
    else if (!LABEL_RE.test(s)) err("labelBad", [l.id], { name: s });
  }
  for (const n of conn.nets) if (n.labels.length > 1) warn("twoLabels", n.allPins.filter((p) => p.el.kind === "LABEL").map((p) => p.el.id), { a: n.labels[0], b: n.labels[1] });

  // ---- ground
  const gnd = conn.nets.find((n) => n.ground);
  if (!gnd || !gnd.pins.length) err("noGround", doc.elements.filter((e) => e.kind === "GND").map((e) => e.id));

  // ---- values and waveforms
  for (const e of els) {
    if (e.kind === "R" && !(typeof e.value === "number" && Number.isFinite(e.value) && e.value > 0)) err("badR", [e.id], { name: e.name });
    if (e.kind === "C") {
      if (!(typeof e.value === "number" && Number.isFinite(e.value) && e.value >= 0)) err("badC", [e.id], { name: e.name });
      else if (e.value === 0) warn("zeroC", [e.id], { name: e.name });
    }
    if ((e.kind === "V" || e.kind === "I") && !e.wave) err("noWave", [e.id], { name: e.name });
    for (const w of [e.wave, e.light].filter(Boolean)) {
      for (const iss of waveIssues(w!)) (iss.level === "error" ? err : warn)(`wave.${iss.key}`, [e.id], { name: e.name, ...(iss.vars ?? {}) });
    }
    if (e.kind === "STL" && !e.stl?.device) err("noDevice", [e.id], { name: e.name });
    if (e.kind === "STL" && e.stl && !isSupportedTechnology(e.stl.technology ?? "FDSOI")) err("unsupportedTechnology", [e.id], { name: e.name, tech: e.stl.technology ?? "" });
    if (e.kind === "CMP") {
      const c = e.cmp;
      if (!c || ![c.v_ref, c.v_high, c.v_low, c.hysteresis].every((x) => Number.isFinite(x)) || c.v_high === c.v_low || c.hysteresis < 0) err("cmpBad", [e.id], { name: e.name });
    }
  }

  const nets = new Map(els.map((e) => [e.id, elementNets(e, conn)] as const));

  // ---- shorted parts
  for (const e of els) {
    if (["STL", "MOS", "BJT"].includes(e.kind)) continue;
    const [a, b] = nets.get(e.id)!;
    if (e.kind === "CMP") {
      if (a && b && a === b) warn("cmpFeedback", [e.id], { name: e.name, node: a.name });
      continue;
    }
    if (a && b && a === b) (e.kind === "V" ? err : warn)(e.kind === "V" ? "vShort" : "shorted", [e.id], { name: e.name, node: a.name });
  }

  // ---- voltage-source loops (V sources and comparator outputs, which are voltage sources to ground)
  const vuf = new UnionFind();
  const cmpOut = new Map<string, SElement>();
  for (const e of els.filter((x) => x.kind === "CMP")) {
    const q = nets.get(e.id)![1];
    if (!q || !gnd) continue;
    if (q.id === gnd.id) err("cmpDriven", [e.id], { name: e.name, other: "GND" });
    else if (cmpOut.has(q.id)) err("cmpDriven", [e.id, cmpOut.get(q.id)!.id], { name: e.name, other: cmpOut.get(q.id)!.name });
    else {
      cmpOut.set(q.id, e);
      vuf.union(q.id, gnd.id);
    }
  }
  for (const e of els.filter((x) => x.kind === "V")) {
    const [a, b] = nets.get(e.id)!;
    if (!a || !b || a === b) continue;
    if (!vuf.union(a.id, b.id)) {
      const c = cmpOut.get(a.id) ?? cmpOut.get(b.id);
      if (c) err("cmpDriven", [c.id, e.id], { name: c.name, other: e.name });
      else err("vLoop", [e.id], { name: e.name });
    }
  }

  // ---- DC path to ground (R, V and the STL drain–source path conduct; C, I and the gate do not)
  const reach = new Set<string>();
  if (gnd) {
    const adj = new Map<string, Set<string>>();
    const link = (a?: NetInfo, b?: NetInfo) => {
      if (!a || !b) return;
      if (!adj.has(a.id)) adj.set(a.id, new Set());
      if (!adj.has(b.id)) adj.set(b.id, new Set());
      adj.get(a.id)!.add(b.id);
      adj.get(b.id)!.add(a.id);
    };
    for (const e of els) {
      const ns = nets.get(e.id)!;
      if (e.kind === "R" || e.kind === "V" || e.kind === "D") link(ns[0], ns[1]);
      if (e.kind === "BJT") { link(ns[0], ns[1]); link(ns[1], ns[2]); }
      if (e.kind === "STL" || e.kind === "MOS") link(ns[0], ns[2]);
      // The physical body has intrinsic recombination / junction loss to source. External C is
      // therefore valid here even without an added resistor; insulated gates still need a driver.
      if (e.kind === "STL" && optionalPinConnected(e.id, "b", conn)) link(ns[4], ns[2]);
      if (e.kind === "CMP") link(ns[1], gnd); // the output is a voltage source to ground; the input is ideal
    }
    reach.add(gnd.id);
    const stack = [gnd.id];
    while (stack.length) {
      const cur = stack.pop()!;
      for (const nb of adj.get(cur) ?? []) if (!reach.has(nb)) {
        reach.add(nb);
        stack.push(nb);
      }
    }
    for (const n of circuitNets) {
      if (reach.has(n.id) || n.pins.length < 2) continue;
      const kinds = new Set(n.pins.map((p) => (p.el.kind === "STL" ? `STL.${p.pin}` : p.el.kind)));
      const ids = [...new Set(n.pins.map((p) => p.el.id))];
      const names = [...new Set(n.pins.map((p) => p.el.name))].join(", ");
      if ([...kinds].every((k) => k === "I")) err("iSeries", ids, { node: n.name, names }, n.wires.map((w) => w.id));
      else if ([...kinds].every((k) => (k === "STL.g" || k === "STL.bg"))) err("gateOnly", ids, { node: n.name, names }, n.wires.map((w) => w.id));
      else err("noDcPath", ids, { node: n.name, names }, n.wires.map((w) => w.id));
    }
  }

  // ---- unconnected pins / single-terminal nodes: an error when that node has no DC path to ground (the
  //      matrix would be singular), otherwise a warning (a dangling end, like the server's ERC)
  const unconnectedIds = new Set<string>();
  for (const p of conn.unconnected) {
    if (p.el.kind === "GND") continue;
    if (p.el.kind === "LABEL") {
      warn("labelAlone", [p.el.id], { name: p.el.label ?? "" });
      continue;
    }
    if (isOptionalStlPin(p.el, p.pin) && !optionalPinConnected(p.el.id, p.pin, conn)) continue;
    unconnectedIds.add(pinId(p.el.id, p.pin));
    const n = conn.pinNet.get(pinId(p.el.id, p.pin));
    (n && reach.has(n.id) ? warn : err)("unconnected", [p.el.id], { name: p.el.name, pin: PIN_LABEL[p.pin] ?? p.pin });
  }
  for (const n of circuitNets) {
    if (n.ground || n.pins.length !== 1) continue;
    const p = n.pins[0];
    if (p.el.kind === "CMP" && p.pin === "q") continue; // an unloaded comparator output is fine (probe it)
    if (unconnectedIds.has(pinId(p.el.id, p.pin))) continue; // already reported
    (reach.has(n.id) ? warn : err)("floating", [p.el.id], { node: n.name, name: p.el.name, pin: PIN_LABEL[p.pin] ?? p.pin }, n.wires.map((w) => w.id));
  }

  // ---- settings
  const tr = doc.tran;
  if (!(tr.t_stop_s > 0)) err("tStop", []);
  if (tr.t_start_save_s < 0 || tr.t_start_save_s >= tr.t_stop_s) err("tStart", []);
  if (tr.dt_max_s != null && !(tr.dt_max_s > 0)) err("dtMax", []);
  if (tr.dt_min_s != null && tr.dt_max_s != null && tr.dt_min_s > tr.dt_max_s) err("dtOrder", []);
  if (!(tr.reltol > 0 && tr.reltol <= 0.1)) err("reltol", []);
  return out;
}

export const hasErrors = (items: ErcItem[]) => items.some((i) => i.level === "error");
