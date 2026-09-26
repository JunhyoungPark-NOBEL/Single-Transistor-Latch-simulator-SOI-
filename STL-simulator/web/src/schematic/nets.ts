// Connectivity: nets from union-find over pins, wire end points (incl. T-junctions on a wire's interior)
// and net labels (same text = same net; "0"/"gnd" = ground). Also junction dots and unconnected pins.
import { isOptionalStlPin, onWireInterior, pinPositions, ptKey, type PinPos, type Pt, type SchematicDoc, type Wire } from "./model";

export class UnionFind {
  private parent = new Map<string, string>();
  find(a: string): string {
    let p = this.parent.get(a);
    if (p === undefined) {
      this.parent.set(a, a);
      return a;
    }
    let root = a;
    while (p !== root) {
      root = p;
      p = this.parent.get(root) ?? root;
    }
    // path compression
    let cur = a;
    while (cur !== root) {
      const next = this.parent.get(cur) ?? root;
      this.parent.set(cur, root);
      cur = next;
    }
    return root;
  }
  union(a: string, b: string): boolean {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    // deterministic root: lexicographically smaller key wins
    if (ra < rb) this.parent.set(rb, ra);
    else this.parent.set(ra, rb);
    return true;
  }
}

export const GROUND_NAMES = new Set(["0", "gnd", "GND", "Gnd"]);
export const isGroundName = (s: string) => GROUND_NAMES.has(s.trim());

export interface NetInfo {
  /** Root key (stable within one extraction). */
  id: string;
  /** Node name used in the netlist: "0" for ground, the label text, or N001, N002 … */
  name: string;
  ground: boolean;
  labels: string[];
  /** Pins of circuit elements (R, C, V, I, STL) on this net. */
  pins: PinPos[];
  /** All symbol pins incl. GND/LABEL. */
  allPins: PinPos[];
  wires: Wire[];
  /** Representative point for on-canvas annotations. */
  anchor: Pt;
}

export interface Connectivity {
  nets: NetInfo[];
  byName: Map<string, NetInfo>;
  /** `${elementId}:${pin}` → net */
  pinNet: Map<string, NetInfo>;
  wireNet: Map<string, NetInfo>;
  junctions: Pt[];
  /** Pins with nothing else attached at their location. */
  unconnected: PinPos[];
}

/** Optional terminals with no wire, label or other pin use their intrinsic device defaults. */
export function optionalPinConnected(elId: string, pin: string, conn: Connectivity): boolean {
  const net = conn.pinNet.get(pinId(elId, pin));
  return !!net && (net.wires.length > 0 || net.allPins.some((p) => p.el.id !== elId || p.pin !== pin));
}

/** Omitted BG/B pins are not independent simulator nodes. */
export function isCircuitNet(net: NetInfo, conn: Connectivity): boolean {
  return net.pins.some((p) => !isOptionalStlPin(p.el, p.pin) || optionalPinConnected(p.el.id, p.pin, conn));
}

export const pinId = (elId: string, pin: string) => `${elId}:${pin}`;

export function extractNets(doc: Pick<SchematicDoc, "elements" | "wires">): Connectivity {
  const uf = new UnionFind();
  const pins = doc.elements.flatMap(pinPositions);
  const wires = doc.wires.filter((w) => w.x1 !== w.x2 || w.y1 !== w.y2);
  const count = new Map<string, number>();
  const bump = (k: string, n = 1) => count.set(k, (count.get(k) ?? 0) + n);
  const GROUND = "__gnd__";

  for (const w of wires) {
    const a = ptKey(w.x1, w.y1);
    const b = ptKey(w.x2, w.y2);
    uf.union(a, b);
    bump(a);
    bump(b);
  }
  // `count` drives junction dots (net labels don't make a junction); `attach` finds unconnected pins
  const attach = new Map<string, number>();
  const nonLabel = new Map<string, number>();
  for (const p of pins) {
    const k = ptKey(p.x, p.y);
    uf.find(k);
    if (p.el.kind !== "LABEL") {
      bump(k);
      nonLabel.set(k, (nonLabel.get(k) ?? 0) + 1);
    }
    attach.set(k, (attach.get(k) ?? 0) + 1);
  }
  // T-junctions: a pin or wire end point on another wire's interior connects to it
  const probePts: Pt[] = [...pins.map((p) => ({ x: p.x, y: p.y })), ...wires.flatMap((w) => [{ x: w.x1, y: w.y1 }, { x: w.x2, y: w.y2 }])];
  const seen = new Set<string>();
  for (const pt of probePts) {
    const k = ptKey(pt.x, pt.y);
    if (seen.has(k)) continue;
    seen.add(k);
    for (const w of wires) {
      if (onWireInterior(w, pt.x, pt.y)) {
        uf.union(k, ptKey(w.x1, w.y1));
        bump(k, 2);
      }
    }
  }
  // labels and ground symbols
  const labelFirst = new Map<string, string>();
  for (const p of pins) {
    const k = ptKey(p.x, p.y);
    if (p.el.kind === "GND") uf.union(k, GROUND);
    else if (p.el.kind === "LABEL") {
      const name = (p.el.label ?? "").trim();
      if (!name) continue;
      if (isGroundName(name)) uf.union(k, GROUND);
      else {
        const f = labelFirst.get(name);
        if (f) uf.union(k, f);
        else labelFirst.set(name, k);
      }
    }
  }
  const groundRoot = uf.find(GROUND);

  // group
  const groups = new Map<string, { pins: PinPos[]; all: PinPos[]; wires: Wire[]; labels: Set<string>; pts: Pt[] }>();
  const g = (root: string) => {
    let x = groups.get(root);
    if (!x) {
      x = { pins: [], all: [], wires: [], labels: new Set(), pts: [] };
      groups.set(root, x);
    }
    return x;
  };
  for (const p of pins) {
    const grp = g(uf.find(ptKey(p.x, p.y)));
    grp.all.push(p);
    grp.pts.push({ x: p.x, y: p.y });
    if (p.el.kind === "LABEL" && p.el.label?.trim() && !isGroundName(p.el.label)) grp.labels.add(p.el.label.trim());
    else if (p.el.kind !== "GND" && p.el.kind !== "LABEL") grp.pins.push(p);
  }
  for (const w of wires) {
    const grp = g(uf.find(ptKey(w.x1, w.y1)));
    grp.wires.push(w);
    grp.pts.push({ x: w.x1, y: w.y1 }, { x: w.x2, y: w.y2 });
  }

  const nets: NetInfo[] = [];
  for (const [root, grp] of groups) {
    if (!grp.all.length && !grp.wires.length) continue;
    const ground = root === groundRoot && grp.all.some((p) => p.el.kind === "GND" || (p.el.kind === "LABEL" && isGroundName(p.el.label ?? "")));
    const labels = [...grp.labels].sort();
    nets.push({ id: root, name: "", ground, labels, pins: grp.pins, allPins: grp.all, wires: grp.wires, anchor: anchorOf(grp.wires, grp.all) });
  }
  // names: ground "0", labels, then N001… ordered by position (top-left first) for stability
  const used = new Set<string>(["0"]);
  for (const n of nets) {
    if (n.ground) n.name = "0";
    else if (n.labels.length) {
      n.name = n.labels[0];
      used.add(n.name);
    }
  }
  let k = 0;
  nets
    .filter((n) => !n.name)
    .sort((a, b) => a.anchor.y - b.anchor.y || a.anchor.x - b.anchor.x)
    .forEach((n) => {
      let name: string;
      do name = `N${String(++k).padStart(3, "0")}`;
      while (used.has(name));
      n.name = name;
      used.add(name);
    });

  const byName = new Map<string, NetInfo>();
  const pinNet = new Map<string, NetInfo>();
  const wireNet = new Map<string, NetInfo>();
  for (const n of nets) {
    // two ground groups cannot exist (all merge into GROUND); labelled groups share a root
    if (!byName.has(n.name)) byName.set(n.name, n);
    for (const p of n.allPins) pinNet.set(pinId(p.el.id, p.pin), n);
    for (const w of n.wires) wireNet.set(w.id, n);
  }

  const junctions: Pt[] = [];
  for (const [k2, c] of count) {
    if (c >= 3) {
      const [x, y] = k2.split(",").map(Number);
      junctions.push({ x, y });
    }
  }
  // unconnected: no wire and no other pin at the pin location
  const unconnected = pins.filter((p) => {
    const k = ptKey(p.x, p.y);
    const wiresHere = (count.get(k) ?? 0) - (nonLabel.get(k) ?? 0);
    return wiresHere + (attach.get(k) ?? 0) - 1 <= 0;
  });
  return { nets, byName, pinNet, wireNet, junctions, unconnected };
}

function anchorOf(wires: Wire[], pins: PinPos[]): Pt {
  const lab = pins.find((p) => p.el.kind === "LABEL");
  if (lab) return { x: lab.x, y: lab.y };
  let best: Wire | null = null;
  let bestLen = -1;
  for (const w of wires) {
    const len = Math.abs(w.x2 - w.x1) + Math.abs(w.y2 - w.y1);
    // prefer horizontal wires (room for a text label above them)
    const score = len + (w.y1 === w.y2 ? 1000 : 0);
    if (score > bestLen) {
      bestLen = score;
      best = w;
    }
  }
  if (best) return { x: (best.x1 + best.x2) / 2, y: (best.y1 + best.y2) / 2 };
  const p = pins[0];
  return p ? { x: p.x, y: p.y } : { x: 0, y: 0 };
}
