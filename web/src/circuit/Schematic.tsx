// Generic schematic renderer for CircuitResult.schematic ({nodes, elements}). Layout: non-ground nodes as
// columns on a top rail, two-terminal elements to ground drawn vertically, node-to-node elements drawn
// horizontally (stacked on higher rails when they skip columns), STL elements vertically at the drain
// column with the gate as a net label. Ground is the bottom rail.
import type { ReactNode } from "react";
import type { SchematicElement } from "../api/types";

const GROUND = new Set(["0", "gnd", "GND", "ground", "vss", "VSS"]);
const isGnd = (n: string) => GROUND.has(n);

const TOP = 70;
const BOT = 250;
const MID = (TOP + BOT) / 2;

function Resistor({ x, y, vertical }: { x: number; y: number; vertical: boolean }) {
  const pts: string[] = [];
  const L = 44;
  for (let i = 0; i <= 8; i++) {
    const s = -L / 2 + (L * i) / 8;
    const o = i === 0 || i === 8 ? 0 : i % 2 ? 7 : -7;
    pts.push(vertical ? `${x + o},${y + s}` : `${x + s},${y + o}`);
  }
  return <polyline points={pts.join(" ")} fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinejoin="round" />;
}
function Capacitor({ x, y, vertical }: { x: number; y: number; vertical: boolean }) {
  return vertical ? (
    <g stroke="currentColor" strokeWidth={2}>
      <line x1={x - 12} y1={y - 4} x2={x + 12} y2={y - 4} />
      <line x1={x - 12} y1={y + 4} x2={x + 12} y2={y + 4} />
    </g>
  ) : (
    <g stroke="currentColor" strokeWidth={2}>
      <line x1={x - 4} y1={y - 12} x2={x - 4} y2={y + 12} />
      <line x1={x + 4} y1={y - 12} x2={x + 4} y2={y + 12} />
    </g>
  );
}
function Source({ x, y, kind }: { x: number; y: number; kind: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={15} fill="var(--surface)" stroke="currentColor" strokeWidth={1.8} />
      {kind === "I" ? (
        <path d={`M${x} ${y + 8} V${y - 7} M${x - 4} ${y - 3} L${x} ${y - 8} L${x + 4} ${y - 3}`} stroke="currentColor" strokeWidth={1.6} fill="none" />
      ) : (
        <g stroke="currentColor" strokeWidth={1.5}>
          <line x1={x - 4} y1={y - 6} x2={x + 4} y2={y - 6} />
          <line x1={x} y1={y - 10} x2={x} y2={y - 2} />
          <line x1={x - 4} y1={y + 7} x2={x + 4} y2={y + 7} />
        </g>
      )}
    </g>
  );
}
function Comparator({ x, y }: { x: number; y: number }) {
  return <path d={`M${x - 16} ${y - 18} L${x + 18} ${y} L${x - 16} ${y + 18} Z`} fill="var(--surface)" stroke="currentColor" strokeWidth={1.8} />;
}
function Stl({ x, y, name, gate }: { x: number; y: number; name: string; gate: string }) {
  // drain at (x, y-30), source at (x, y+30), gate terminal at (x-38, y)
  return (
    <g>
      <line x1={x} y1={y - 30} x2={x} y2={y - 14} stroke="currentColor" strokeWidth={1.8} />
      <line x1={x} y1={y + 14} x2={x} y2={y + 30} stroke="currentColor" strokeWidth={1.8} />
      <line x1={x - 6} y1={y - 16} x2={x - 6} y2={y + 16} stroke="currentColor" strokeWidth={2.4} />
      <line x1={x - 6} y1={y - 14} x2={x} y2={y - 14} stroke="currentColor" strokeWidth={1.8} />
      <line x1={x - 6} y1={y + 14} x2={x} y2={y + 14} stroke="currentColor" strokeWidth={1.8} />
      <path d={`M${x - 5} ${y + 14} l5 -3 l0 6 z`} fill="currentColor" transform={`rotate(0 ${x} ${y})`} />
      <line x1={x - 13} y1={y - 12} x2={x - 13} y2={y + 12} stroke="currentColor" strokeWidth={2} />
      <line x1={x - 13} y1={y} x2={x - 36} y2={y} stroke="currentColor" strokeWidth={1.8} />
      <circle cx={x - 38} cy={y} r={2.6} fill="var(--surface)" stroke="currentColor" strokeWidth={1.4} />
      <text x={x - 44} y={y + 4} textAnchor="end" fontSize={11} fill="var(--muted)" fontFamily="var(--mono)">{gate}</text>
      <rect x={x - 22} y={y - 22} width={30} height={44} rx={5} fill="none" stroke="var(--accent)" strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
      <text x={x + 10} y={y - 4} fontSize={12} fontWeight={700} fill="currentColor">{name}</text>
      <text x={x + 10} y={y + 11} fontSize={10.5} fill="var(--accent-strong)" fontWeight={700}>STL</text>
    </g>
  );
}

function Label({ x, y, name, value, anchor = "start" }: { x: number; y: number; name: string; value?: string; anchor?: "start" | "middle" | "end" }) {
  return (
    <g>
      <text x={x} y={y} fontSize={12} fontWeight={700} fill="currentColor" textAnchor={anchor}>{name}</text>
      {value && <text x={x} y={y + 13} fontSize={10.5} fill="var(--muted)" textAnchor={anchor} fontFamily="var(--mono)">{value}</text>}
    </g>
  );
}

function GroundSym({ x }: { x: number }) {
  return (
    <g stroke="currentColor" strokeWidth={1.6}>
      <line x1={x - 10} y1={BOT} x2={x + 10} y2={BOT} />
      <line x1={x - 6} y1={BOT + 4} x2={x + 6} y2={BOT + 4} />
      <line x1={x - 2.5} y1={BOT + 8} x2={x + 2.5} y2={BOT + 8} />
    </g>
  );
}

export function Schematic({ nodes, elements, title }: { nodes: string[]; elements: SchematicElement[]; title?: string }) {
  const gateNodes = new Set(elements.filter((e) => e.kind === "STL" && e.nodes[1]).map((e) => e.nodes[1]));
  const allNodes = [...nodes, ...elements.flatMap((e) => e.nodes)].filter((n, i, a) => a.indexOf(n) === i && !isGnd(n));
  const main = allNodes.filter((n) => !gateNodes.has(n));
  const gates = allNodes.filter((n) => gateNodes.has(n));
  const cols = [...main, ...gates];

  // vertical elements per column
  type Placed = { el: SchematicElement; col: string; slot: number };
  const vertical: Placed[] = [];
  const horizontal: { el: SchematicElement; a: string; b: string }[] = [];
  const slots = new Map<string, number>();
  // STL elements first at their column (their gate lead needs the free space left of the column)
  const ordered = [...elements].sort((a, b) => Number(b.kind === "STL") - Number(a.kind === "STL"));
  for (const el of ordered) {
    const ns = el.nodes;
    if (el.kind === "STL") {
      const col = ns[0];
      const k = slots.get(col) ?? 0;
      slots.set(col, k + 1);
      vertical.push({ el, col, slot: k });
    } else if (ns.length >= 2 && (isGnd(ns[0]) || isGnd(ns[1]))) {
      const col = isGnd(ns[0]) ? ns[1] : ns[0];
      const k = slots.get(col) ?? 0;
      slots.set(col, k + 1);
      vertical.push({ el, col, slot: k });
    } else if (ns.length >= 2) {
      horizontal.push({ el, a: ns[0], b: ns[1] });
    }
  }
  const SLOT = 78;
  const GAP = 120;
  const xs = new Map<string, number>();
  let x = 46;
  cols.forEach((n, i) => {
    if (i > 0 && gates.includes(n) && !gates.includes(cols[i - 1])) x += 30;
    xs.set(n, x);
    x += Math.max(1, slots.get(n) ?? 0) * SLOT - SLOT + GAP;
  });
  const width = Math.max(320, x - GAP + 110);

  // horizontal elements: adjacent columns on the top rail, others on stacked rails
  const hLevels: number[] = [];
  const hPlaced = horizontal.map((h) => {
    const xa = xs.get(h.a) ?? 0;
    const xb = xs.get(h.b) ?? 0;
    const ia = cols.indexOf(h.a);
    const ib = cols.indexOf(h.b);
    const adjacent = Math.abs(ia - ib) === 1;
    let level = 0;
    if (!adjacent) {
      level = 1;
      while (hLevels.includes(level)) level++;
      hLevels.push(level);
    }
    return { ...h, xa, xb, level };
  });
  const maxLevel = Math.max(0, ...hPlaced.map((h) => h.level));
  const yOff = maxLevel * 56;
  const height = BOT + 34 + yOff;
  const railY = (lvl: number) => TOP - lvl * 56;

  const parts: ReactNode[] = [];
  // ground rail
  const gndUsed = vertical.some((v) => v.el.kind === "STL" ? isGnd(v.el.nodes[2] ?? "0") : true);
  if (gndUsed) {
    const vx = vertical.map((v) => (xs.get(v.col) ?? 0) + v.slot * SLOT);
    const gx1 = Math.min(...vx);
    const gx2 = Math.max(...vx);
    parts.push(<line key="gnd" x1={gx1} y1={BOT} x2={gx2} y2={BOT} stroke="currentColor" strokeWidth={1.6} />);
    parts.push(<GroundSym key="gnds" x={(gx1 + gx2) / 2} />);
  }
  // node rails (top) + dots + names
  cols.forEach((n) => {
    const x0 = xs.get(n)!;
    const w = (Math.max(1, slots.get(n) ?? 0) - 1) * SLOT;
    if (w > 0) parts.push(<line key={`rail-${n}`} x1={x0} y1={TOP} x2={x0 + w} y2={TOP} stroke="currentColor" strokeWidth={1.6} />);
    parts.push(<circle key={`dot-${n}`} cx={x0} cy={TOP} r={3.4} fill="currentColor" />);
    parts.push(
      <text key={`nn-${n}`} x={x0} y={TOP - 10} fontSize={11} textAnchor="middle" fill="var(--accent-strong)" fontFamily="var(--mono)" fontWeight={700}>
        {n}
      </text>,
    );
  });
  // vertical elements
  vertical.forEach(({ el, col, slot }) => {
    const vx = (xs.get(col) ?? 0) + slot * SLOT;
    if (el.kind === "STL") {
      const sGnd = isGnd(el.nodes[2] ?? "0");
      parts.push(<line key={`${el.name}-d`} x1={vx} y1={TOP} x2={vx} y2={MID - 30} stroke="currentColor" strokeWidth={1.6} />);
      parts.push(<line key={`${el.name}-s`} x1={vx} y1={MID + 30} x2={vx} y2={sGnd ? BOT : MID + 52} stroke="currentColor" strokeWidth={1.6} />);
      if (!sGnd) parts.push(<text key={`${el.name}-sn`} x={vx} y={MID + 64} textAnchor="middle" fontSize={11} fill="var(--muted)">{el.nodes[2]}</text>);
      parts.push(<Stl key={el.name} x={vx} y={MID} name={el.name} gate={el.nodes[1] ?? "g"} />);
    } else {
      parts.push(<line key={`${el.name}-w`} x1={vx} y1={TOP} x2={vx} y2={BOT} stroke="currentColor" strokeWidth={1.6} />);
      const sym =
        el.kind === "R" ? <Resistor x={vx} y={MID} vertical /> : el.kind === "C" ? <Capacitor x={vx} y={MID} vertical /> : el.kind === "CMP" ? <Comparator x={vx} y={MID} /> : <Source x={vx} y={MID} kind={el.kind} />;
      parts.push(
        <g key={el.name}>
          <rect x={vx - 16} y={MID - 26} width={32} height={52} fill="var(--surface)" />
          {sym}
          <Label x={vx + 20} y={MID - 2} name={el.name} value={el.value} />
        </g>,
      );
    }
  });
  // horizontal elements
  hPlaced.forEach((h) => {
    const y = railY(h.level);
    const x1 = Math.min(h.xa, h.xb);
    const x2 = Math.max(h.xa, h.xb);
    if (h.level > 0) {
      parts.push(<line key={`${h.el.name}-u1`} x1={x1} y1={TOP} x2={x1} y2={y} stroke="currentColor" strokeWidth={1.6} />);
      parts.push(<line key={`${h.el.name}-u2`} x1={x2} y1={TOP} x2={x2} y2={y} stroke="currentColor" strokeWidth={1.6} />);
    }
    const startX = h.level === 0 ? x1 + (Math.max(1, slots.get(h.xa < h.xb ? h.a : h.b) ?? 1) - 1) * SLOT : x1;
    parts.push(<line key={`${h.el.name}-w`} x1={startX} y1={y} x2={x2} y2={y} stroke="currentColor" strokeWidth={1.6} />);
    const mid = (startX + x2) / 2;
    const sym = h.el.kind === "R" ? <Resistor x={mid} y={y} vertical={false} /> : h.el.kind === "C" ? <Capacitor x={mid} y={y} vertical={false} /> : h.el.kind === "CMP" ? <Comparator x={mid} y={y} /> : <Source x={mid} y={y} kind={h.el.kind} />;
    parts.push(
      <g key={h.el.name}>
        <rect x={mid - 26} y={y - 16} width={52} height={32} fill="var(--surface)" />
        {sym}
        <Label x={mid} y={y - 22} name={h.el.name} value={undefined} anchor="middle" />
        {h.el.value && <text x={mid} y={y + 30} fontSize={10.5} fill="var(--muted)" textAnchor="middle" fontFamily="var(--mono)">{h.el.value}</text>}
      </g>,
    );
  });
  return (
    <svg viewBox={`0 ${-yOff} ${width} ${height}`} role="img" aria-label={title ?? "schematic"} data-testid="schematic" style={{ color: "var(--text)" }}>
      {parts}
    </svg>
  );
}
