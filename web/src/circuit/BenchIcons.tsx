// Mini schematics for the bench selector cards (200 × 70 viewBox, currentColor strokes), and 20 px glyphs for
// the compact bench chips (24 × 24 viewBox): ramp, pulse train, comparator, coupled pair.
import type { BenchId } from "../api/types";

const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

function Zig({ x, y, w = 30 }: { x: number; y: number; w?: number }) {
  const pts = Array.from({ length: 9 }, (_, i) => `${x + (w * i) / 8},${y + (i === 0 || i === 8 ? 0 : i % 2 ? -5 : 5)}`).join(" ");
  return <polyline points={pts} {...S} />;
}
function Fet({ x, y }: { x: number; y: number }) {
  return (
    <g {...S}>
      <line x1={x} y1={y - 16} x2={x} y2={y - 8} />
      <line x1={x} y1={y + 8} x2={x} y2={y + 16} />
      <line x1={x - 5} y1={y - 10} x2={x - 5} y2={y + 10} strokeWidth={2.2} />
      <line x1={x - 5} y1={y - 8} x2={x} y2={y - 8} />
      <line x1={x - 5} y1={y + 8} x2={x} y2={y + 8} />
      <line x1={x - 10} y1={y - 8} x2={x - 10} y2={y + 8} strokeWidth={1.8} />
      <line x1={x - 10} y1={y} x2={x - 18} y2={y} />
    </g>
  );
}
const Gnd = ({ x, y }: { x: number; y: number }) => (
  <g {...S}>
    <line x1={x - 7} y1={y} x2={x + 7} y2={y} />
    <line x1={x - 4} y1={y + 3} x2={x + 4} y2={y + 3} />
  </g>
);

export function BenchIcon({ bench }: { bench: BenchId }) {
  switch (bench) {
    case "load_line":
      return (
        <svg viewBox="0 0 200 70" aria-hidden>
          <circle cx={30} cy={36} r={11} {...S} />
          <path d="M24 40 l4 -8 l4 8 l4 -8" {...S} strokeWidth={1.3} />
          <path d="M30 25 V12 H70" {...S} />
          <Zig x={70} y={12} />
          <path d="M100 12 H140 V20" {...S} />
          <Fet x={140} y={36} />
          <path d="M140 52 V60 M30 47 V60 M24 60 H146" {...S} />
          <Gnd x={85} y={63} />
        </svg>
      );
    case "pulse":
      return (
        <svg viewBox="0 0 200 70" aria-hidden>
          <path d="M8 44 H18 V26 H28 V44 H38 V26 H48 V44 H58" {...S} />
          <path d="M58 44 H62 V12 H80" {...S} />
          <Zig x={80} y={12} />
          <path d="M110 12 H146 V20" {...S} />
          <Fet x={146} y={36} />
          <path d="M146 52 V60 M62 44 V60 M62 60 H152" {...S} />
          <Gnd x={104} y={63} />
          <text x={168} y={40} fontSize={11} fill="currentColor" fontFamily="var(--mono)">Q_B</text>
        </svg>
      );
    case "pbit":
      // drain pulses → STL; source → R_S → ground; comparator on the source node → random firing
      return (
        <svg viewBox="0 0 200 70" aria-hidden>
          <path d="M6 18 H12 V6 H22 V18 H28 V6 H38 V18 H44" {...S} strokeWidth={1.3} />
          <path d="M44 18 H56 V6 H80 V14" {...S} />
          <Fet x={80} y={30} />
          <polyline points="80,46 80,48 75,50 85,53 75,56 85,59 80,61 80,63" {...S} />
          <Gnd x={80} y={64} />
          <path d="M80 47 H112" {...S} />
          <path d="M112 35 L136 47 L112 59 Z" {...S} />
          <path d="M136 47 H146 M146 47 H152 V36 H160 V58 H166 V36 H174 V58 H180 V47 H194" {...S} strokeWidth={1.3} />
        </svg>
      );
    case "coupled":
      return (
        <svg viewBox="0 0 200 70" aria-hidden>
          <path d="M20 10 H176" {...S} opacity={0.6} />
          <path d="M50 10 V18 M150 10 V18" {...S} />
          <Fet x={50} y={38} />
          <Fet x={150} y={38} />
          <path d="M50 54 V62 M150 54 V62 M44 62 H156" {...S} />
          <path d="M50 26 H80" {...S} />
          <Zig x={80} y={26} w={40} />
          <path d="M120 26 H150" {...S} />
          <text x={92} y={44} fontSize={10} fill="currentColor" fontFamily="var(--mono)">R_c</text>
        </svg>
      );
  }
}

const G = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

/** 20 px glyph of a bench (compact chips): what drives the STL, at a glance. */
export function BenchGlyph({ bench, size = 20 }: { bench: BenchId; size?: number }) {
  const d =
    bench === "load_line"
      ? "M3 19 L12 5 L21 19 M2 20 H22" // triangle ramp 0 → V_max → 0
      : bench === "pulse"
        ? "M2 18 H5 V7 H9 V18 H13 V7 H17 V18 H22" // pulse train
        : bench === "pbit"
          ? "M4 5 L16 12 L4 19 Z M16 12 H18 V8 H21 M8 10 V14 M6 12 H10" // comparator → bit
          : "M4 9 H19 M15 5 L19 9 L15 13 M20 15 H5 M9 11 L5 15 L9 19"; // coupled pair ⇄
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden focusable="false" className="bench-glyph">
      <path d={d} {...G} />
    </svg>
  );
}
