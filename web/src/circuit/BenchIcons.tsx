// Mini schematics for the bench selector cards (200 × 70 viewBox, currentColor strokes).
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
      return (
        <svg viewBox="0 0 200 70" aria-hidden>
          <path d="M20 12 H40" {...S} />
          <text x={6} y={16} fontSize={10} fill="currentColor">V</text>
          <Zig x={40} y={12} />
          <path d="M70 12 H90 V20" {...S} />
          <Fet x={90} y={36} />
          <path d="M90 52 V60 M84 60 H96" {...S} />
          <path d="M90 16 H120" {...S} />
          <path d="M120 4 L146 16 L120 28 Z" {...S} />
          <path d="M146 16 H158 M158 16 V30 H166 V8 H174 V30 H182 V16 H192" {...S} strokeWidth={1.3} />
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
