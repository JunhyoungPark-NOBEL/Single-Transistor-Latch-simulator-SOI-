// Schematic symbols drawn in local coordinates (unrotated: two-terminal parts vertical with the first
// node on top; STL drain up, gate left, source down). Text is drawn upright outside the rotation.
import { memo } from "react";
import { BiristorGlyph } from "../components/Logo";
import type { Wave } from "../api/circuitCustom";
import { labelWidth, rotatePt, type ElKind, type SElement } from "./model";
import { fmtSI } from "./si";

const Z = "M0 -40 V-26 L7 -22 L-7 -14 L7 -6 L-7 2 L7 10 L-7 18 L0 22 V40"; // resistor zigzag

function Body({ kind, polarity }: { kind: ElKind; polarity?: string }) {
  switch (kind) {
    case "R":
      return <path d={Z} className="sch-stroke" />;
    case "C":
      return (
        <>
          <path d="M0 -40 V-6 M0 6 V40" className="sch-stroke" />
          <path d="M-15 -6 H15 M-15 6 H15" className="sch-stroke thick" />
        </>
      );
    case "V":
      return (
        <>
          <path d="M0 -40 V-18 M0 18 V40" className="sch-stroke" />
          <circle r={18} className="sch-stroke sch-fill" />
        </>
      );
    case "I":
      return (
        <>
          <path d="M0 -40 V-18 M0 18 V40" className="sch-stroke" />
          <circle r={18} className="sch-stroke sch-fill" />
          <path d="M0 -10 V9 M-5 3 L0 10 L5 3" className="sch-stroke" />
        </>
      );
    case "GND":
      return <path d="M0 0 V8 M-13 8 H13 M-8 13 H8 M-3 18 H3" className="sch-stroke" />;
    case "CMP":
      return (
        <>
          {/* comparator: leads, triangle, + input mark and a step (digital output) mark */}
          <path d="M-40 0 H-22 M22 0 H40" className="sch-stroke" />
          <path d="M-22 -24 L22 0 L-22 24 Z" className="sch-stroke sch-fill" />
          <path d="M-17 -9 H-9 M-13 -13 V-5 M-17 9 H-9" className="sch-stroke thin" />
          <path d="M-4 5 H1 V-5 H8" className="sch-accent" />
        </>
      );
    case "D": return <><path d="M0 -40 V-12 M0 12 V40 M-13 -12 L0 12 L13 -12 Z M-13 12 H13" className="sch-stroke" /></>;
    case "MOS": return <>
      <path d="M0 -40 V-18 H-9 M0 40 V18 H-9 M-40 0 H-19 M-19 -20 V20 M-9 -23 V23" className="sch-stroke" />
      <path d={polarity === "pmos" ? "M-2 18 L-9 14 L-9 22 Z" : "M-9 18 L-2 14 L-2 22 Z"} className="sch-solid" />
    </>;
    case "BJT": return <>
      <circle cx={-1} r={24} className="sch-stroke sch-fill" />
      <path d="M0 -40 V-22 L-13 -8 M0 40 V22 L-13 8 M-40 0 H-13 M-13 -15 V15" className="sch-stroke" />
      <path d={polarity === "pnp" ? "M-10 11 L-2 12 L-7 19 Z" : "M-1 22 L-9 19 L-3 13 Z"} className="sch-solid" />
    </>;
    case "STL":
      return (
        <>
          <path d="M0 -40 H31 V-16 H0 M0 40 H-31 V16 H0 M-40 0 H-24" className="sch-stroke" />
          <circle r={24} className="sch-stroke sch-fill" />
          <path d="M0 -16 H13 L0 16 H-13 Z M0 -16 H31 M-31 16 H0" className="sch-accent" />
        </>
      );
    case "LABEL":
      return null;
  }
}

const sign = (x: number, y: number, minus: boolean) => (minus ? `M${x - 4} ${y} H${x + 4}` : `M${x - 4} ${y} H${x + 4} M${x} ${y - 4} V${y + 4}`);

export function waveShort(w: Wave | null | undefined, unit: string): string {
  if (!w) return "";
  switch (w.kind) {
    case "dc":
      return fmtSI(w.value, unit, 3);
    case "pulse":
      return `PULSE ${fmtSI(w.v1, "", 3).trim()}→${fmtSI(w.v2, unit, 3)}`;
    case "pwl":
      return `PWL · ${w.t.length} pts`;
    case "sine":
      return `SINE ${fmtSI(w.va, unit, 2)} · ${fmtSI(w.freq, "Hz", 3)}`;
  }
}

const shortName = (s: string, n = 22) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

export function elementText(el: SElement): string[] {
  switch (el.kind) {
    case "R":
      return [el.name, fmtSI(el.value, "Ω", 3)];
    case "C":
      return [el.name, fmtSI(el.value, "F", 3)];
    case "V":
      return [el.name, waveShort(el.wave, "V")];
    case "I":
      return [el.name, waveShort(el.wave, "A")];
    case "MOS": return [el.name, `${el.mos?.polarity === "pmos" ? "PMOS" : "NMOS"} · ${el.mos?.W_um ?? 10}/${el.mos?.L_um ?? 1} µm`];
    case "D": return [el.name];
    case "BJT": return [el.name, el.bjt?.polarity.toUpperCase() ?? "NPN"];
    case "STL":
      return [el.name, shortName(el.stl?.name ?? "—")];
    case "CMP":
      return [el.name, `V_ref ${fmtSI(el.cmp?.v_ref ?? 0, "V", 3)}`];
    default:
      return [];
  }
}

const TEXT_OFF: Partial<Record<ElKind, number>> = { R: 16, C: 22, V: 26, I: 26, STL: 34, MOS: 18, D: 22, BJT: 30, CMP: 30 };

export interface ElementViewProps {
  el: SElement;
  selected?: boolean;
  hovered?: boolean;
  flagged?: boolean;
  ghost?: boolean;
  probing?: boolean;
}

function ElementViewImpl({ el, selected, hovered, flagged, ghost, probing }: ElementViewProps) {
  const cls = ["sch-el", selected && "sel", hovered && "hov", flagged && "flag", ghost && "ghost", probing && "probing"].filter(Boolean).join(" ");
  const rot = el.rot * 90;
  const tf = `translate(${el.x} ${el.y}) rotate(${rot})${el.mirror ? " scale(-1 1)" : ""}`;
  if (el.kind === "LABEL") {
    const text = el.label || "?";
    const w = labelWidth(text);
    // stem direction follows the rotation (up, right, down, left); the tag itself stays upright
    const dir = rotatePt(0, -1, el.rot, el.mirror);
    const sx = el.x + dir.x * 8;
    const sy = el.y + dir.y * 8;
    const tx = dir.x < 0 ? sx - w : dir.x > 0 ? sx : sx - w / 2;
    const ty = dir.y < 0 ? sy - 18 : dir.y > 0 ? sy : sy - 9;
    return (
      <g className={cls} data-el={ghost ? undefined : el.id} data-kind={ghost ? undefined : "LABEL"}>
        <line x1={el.x} y1={el.y} x2={sx} y2={sy} className="sch-stroke" />
        <circle cx={el.x} cy={el.y} r={2.4} className="sch-solid" />
        <rect x={tx} y={ty} width={w} height={18} rx={9} className="sch-tag" />
        <text x={tx + w / 2} y={ty + 12.5} textAnchor="middle" className="sch-tag-text">
          {text}
        </text>
      </g>
    );
  }
  const lines = elementText(el);
  const off = TEXT_OFF[el.kind] ?? 20;
  // horizontal parts (comparator) carry their text above the body, the vertical ones to the right
  const o = el.kind === "CMP" ? rotatePt(0, -off, el.rot, el.mirror) : rotatePt(off, 0, el.rot, el.mirror);
  let texts: { x: number; y: number; anchor: "start" | "middle" | "end"; t: string; k: number }[] = [];
  if (lines.length) {
    if (Math.abs(o.x) >= Math.abs(o.y)) {
      const anchor = o.x >= 0 ? "start" : "end";
      texts = lines.map((t, k) => ({ x: el.x + o.x, y: el.y - (lines.length - 1) * 7 + k * 14 + 4, anchor, t, k }));
    } else {
      const below = o.y > 0;
      texts = lines.map((t, k) => ({ x: el.x, y: below ? el.y + o.y + 12 + k * 14 : el.y + o.y - 6 - (lines.length - 1 - k) * 14, anchor: "middle", t, k }));
    }
  }
  const signs =
    el.kind === "V"
      ? (() => {
          const p = rotatePt(0, -8, el.rot, el.mirror);
          const m = rotatePt(0, 8, el.rot, el.mirror);
          return <path d={`${sign(el.x + p.x, el.y + p.y, false)} ${sign(el.x + m.x, el.y + m.y, true)}`} className="sch-stroke thin" />;
        })()
      : null;
  const pinLetters =
    el.kind === "STL" || el.kind === "MOS" || el.kind === "BJT"
      ? ([["D", 7, -30], ["G", -33, -6], ["S", 7, 34]] as const).map(([l, x, y]) => {
          const p = rotatePt(x, y, el.rot, el.mirror);
          return (
            <text key={l} x={el.x + p.x} y={el.y + p.y + 3} textAnchor="middle" className="sch-pin-letter">
              {el.kind === "BJT" ? ({ D: "C", G: "B", S: "E" } as const)[l] : l}
            </text>
          );
        })
      : null;
  return (
    <g className={cls} data-el={ghost ? undefined : el.id} data-kind={ghost ? undefined : el.kind}>
      <g transform={tf}>
        <Body kind={el.kind} polarity={el.mos?.polarity ?? el.bjt?.polarity} />
      </g>
      {signs}
      {pinLetters}
      {texts.map((x) => (
        <text key={x.k} x={x.x} y={x.y} textAnchor={x.anchor} className={x.k === 0 ? "sch-name" : "sch-value"}>
          {x.t}
        </text>
      ))}
    </g>
  );
}

export const ElementView = memo(ElementViewImpl);

/** Small toolbar icons (24×24 viewBox) for each part. */
export function PartIcon({ kind, size = 18 }: { kind: ElKind | "wire" | "select" | "probe"; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  switch (kind) {
    case "select":
      return <svg {...common}><path d="M6 4 L6 18 L10 14 L13 20 L15.5 19 L12.5 13 L18 13 Z" fill="currentColor" fillOpacity={0.12} /></svg>;
    case "wire":
      return <svg {...common}><path d="M4 18 H11 V6 H20" /><circle cx={4} cy={18} r={1.6} fill="currentColor" /><circle cx={20} cy={6} r={1.6} fill="currentColor" /></svg>;
    case "probe":
      return <svg {...common}><path d="M14 4 L20 10 L11 19 L6 20 L5 19 L6 14 Z" /><path d="M3 21 L6 18" /></svg>;
    case "R":
      return <svg {...common}><path d="M2 12 H5 L7 8 L10 16 L13 8 L16 16 L18 12 H22" /></svg>;
    case "C":
      return <svg {...common}><path d="M2 12 H10 M14 12 H22" /><path d="M10 5 V19 M14 5 V19" strokeWidth={2.2} /></svg>;
    case "V":
      return <svg {...common}><circle cx={12} cy={12} r={8} /><path d="M12 7.5 V11.5 M10 9.5 H14 M10 15 H14" /></svg>;
    case "I":
      return <svg {...common}><circle cx={12} cy={12} r={8} /><path d="M12 7 V17 M9 13.5 L12 17 L15 13.5" /></svg>;
    case "GND":
      return <svg {...common}><path d="M12 4 V12 M5 12 H19 M8 16 H16 M11 20 H13" /></svg>;
    case "LABEL":
      return <svg {...common}><path d="M3 8 H15 L20 12 L15 16 H3 Z" /><path d="M7 12 H12" /></svg>;
    case "STL": return <BiristorGlyph size={size} />;
    case "MOS": return <svg {...common}><path d="M15 2 V7 H10 M15 22 V17 H10 M2 12 H6 M6 6 V18 M10 5 V19 M15 17 L11 15 V19 Z" /></svg>;
    case "D": return <svg {...common}><path d="M12 2 V6 M12 18 V22 M5 6 H19 L12 18 Z M5 18 H19" /></svg>;
    case "BJT": return <svg {...common}><circle cx={13} cy={12} r={9} /><path d="M1 12 H9 M9 6 V18 M9 9 L16 5 M9 15 L16 19 M16 19 L12 18 L14 15 Z" /></svg>;
    case "CMP":
      return (
        <svg {...common}>
          <path d="M2 12 H6 M18 12 H22" />
          <path d="M6 4 L18 12 L6 20 Z" />
          <path d="M8.5 9 H11.5 M10 7.5 V10.5" strokeWidth={1.3} />
        </svg>
      );
  }
}
