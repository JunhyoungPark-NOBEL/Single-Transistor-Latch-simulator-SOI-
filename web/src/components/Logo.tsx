// Brand mark: the biristor symbol — an NPN bipolar-transistor symbol in a circle with the collector (C) going
// up, the emitter going down with the outward NPN arrow, and the base (B) left open: the biristor is a
// gateless two-terminal n⁺/p/n⁺ device whose base (the floating body) has no contact. White on the
// rounded-square badge; the badge gradient runs teal → indigo → violet (deterministic → stochastic accents,
// tokens --logo-a/b/c in app.css), so it sits well on the light and the dark theme.
// Detail follows the rendered size: < 24 px heavier strokes and the bare base bar (favicon); 24–39 px adds the
// open base lead (stub ending in a terminal ring); ≥ 40 px adds small C and B letters. Same geometry as
// public/favicon.svg (the < 24 px form) — keep them in sync. `BiristorGlyph` is the symbol alone, monochrome.
import { useId, type CSSProperties } from "react";

type Level = 0 | 1 | 2;
const levelOf = (size: number): Level => (size < 24 ? 0 : size < 40 ? 1 : 2);

/** Symbol geometry in the 64 × 64 badge box: [path d, stroke width]. */
const GEOM = {
  // ≥ 24 px: circle r 18, base bar at x 24.5, leads at x 40 from the top edge to the bottom edge
  full: {
    circle: { r: 18, sw: 3.1 },
    bar: ["M24.5 22V42", 5],
    collector: ["M24.5 27L40 19V3.5", 3.6],
    emitter: ["M24.5 37L33.03 41.4M40 45V60.5", 3.6],
    arrow: ["M39.69 44.84L30.35 44.41L33.93 37.48Z", 1.6],
  },
  // < 24 px: larger circle, ~1.45× strokes, shorter leads, no base lead
  small: {
    circle: { r: 19, sw: 4.5 },
    bar: ["M24 21.5V42.5", 7.25],
    collector: ["M24 27L40 18.5V6", 5.2],
    emitter: ["M24 37L31.82 41.15M40 45.5V58", 5.2],
    arrow: ["M39.68 45.33L28.73 44.84L33.14 36.53Z", 2.3],
  },
} as const;
/** Open base lead: stub from the bar to a hollow terminal ring (floating base). */
const BASE_LEAD = { stub: "M22 32H10.7", stubSw: 3.05, ring: { cx: 8, cy: 32, r: 2.7, sw: 2.5 } };
/** Letters as strokes (no font dependency): C right of the collector lead, B above the base ring. */
const LETTERS = { d: "M52.64 4.77A4.5 4.5 0 1 0 52.64 11.23M5 26V17H7.7A2.16 2.16 0 0 1 7.7 21.32H5M7.7 21.32H8.24A2.34 2.34 0 0 1 8.24 26H5", sw: 2 };

/** The symbol strokes (color = stroke), used by the badge and by the monochrome glyph. */
function BiristorPaths({ level, color }: { level: Level; color: string }) {
  const g = level === 0 ? GEOM.small : GEOM.full;
  return (
    <>
      <circle cx="32" cy="32" r={g.circle.r} strokeWidth={g.circle.sw} />
      <path d={g.bar[0]} strokeWidth={g.bar[1]} />
      <path d={g.collector[0]} strokeWidth={g.collector[1]} />
      <path d={g.emitter[0]} strokeWidth={g.emitter[1]} />
      <path d={g.arrow[0]} strokeWidth={g.arrow[1]} fill={color} />
      {level >= 1 && (
        <>
          <path d={BASE_LEAD.stub} strokeWidth={BASE_LEAD.stubSw} />
          <circle cx={BASE_LEAD.ring.cx} cy={BASE_LEAD.ring.cy} r={BASE_LEAD.ring.r} strokeWidth={BASE_LEAD.ring.sw} />
        </>
      )}
    </>
  );
}

export function Logo({ size = 32, className, title }: { size?: number; className?: string; title?: string }) {
  const id = useId().replace(/:/g, "");
  const g = `logo-g-${id}`;
  const h = `logo-h-${id}`;
  const level = levelOf(size);
  const r = level === 0 ? GEOM.small.circle.r : GEOM.full.circle.r;
  return (
    <svg
      className={`logo${className ? ` ${className}` : ""}`}
      width={size}
      height={size}
      viewBox="0 0 64 64"
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
      data-level={level}
    >
      <defs>
        <linearGradient id={g} x1="4" y1="2" x2="60" y2="62" gradientUnits="userSpaceOnUse">
          <stop offset="0" style={{ stopColor: "var(--logo-a, #10ab9e)" }} />
          <stop offset="0.52" style={{ stopColor: "var(--logo-b, #4f6bef)" }} />
          <stop offset="1" style={{ stopColor: "var(--logo-c, #7c3aed)" }} />
        </linearGradient>
        <radialGradient id={h} cx="14" cy="6" r="50" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#fff" stopOpacity="0.3" />
          <stop offset="0.7" stopColor="#fff" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill={`url(#${g})`} />
      <rect width="64" height="64" rx="15" fill={`url(#${h})`} />
      <rect x="0.75" y="0.75" width="62.5" height="62.5" rx="14.25" fill="none" stroke="#fff" strokeOpacity="0.18" strokeWidth="1.5" />
      <g fill="none" strokeLinecap="round" strokeLinejoin="round">
        {/* soft drop shadow of the symbol */}
        <g stroke="#0b1020" strokeOpacity="0.16" fillOpacity="0.16" transform="translate(0 1.2)">
          <BiristorPaths level={level} color="#0b1020" />
        </g>
        <circle cx="32" cy="32" r={r} fill="#fff" fillOpacity="0.1" />
        <g stroke="#fff">
          <BiristorPaths level={level} color="#fff" />
          {level === 2 && <path d={LETTERS.d} strokeWidth={LETTERS.sw} />}
        </g>
      </g>
    </svg>
  );
}

/** The biristor symbol alone, monochrome (currentColor), e.g. as a small inline device icon. */
export function BiristorGlyph({ size = 18, className, title, letters = false, style }: { size?: number; className?: string; title?: string; letters?: boolean; style?: CSSProperties }) {
  const level = levelOf(size);
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="2 1 60 62"
      style={style}
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {title && <title>{title}</title>}
      <g fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
        <BiristorPaths level={level} color="currentColor" />
        {letters && level === 2 && <path d={LETTERS.d} strokeWidth={LETTERS.sw} />}
      </g>
    </svg>
  );
}
