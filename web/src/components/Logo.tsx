// Brand mark: a stylised biristor-style two-terminal symbol (terminal leads + body) whose body holds the
// S-shaped latch characteristic (HRS branch → fold → unstable branch, faded → fold → LRS branch) on a
// rounded-square badge. The badge gradient runs teal → indigo → violet (deterministic → stochastic accents,
// tokens --logo-a/b/c in app.css), so it sits well on the light and the dark theme. Same geometry as
// public/favicon.svg — keep them in sync. Crisp from 16 px (favicon) to 72 px (About).
import { useId } from "react";

export function Logo({ size = 32, className, title }: { size?: number; className?: string; title?: string }) {
  const id = useId().replace(/:/g, "");
  const g = `logo-g-${id}`;
  const h = `logo-h-${id}`;
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
    >
      <defs>
        <linearGradient id={g} x1="4" y1="2" x2="60" y2="62" gradientUnits="userSpaceOnUse">
          <stop offset="0" style={{ stopColor: "var(--logo-a, #12b3a6)" }} />
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
        <g stroke="#0b1020" strokeOpacity="0.16" transform="translate(0 1.4)">
          <path d="M6.5 32H15.5M48.5 32H57.5" strokeWidth="5.5" />
          <rect x="15.5" y="17" width="33" height="30" rx="8.5" strokeWidth="5" />
        </g>
        {/* terminal leads + body */}
        <path d="M6.5 32H15.5M48.5 32H57.5" stroke="#fff" strokeWidth="4.5" />
        <rect x="15.5" y="17" width="33" height="30" rx="8.5" stroke="#fff" strokeWidth="4" fill="#fff" fillOpacity="0.1" />
        {/* S-shaped latch characteristic: HRS → fold, unstable (faded), fold → LRS */}
        <path d="M21.5 40C28.5 40 35.6 39.2 37.8 35.8" stroke="#fff" strokeWidth="3.6" />
        <path d="M37.8 35.8C40.3 31.6 23.8 32.4 26.2 28.2" stroke="#fff" strokeOpacity="0.5" strokeWidth="3" />
        <path d="M26.2 28.2C28.4 24.8 35 24 42.5 24" stroke="#fff" strokeWidth="3.6" />
      </g>
    </svg>
  );
}
