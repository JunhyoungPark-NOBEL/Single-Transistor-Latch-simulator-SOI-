import type { CSSProperties } from "react";

/** Stroke widths (viewBox units, 64 = full size) for a rendered size in px: at least ~1 px for the circle and
 *  ~1.4 px for the loop and leads, capped at the favicon's 4 / 5.5 (16 px) and floored at the large-size
 *  1.5 / 3.2 (≥ 40 px), so the mark neither vanishes in the shelf and schematic nor turns heavy in the About card. */
export function markStrokes(size: number): { circle: number; path: number } {
  const perPx = 64 / Math.max(size, 1);
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
  return { circle: +clamp(perPx * 1.0, 1.5, 4).toFixed(2), path: +clamp(perPx * 1.4, 3.2, 5.5).toFixed(2) };
}

/** Biristor symbol supplied by the user: a slanted loop and opposed horizontal leads. The lead ends are inset
 *  by half a stroke so the square caps stay inside the 64-unit box at every width. */
function Mark({ size }: { size: number }) {
  const w = markStrokes(size);
  const end = +(w.path / 2 + 0.3).toFixed(2);
  return <g fill="none" stroke="currentColor" strokeLinecap="square" strokeLinejoin="miter">
    <circle cx="32" cy="32" r="20" strokeWidth={w.circle} />
    <path d={`M32 18 H44 L32 46 H20 Z M32 18 H${64 - end} M${end} 46 H32`} strokeWidth={w.path} />
  </g>;
}
export function Logo({size=32,className,title}:{size?:number;className?:string;title?:string}) {
  return <svg className={`logo ${className??""}`} width={size} height={size} viewBox="0 0 64 64" role={title?"img":undefined} aria-label={title} aria-hidden={title?undefined:true} focusable="false"><Mark size={size} /></svg>;
}
export function BiristorGlyph({size=18,className,title,style}:{size?:number;className?:string;title?:string;letters?:boolean;style?:CSSProperties}) {
  return <svg className={className} style={style} width={size} height={size} viewBox="0 0 64 64" role={title?"img":undefined} aria-label={title} aria-hidden={title?undefined:true} focusable="false"><Mark size={size} /></svg>;
}
