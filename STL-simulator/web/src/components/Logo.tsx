import type { CSSProperties } from "react";

/** Biristor symbol supplied by the user: a slanted loop and opposed horizontal leads. */
function Mark({ branded = false }: { branded?: boolean }) {
  return <g fill="none" stroke="currentColor" strokeLinecap="square" strokeLinejoin="miter">
    <circle cx="32" cy="32" r="20" stroke={branded ? "var(--logo-a)" : undefined} fill={branded ? "var(--logo-soft)" : undefined} strokeWidth={branded ? "2" : "1.5"} />
    <path d="M32 18 H44 L32 46 H20 Z M32 18 H62 M2 46 H32" stroke={branded ? "var(--logo-b)" : undefined} strokeWidth="3.2" />
    {branded && <path d="M54 18 H62 M2 46 H10" stroke="var(--logo-c)" strokeWidth="3.8" />}
  </g>;
}
export function Logo({size=32,className,title}:{size?:number;className?:string;title?:string}) {
  return <svg className={`logo ${className??""}`} width={size} height={size} viewBox="0 0 64 64" role={title?"img":undefined} aria-label={title} aria-hidden={title?undefined:true} focusable="false"><Mark branded /></svg>;
}
export function BiristorGlyph({size=18,className,title,style}:{size?:number;className?:string;title?:string;letters?:boolean;style?:CSSProperties}) {
  return <svg className={className} style={style} width={size} height={size} viewBox="0 0 64 64" role={title?"img":undefined} aria-label={title} aria-hidden={title?undefined:true} focusable="false"><Mark /></svg>;
}
