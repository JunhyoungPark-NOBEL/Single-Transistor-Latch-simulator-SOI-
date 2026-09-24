// KaTeX rendering (cached). Inline/display; never throws — parse errors render with an error style.
import katex from "katex";
import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";

const cache = new Map<string, string>();

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);
}

export function renderTex(tex: string, display = false): string {
  const key = (display ? "D:" : "I:") + tex;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  let html: string;
  try {
    html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore", output: "htmlAndMathml", errorColor: "currentColor" });
    if (html.includes('class="katex-error"')) html = `<code class="tex-error" title="KaTeX parse error">${escapeHtml(tex)}</code>`;
  } catch {
    html = `<code class="tex-error" title="KaTeX error">${escapeHtml(tex)}</code>`;
  }
  if (cache.size > 2000) cache.clear();
  cache.set(key, html);
  return html;
}

export const Tex = memo(function Tex({ tex, display = false, className }: { tex: string; display?: boolean; className?: string }) {
  const html = useMemo(() => renderTex(tex, display), [tex, display]);
  return display ? (
    <div className={className} dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
  );
});

/** Minimum font scale used to fit a display equation before falling back to horizontal scrolling. */
export const MIN_FIT_SCALE = 0.8;

/**
 * Fit decision for a display equation of natural width `natural` (px at scale 1) in `avail` px:
 * scale down to at most MIN_FIT_SCALE; beyond that keep MIN_FIT_SCALE and scroll horizontally.
 */
export function fitScale(natural: number, avail: number, min = MIN_FIT_SCALE): { scale: number; scroll: boolean } {
  if (!(natural > 0) || !(avail > 0) || natural <= avail + 0.5) return { scale: 1, scroll: false };
  const s = Math.floor((avail / natural) * 1000) / 1000;
  return s >= min ? { scale: s, scroll: false } : { scale: min, scroll: true };
}

/**
 * Display equation that never gets clipped: it shrinks slightly (≥ 80 %) to fit its box, and when that is
 * not enough it scrolls horizontally inside its own box (visible scrollbar, edge fade, keyboard-focusable).
 * Re-measured when the box is resized (e.g. the Details window is resized) and when fonts finish loading.
 */
export const FitTex = memo(function FitTex({ tex, className, label }: { tex: string; className?: string; label?: string }) {
  const html = useMemo(() => renderTex(tex, true), [tex]);
  const ref = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState({ scale: 1, scroll: false });
  const cur = useRef(fit);
  cur.current = fit;
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    let alive = true;
    // Natural width (px at scale 1) is intrinsic to the formula: keep the largest estimate seen, so a fit
    // never oscillates between "fits" (glyph union) and "overflows" (scroll extent incl. overhangs).
    let natural = 0;
    const measure = () => {
      if (!alive) return;
      const avail = el.clientWidth - 2; // small safety margin for sub-pixel glyph overhang
      const scale = cur.current.scale;
      // KaTeX blocks are as wide as the box: measure the union of the glyph boxes, and the scroll extent
      // when it overflows. KaTeX scales linearly with font-size, so natural = measured / scale.
      const content = el.querySelector(".katex-html") ?? el.firstElementChild;
      if (content) {
        const r = document.createRange();
        r.selectNodeContents(content);
        natural = Math.max(natural, r.getBoundingClientRect().width / scale);
      }
      if (el.scrollWidth > el.clientWidth + 0.5) natural = Math.max(natural, el.scrollWidth / scale);
      const next = fitScale(natural, avail);
      // hysteresis: ignore sub-percent changes (rounding) so the measure → restyle loop settles
      if (next.scroll !== cur.current.scroll || Math.abs(next.scale - scale) > 0.002) setFit(next);
    };
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    // estimates taken with fallback fonts are discarded once the KaTeX fonts are in
    void document.fonts?.ready.then(() => {
      natural = 0;
      measure();
    });
    return () => {
      alive = false;
      ro?.disconnect();
    };
  }, [html]);
  return (
    <div
      ref={ref}
      className={`${className ?? ""} fit-tex${fit.scroll ? " scroll" : ""}`}
      style={fit.scale !== 1 ? { fontSize: `${fit.scale}em` } : undefined}
      tabIndex={fit.scroll ? 0 : undefined}
      role={fit.scroll ? "region" : undefined}
      aria-label={fit.scroll ? label : undefined}
      data-fit={fit.scroll ? "scroll" : fit.scale < 1 ? "scaled" : "fits"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});

/** Label text that may contain `_x` style subscripts from plain strings like "V_LU" → rendered as V_{LU}. */
export function symbolTex(s: string): string {
  return s.replace(/([A-Za-z])_([A-Za-z0-9]+)/g, (_, a: string, b: string) => `${a}_{\\mathrm{${b}}}`);
}
