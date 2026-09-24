// KaTeX rendering (cached). Inline/display; never throws — parse errors render with an error style.
import katex from "katex";
import { memo, useMemo } from "react";

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

/** Label text that may contain `_x` style subscripts from plain strings like "V_LU" → rendered as V_{LU}. */
export function symbolTex(s: string): string {
  return s.replace(/([A-Za-z])_([A-Za-z0-9]+)/g, (_, a: string, b: string) => `${a}_{\\mathrm{${b}}}`);
}
