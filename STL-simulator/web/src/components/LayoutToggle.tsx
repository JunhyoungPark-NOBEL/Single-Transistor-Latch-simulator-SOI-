// [간단히 | 모두 보기] switch for the global results layout (state/layout.ts). A segmented radiogroup with
// arrow-key support; the choice persists (localStorage, or `?view=` when the URL carries it).
import { useRef, type KeyboardEvent } from "react";
import { useT } from "../i18n";
import { UX } from "../i18n/strings.ux";
import { useLayout, type LayoutMode } from "../state/layout";
import "./more.css";

const OPTIONS: { v: LayoutMode; label: keyof typeof UX; title: keyof typeof UX }[] = [
  { v: "simple", label: "layout.simple", title: "layout.simple.title" },
  { v: "all", label: "layout.all", title: "layout.all.title" },
];

export function LayoutToggle({ className }: { className?: string }) {
  const t = useT();
  const layout = useLayout((s) => s.layout);
  const setLayout = useLayout((s) => s.setLayout);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const i = OPTIONS.findIndex((o) => o.v === layout);
    const next = e.key === "Home" ? 0 : e.key === "End" ? OPTIONS.length - 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? (i - 1 + OPTIONS.length) % OPTIONS.length : (i + 1) % OPTIONS.length;
    setLayout(OPTIONS[next].v);
    refs.current[next]?.focus();
  };

  return (
    <div className={`seg layout-toggle${className ? ` ${className}` : ""}`} role="radiogroup" aria-label={t.l(UX["layout.label"])} data-testid="layout-toggle" onKeyDown={onKey}>
      {OPTIONS.map((o, i) => {
        const on = layout === o.v;
        return (
          <button
            key={o.v}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={on ? 0 : -1}
            title={t.l(UX[o.title])}
            data-testid={`layout-${o.v}`}
            onClick={() => setLayout(o.v)}
          >
            {t.l(UX[o.label])}
          </button>
        );
      })}
    </div>
  );
}
