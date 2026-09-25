// "한눈에" guide block — the first section of the Details window when it is opened from a parameter group
// or a field (physics/guideTarget.ts), and the row format of the Physics-tab parameter guide list.
// Each row: [symbol, label, (Details window only) the one-line definition + code index, intuitive picture
// (3-line clamp), ⚠ 주의 ▸] │ [키우면: the 3 effect lines].
// Two columns when the window (or the list) is ≥ 520 px wide (container queries in components/guide.css).
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { EffectLine, GuideText, verbLabel } from "../components/GuidePopover";
import { Tex } from "../components/Tex";
import "../components/guide.css";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { GUIDE_LEGEND, groupLead, guideFor, guideVerb } from "../params/guideUi";
import { GROUPS, LIGHT_FIELDS, type FieldDef } from "../params/schema";

/** Every parameter field by key (schema groups + the light block). */
export const FIELD_INDEX: Map<string, FieldDef> = new Map([...GROUPS.flatMap((g) => g.fields), ...Object.values(LIGHT_FIELDS)].map((f) => [f.key, f]));

/** Plain text of a key's row, lower case, for search (both languages). */
export function guideRowText(key: string): string {
  const g = guideFor(key);
  const f = FIELD_INDEX.get(key);
  const parts: string[] = [key];
  if (f) parts.push(f.label.ko, f.label.en, f.help.ko, f.help.en, f.code ?? "");
  if (g) {
    parts.push(g.intuitive.ko, g.intuitive.en, g.basis ?? "");
    for (const e of g.effect) parts.push(e.ko, e.en);
    if (g.caveat) parts.push(g.caveat.ko, g.caveat.en);
  }
  return parts.join(" \n ").toLowerCase();
}

/** Intuitive text: 3-line clamp with a "더 보기" toggle (also a click on the text) when it overflows. */
function Clamp({ text, open, onToggle }: { text: string; open: boolean; onToggle: () => void }) {
  const t = useT();
  const ref = useRef<HTMLParagraphElement>(null);
  const [long, setLong] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setLong(el.scrollHeight > el.clientHeight + 2);
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [text, open]);
  return (
    <>
      <p ref={ref} className={`grow-easy${open ? "" : " clamp"}`} onClick={long || open ? onToggle : undefined}>
        <GuideText text={text} />
      </p>
      {(long || open) && (
        <button type="button" className="link-btn grow-expand" aria-expanded={open} onClick={onToggle}>
          {t.l(GUIDE[open ? "guide.caveat.less" : "guide.caveat.more"])}
        </button>
      )}
    </>
  );
}

export function GuideRow({ fieldKey, testId, focused, heading = "h4", main, definition = false }: { fieldKey: string; testId: string; focused?: boolean; heading?: "h4" | "h5"; main?: boolean; definition?: boolean }) {
  const t = useT();
  const g = guideFor(fieldKey);
  const f = FIELD_INDEX.get(fieldKey);
  const [open, setOpen] = useState(!!focused);
  const [cavOpen, setCavOpen] = useState(!!focused);
  useEffect(() => {
    if (focused) {
      setOpen(true);
      setCavOpen(true);
    }
  }, [focused]);
  if (!g) return null;
  const H = heading;
  const label = f ? t.l(f.label) : fieldKey;
  return (
    <article className={`grow${focused ? " hl" : ""}`} data-testid={testId} id={testId} aria-label={label}>
      <div className="grow-left">
        <div className="grow-name">
          <H>
            {f?.sym && <Tex tex={f.sym} className="grow-sym" />}
            <span>{label}</span>
          </H>
          {main && <span className="badge">{t.l(GUIDE["guide.main"])}</span>}
        </div>
        {/* Details window: the one-line technical definition (f.help, e.g. the E_G formula of l_GIDL) and the code index */}
        {definition && f && (
          <p className="grow-def" data-testid={`${testId}-def`}>
            <GuideText text={t.l(f.help)} plain />
            {f.code && <code className="code-chip">{f.code}</code>}
          </p>
        )}
        <Clamp text={t.l(g.intuitive)} open={open} onToggle={() => setOpen(!open)} />
        {g.caveat && (
          <details className="grow-cav" open={cavOpen} onToggle={(e) => setCavOpen((e.currentTarget as HTMLDetailsElement).open)}>
            <summary>⚠ {t.l(GUIDE["guide.caveat"])}</summary>
            <p>
              <GuideText text={t.l(g.caveat)} plain />
            </p>
          </details>
        )}
      </div>
      <div className="grow-eff">
        <div className="gp-h">
          {verbLabel(t, guideVerb(f))}
          {guideVerb(f) === "raise" && <span aria-hidden> ↑</span>}
        </div>
        <ul>
          {g.effect.map((e, i) => (
            <li key={i} className={i === 2 ? "why" : undefined}>
              <EffectLine line={t.l(e)} />
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}

/** The Details window's first section: lead, one row per key, the arrows legend. */
export function GuideBlock({ keys, focus, group, nonce }: { keys: string[]; focus?: string; group?: string; nonce: number }) {
  const t = useT();
  const ref = useRef<HTMLElement>(null);
  const [hl, setHl] = useState<string | undefined>(focus);
  const rows = keys.filter((k) => !!guideFor(k));

  // focused row: expanded, highlighted for 1.5 s and scrolled into view inside the window body
  useEffect(() => {
    setHl(focus);
    if (!focus) return;
    const scroll = setTimeout(() => {
      const row = ref.current?.querySelector<HTMLElement>(`[data-testid="pw-guide-row-${CSS.escape(focus)}"]`);
      const body = ref.current?.closest<HTMLElement>(".pw-body");
      if (row && body) {
        const top = row.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop - 8;
        if (top > body.scrollTop + body.clientHeight - 80 || top < body.scrollTop) body.scrollTo({ top, behavior: "smooth" });
      }
    }, 80);
    const off = setTimeout(() => setHl(undefined), 1500);
    return () => {
      clearTimeout(scroll);
      clearTimeout(off);
    };
  }, [focus, nonce]);

  if (!rows.length) return null;
  return (
    <section ref={ref} className="pw-guide" data-testid="pw-guide" id="pw-sec-guide" data-section="-1" aria-labelledby="pw-guide-title">
      <h3 id="pw-guide-title">
        <GuideText text={t.l(GUIDE["guide.glance.title"])} />
      </h3>
      <p className="pwg-lead">
        <GuideText text={t.l(groupLead(group))} />
      </p>
      {rows.map((k) => (
        <GuideRow key={`${k}-${nonce}`} fieldKey={k} testId={`pw-guide-row-${k}`} focused={hl === k} definition />
      ))}
      <p className="pwg-foot">
        <GuideText text={t.l(GUIDE_LEGEND.arrows)} plain />
      </p>
    </section>
  );
}

