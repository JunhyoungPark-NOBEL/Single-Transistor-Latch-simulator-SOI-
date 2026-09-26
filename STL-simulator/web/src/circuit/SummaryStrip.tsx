// Result summary of a circuit run (quick benches and schematic results): at most four cells chosen by a
// priority list in one card (hairline-separated, V_LU in --hrs and V_LD in --lrs), the other metrics behind
// "지표 n개 더 ▸" as a compact label | value table that stays in the DOM (a <details>). In the 모두 보기 layout
// every metric is a cell, as before.
import type { CSSProperties } from "react";
import type { SummaryItem } from "../api/types";
import { useT } from "../i18n";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { useIsAll } from "../state/layout";
import { isNum } from "../utils/format";
import { fmtCoefValue, isCoefKey, splitSpread, splitUnit } from "./summary";

/** Run metadata that never takes a summary cell (it is in the solver footnote or the run settings). */
const META = new Set(["runs", "steps_per_run", "t_noise_resolved_frac", "truncated_runs"]);

/**
 * The summary items to show first: `priority` entries in order (an entry matches a key exactly, or the part
 * after "<element>." — "n_latch_up" matches "X1.n_latch_up"; with `firstOf` only that element's keys match),
 * then the remaining items in server order, up to `max`. Returns [shown, rest].
 */
export function pickSummary(items: SummaryItem[], priority: string[], max = 4, firstOf?: string): [SummaryItem[], SummaryItem[]] {
  // a cell that repeats another within rounding (p-bit: latched fraction = P(1)) yields its place
  const num = (k: string) => {
    const v = items.find((s) => s.key === k)?.value;
    return typeof v === "number" ? v : null;
  };
  const p1 = num("P1");
  const pl = num("P_latched");
  const dup = new Set(p1 !== null && pl !== null && Math.abs(p1 - pl) < 0.005 ? ["P_latched"] : []);
  const pool = items.filter((s) => !META.has(s.key) && !dup.has(s.key));
  const shown: SummaryItem[] = [];
  const matches = (s: SummaryItem, p: string) => {
    if (s.key === p) return true;
    const dot = s.key.indexOf(".");
    if (dot < 0) return false;
    const el = s.key.slice(0, dot);
    const name = s.key.slice(dot + 1);
    if (p.startsWith("*.")) return name === p.slice(2); // any element (e.g. every comparator's p_fire)
    return name === p && (!firstOf || el === firstOf);
  };
  for (const p of priority) {
    if (shown.length >= max) break;
    const hit = pool.find((s) => !shown.includes(s) && matches(s, p));
    if (hit) shown.push(hit);
  }
  for (const s of pool) {
    if (shown.length >= max) break;
    if (!shown.includes(s)) shown.push(s);
  }
  const rest = items.filter((s) => !shown.includes(s));
  return [shown, rest];
}

/** Value colour of a summary quantity: latch-up voltages --hrs, latch-down voltages --lrs. */
export function summaryColor(key: string): string | undefined {
  const k = key.includes(".") ? key.slice(key.indexOf(".") + 1) : key;
  if (/^(V_LU|fold_V_LU|vd_first_lu|vd_lu_mean|lag_LU)/.test(k)) return "var(--hrs)";
  if (/^(V_LD|fold_V_LD|vd_ld_mean|lag_LD)/.test(k)) return "var(--lrs)";
  return undefined;
}

function display(s: SummaryItem) {
  const coef = isCoefKey(s.key) && typeof s.value === "number" && isNum(s.value);
  const v = coef ? { value: fmtCoefValue(s.value as number), unit: "" } : splitUnit(s.value, s.unit);
  // at most 2 significant digits, hidden when it rounds to 0 in the shown unit ("3.689 V", not "± 0.0 mV")
  const sp = isNum(s.spread) ? splitSpread(s.spread, s.unit) : null;
  return { v, sp };
}

/** Server labels: symbols as subscripts, "SD" as σ (the device tab's word for the spread). */
const labelText = (label: string) => subs(label.replace(/\bSD\b/g, "σ"));

function Cell({ s, prefix }: { s: SummaryItem; prefix: string }) {
  const t = useT();
  const { v, sp } = display(s);
  const color = summaryColor(s.key);
  const label = t.l(s.label);
  return (
    <div className="sum-cell" data-testid={`kpi-${prefix}${s.key}`} title={`${label} · ${s.key}`}>
      <div className="sum-label">
        <SubText text={labelText(label)} />
      </div>
      <div className="sum-value" style={color ? ({ color } as CSSProperties) : undefined}>
        <span data-testid={`kpi-${prefix}${s.key}-value`}>
          {v.value}
          {v.unit && <span className="u">{v.unit}</span>}
        </span>
        {sp && (
          <span className="sum-spread" title={t("c.spread.title")}>
            ± {sp.value} {sp.unit}
          </span>
        )}
      </div>
    </div>
  );
}

export interface SummaryStripProps {
  items: SummaryItem[];
  /** Keys in priority order (see pickSummary). */
  priority: string[];
  /** Container testid ("circuit-summary", "sch-summary"). */
  testId: string;
  /** Cell testid prefix: `kpi-<prefix><key>` and `kpi-<prefix><key>-value` ("c-", "sch-"). */
  prefix: string;
  /** Testid of the "지표 n개 더" toggle ("summary-more", "sch-summary-more"). */
  moreTestId: string;
  /** Restrict element-scoped priority keys to this element (the first STL, e.g. "X1"). */
  firstOf?: string;
  /** Results from an older parameter set: values are dimmed. */
  stale?: boolean;
  className?: string;
}

export function SummaryStrip({ items, priority, testId, prefix, moreTestId, firstOf, stale, className }: SummaryStripProps) {
  const t = useT();
  const all = useIsAll();
  if (!items.length) return null;
  const [shown, rest] = all ? [items, [] as SummaryItem[]] : pickSummary(items, priority, 4, firstOf);
  const n = shown.length;
  return (
    <section
      className={`sum-strip${all ? " all" : ""}${stale ? " stale" : ""}${className ? ` ${className}` : ""}`}
      data-testid={testId}
      aria-label={t("c.summary")}
      title={stale ? t("stale") : undefined}
      style={{ "--sum-n": all ? undefined : n } as CSSProperties}
    >
      <div className={`sum-cells n${Math.min(n, 4)}`}>
        {shown.map((s) => (
          <Cell key={s.key} s={s} prefix={prefix} />
        ))}
      </div>
      {rest.length > 0 && (
        <details className="sum-more">
          <summary data-testid={moreTestId}>{t("c.summary.more", { n: rest.length })}</summary>
          <dl className="sum-table">
            {rest.map((s) => {
              const { v, sp } = display(s);
              const color = summaryColor(s.key);
              return (
                <div key={s.key} className="sum-row" data-testid={`kpi-${prefix}${s.key}`} title={s.key}>
                  <dt>
                    <SubText text={labelText(t.l(s.label))} />
                  </dt>
                  <dd className="mono" style={color ? { color } : undefined}>
                    <span data-testid={`kpi-${prefix}${s.key}-value`}>
                      {v.value}
                      {v.unit ? ` ${v.unit}` : ""}
                    </span>
                    {sp ? ` ± ${sp.value} ${sp.unit}` : ""}
                  </dd>
                </div>
              );
            })}
          </dl>
        </details>
      )}
    </section>
  );
}
