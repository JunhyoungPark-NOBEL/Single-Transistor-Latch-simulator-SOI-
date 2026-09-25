// Compact descriptive-statistics table (docs/WEB_CONTRACT.md §9), shared by the device tab (stochastic MC)
// and the circuit editor (stochastic runs).
//
//   <StatsTable rows={[{ key: "V_LU", label: <>V<sub>LU</sub></>, unit: "V", values: res.V_LU }]}
//               measured={{ V_LU: res.measured?.V_LU }} csvName="stats" />
//
// Rows are quantities, columns are grouped (centre · spread · tails · shape · sequence · counts · vs measured).
// Levels print in the base unit (V), spreads with their own prefix (mV); every header explains itself on
// hover/focus. The table scrolls horizontally inside its card on narrow screens (sticky label column).
//
// Compact mode (the 간단히 layout's default): 6 columns — mean, SD, p5, p95, Δmeasured, KS p — in one header
// row; [모든 통계 열 (n)] (`<testId>-all-cols`) restores the grouped table. The CSV always has every column.
import { Fragment, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { IconCheck, IconCopy, IconDownload } from "../components/icons";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { useIsAll } from "../state/layout";
import { downloadText } from "../utils/csv";
import { fmtInt, isNum } from "../utils/format";
import type { Num } from "./describe";
import { DASH, fmtCoef, fmtLevel, fmtP, fmtPct, fmtShare, fmtSpread, levelDecimals, planUnits, type UnitPlan } from "./format";
import { HoverTip } from "./HoverTip";
import "./stats.css";
import {
  ALL_COLUMNS, cellValue, COLUMN_GROUP, COLUMN_KIND, COMPACT_COLUMNS, computeRows, statsCsv, visibleColumns,
  type ComputedRow, type StatsColumn, type StatsGroup, type StatsRow,
} from "./table";

export type { StatsColumn, StatsGroup, StatsRow } from "./table";
export { COMPACT_COLUMNS } from "./table";

export interface StatsTableProps {
  rows: StatsRow[];
  /** measured samples by row key (a row's own `measured` wins) */
  measured?: Record<string, readonly Num[] | null | undefined>;
  /** "rows" (default): measured statistics in a sub-row under each quantity; "columns": measured mean/SD as columns */
  measuredLayout?: "rows" | "columns";
  /** columns to show, in order (default: all; the comparison group appears only when a row has measured data) */
  columns?: StatsColumn[];
  csvName?: string;
  /** data-testid of the <table> (default "stats-table") */
  testId?: string;
  /** left side of the action bar (e.g. a short note) */
  caption?: ReactNode;
  /** show the copy/download buttons (default true) */
  actions?: boolean;
  /**
   * Low-priority columns (default min, max) that are hidden automatically when the table would not fit its card;
   * a toggle in the action bar shows them again (the table then scrolls). They are always in the CSV.
   */
  optional?: StatsColumn[];
  /**
   * Columns of the compact view. Default: COMPACT_COLUMNS in the 간단히 layout when `columns` is not given,
   * none in 모두 보기; `null` = no compact view.
   */
  compactColumns?: StatsColumn[] | null;
  /** Controlled "all columns" state (default: uncontrolled, on in 모두 보기, off in 간단히). */
  allColumns?: boolean;
  onAllColumnsChange?: (all: boolean) => void;
}

function rowPlan(rows: ComputedRow[]): UnitPlan {
  let lm = 0;
  let sm = 0;
  for (const r of rows) {
    for (const d of [r.d, r.m]) {
      if (!d) continue;
      for (const v of [d.mean, d.min, d.max]) if (isNum(v)) lm = Math.max(lm, Math.abs(v));
      if (isNum(d.sd)) sm = Math.max(sm, d.sd);
    }
  }
  return planUnits(rows[0]?.unit ?? "V", lm || null, sm || null);
}

const DEFAULT_OPTIONAL: StatsColumn[] = ["min", "max"];

export function StatsTable({
  rows, measured, measuredLayout = "rows", columns, csvName = "statistics", testId = "stats-table", caption, actions = true, optional = DEFAULT_OPTIONAL,
  compactColumns, allColumns, onAllColumnsChange,
}: StatsTableProps) {
  const t = useT();
  const isAll = useIsAll();
  const [copied, setCopied] = useState<"ok" | "fail" | null>(null);
  const [allUser, setAllUser] = useState<boolean | null>(null);
  const compactSet = compactColumns === undefined ? (columns || isAll ? null : COMPACT_COLUMNS) : compactColumns && compactColumns.length ? compactColumns : null;
  const showAll = !compactSet || (allColumns ?? allUser ?? isAll);
  const compact = !showAll;
  const setShowAll = (v: boolean) => {
    setAllUser(v);
    onAllColumnsChange?.(v);
  };
  const csvRows = useMemo(() => computeRows(rows, measured), [rows, measured]);
  // rows marked fullOnly (e.g. the analytic hazard row) appear with all columns only; the CSV keeps them
  const computed = useMemo(() => (compact ? csvRows.filter((r) => !r.spec.fullOnly) : csvRows), [compact, csvRows]);
  const anyMeas = computed.some((r) => r.m);
  const fullCols = useMemo(() => {
    let c = (columns ?? ALL_COLUMNS).filter((k) => anyMeas || COLUMN_GROUP[k] !== "compare");
    if (measuredLayout === "rows") c = c.filter((k) => k !== "m_mean" && k !== "m_sd");
    return c;
  }, [columns, anyMeas, measuredLayout]);
  const allCols = useMemo(() => visibleColumns(fullCols, compact ? compactSet : null, computed), [fullCols, compact, compactSet, computed]);
  // optional columns (full view): auto-hidden when the full table overflows its card, or as the user chose
  const optCols = useMemo(() => (compact ? [] : optional.filter((c) => allCols.includes(c))), [compact, optional, allCols]);
  const [userShow, setUserShow] = useState<boolean | null>(null);
  const [fits, setFits] = useState(true);
  const needW = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const tableRef = useRef<HTMLTableElement>(null);
  const showOpt = userShow ?? fits;
  const cols = useMemo(() => (showOpt ? allCols : allCols.filter((c) => !optCols.includes(c))), [showOpt, allCols, optCols]);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const tb = tableRef.current;
    if (!el || !tb || !optCols.length || userShow !== null) return;
    const check = () => {
      const cw = el.clientWidth;
      if (fits) {
        if (tb.scrollWidth > cw + 1) {
          needW.current = tb.scrollWidth;
          setFits(false);
        }
      } else if (needW.current && cw >= needW.current) setFits(true);
    };
    check();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(check);
    ro.observe(el);
    return () => ro.disconnect();
  }, [computed, optCols, userShow, fits]);
  const groups = useMemo(() => {
    const g: { g: StatsGroup; span: number; first: StatsColumn }[] = [];
    for (const c of cols) {
      const gg = COLUMN_GROUP[c];
      if (g.length && g[g.length - 1].g === gg) g[g.length - 1].span++;
      else g.push({ g: gg, span: 1, first: c });
    }
    return g;
  }, [cols]);
  const groupStart = new Set(groups.map((g) => g.first));

  // units: one plan for the whole table when every row has the same unit, else one per row
  const homogeneous = new Set(computed.map((r) => r.unit)).size <= 1;
  const tablePlan = useMemo(() => rowPlan(computed), [computed]);
  const plans = useMemo(() => computed.map((r) => (homogeneous ? tablePlan : rowPlan([r]))), [computed, homogeneous, tablePlan]);

  const unitOf = (c: StatsColumn, plan: UnitPlan): string => {
    const k = COLUMN_KIND[c];
    if (k === "level") return plan.levelUnit;
    if (k === "spread" || k === "signed") return plan.spreadUnit;
    if (k === "pct") return "%";
    return "";
  };

  const fmt = (r: ComputedRow, c: StatsColumn, series: "model" | "measured", plan: UnitPlan): ReactNode => {
    const v = cellValue(r, c, series);
    const d = series === "model" ? r.d : r.m;
    const sds = [r.d.sd, r.m?.sd].filter((x): x is number => isNum(x) && x > 0);
    const dec = levelDecimals(sds.length ? Math.min(...sds) : null, plan);
    switch (COLUMN_KIND[c]) {
      case "level": return fmtLevel(v, plan, dec);
      case "spread": return isNum(v) && c === "ci95" ? `±${fmtSpread(v, plan)}` : fmtSpread(v, plan);
      case "signed": return fmtSpread(v, plan, true);
      case "pct": return fmtPct(v);
      case "coef": return fmtCoef(v, 2);
      case "ksd": return fmtCoef(v, 3);
      case "p": return fmtP(v);
      case "ratio": return fmtCoef(v, 2);
      case "int": return isNum(v) ? fmtInt(v) : DASH;
      case "cens": {
        if (!isNum(v)) return DASH;
        const tot = d?.n_total;
        return (
          <>
            {fmtInt(v)}
            {v > 0 && isNum(tot) && tot > 0 && <span className="stats-frac">{fmtShare(v / tot)}</span>}
          </>
        );
      }
    }
  };

  const csv = () => statsCsv(csvRows, { model: t("stats.row.model"), measured: t("stats.row.measured") });
  const copy = async () => {
    const text = csv();
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        ta.remove();
      } catch {
        ok = false;
      }
    }
    setCopied(ok ? "ok" : "fail");
    window.setTimeout(() => setCopied(null), 1600);
  };

  const unitHint = homogeneous && computed.length > 0 && tablePlan.levelUnit
    ? t("stats.units", { level: tablePlan.levelUnit, spread: tablePlan.spreadUnit })
    : null;

  const colTitle = (c: StatsColumn) => {
    if (compact) {
      const k = `stats.cc.${c}` as StrKey;
      const v = t(k);
      if (v !== k) return v;
    }
    return t(`stats.c.${c}` as StrKey);
  };
  const colTip = (c: StatsColumn) => t(`stats.tip.${c}` as StrKey);

  return (
    <div className="stats-block">
      {(actions || caption || unitHint) && (
        <div className="stats-bar">
          <div className="stats-bar-note small muted">
            {caption}
            {caption && unitHint ? " · " : null}
            {unitHint}
          </div>
          {compactSet && (
            <button type="button" className="btn sm ghost stats-allcols" aria-pressed={showAll} onClick={() => setShowAll(!showAll)} data-testid={`${testId}-all-cols`}>
              {showAll ? t("stats.cols.fewer") : t("stats.cols.all", { n: fullCols.length })}
            </button>
          )}
          {optCols.length > 0 && (!fits || userShow !== null) && (
            <button type="button" className="btn sm ghost stats-opt" aria-pressed={showOpt} onClick={() => setUserShow(!showOpt)} data-testid={`${testId}-optional`}>
              {t(showOpt ? "stats.opt.hide" : "stats.opt.show", { cols: optCols.map((c) => t(`stats.c.${c}` as StrKey)).join("·") })}
            </button>
          )}
          {actions && (
            <div className="stats-actions">
              <button type="button" className="btn sm ghost" onClick={() => void copy()} data-testid={`${testId}-copy`} aria-live="polite">
                {copied === "ok" ? <IconCheck size={13} /> : <IconCopy size={13} />}
                {copied === "ok" ? t("stats.copied") : copied === "fail" ? t("stats.copy.fail") : t("stats.copy")}
              </button>
              <button type="button" className="btn sm ghost" onClick={() => downloadText(`${csvName}.csv`, csv())} data-testid={`${testId}-download`}>
                <IconDownload size={13} /> {t("stats.download")}
              </button>
            </div>
          )}
        </div>
      )}
      <div className={`stats-scroll${compact ? " compact" : ""}`} ref={scrollRef} tabIndex={0} role="region" aria-label={t("stats.aria.table")}>
        <table className={`stats-table${compact ? " compact" : ""}`} ref={tableRef} data-testid={testId}>
          <thead>
            {compact ? null : (
              <tr className="stats-groups">
                <th rowSpan={2} className="stats-label-col" scope="col">
                  {t("stats.quantity")}
                </th>
                {groups.map((g) => (
                  <th key={g.g} colSpan={g.span} scope="colgroup" className="g-start">
                    {t(`stats.g.${g.g}` as StrKey)}
                  </th>
                ))}
              </tr>
            )}
            <tr>
              {compact && (
                <th className="stats-label-col" scope="col">
                  {t("stats.quantity")}
                </th>
              )}
              {cols.map((c) => (
                <th key={c} scope="col" className={`num${groupStart.has(c) && !compact ? " g-start" : ""}`} data-col={c}>
                  <HoverTip tip={colTip(c)}>{colTitle(c)}</HoverTip>
                  {homogeneous && <span className="stats-unit">{unitOf(c, tablePlan) || " "}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {computed.map((r, i) => {
              const plan = plans[i];
              const showMeasRow = measuredLayout === "rows" && !!r.m;
              const unitChip = !homogeneous ? (plan.levelUnit === plan.spreadUnit ? plan.levelUnit : `${plan.levelUnit} · ${plan.spreadUnit}`) : null;
              return (
                <Fragment key={r.spec.key}>
                  <tr className={`${r.spec.secondary ? "secondary" : ""}${showMeasRow ? " has-meas" : ""}`} data-row={r.spec.key}>
                    <th scope="row" className="stats-label-col">
                      <div className="stats-label">
                        {r.spec.color && <span className="stats-swatch" style={{ background: r.spec.color }} aria-hidden />}
                        <HoverTip tip={r.spec.tip}>
                          <span className="stats-name">{r.spec.label}</span>
                        </HoverTip>
                      </div>
                      {(r.spec.sub || unitChip) && (
                        <div className="stats-sub">
                          {r.spec.sub}
                          {r.spec.sub && unitChip ? " · " : null}
                          {unitChip && <span className="mono">{unitChip}</span>}
                        </div>
                      )}
                    </th>
                    {cols.map((c) => (
                      <td key={c} className={`num${groupStart.has(c) && !compact ? " g-start" : ""}${COLUMN_GROUP[c] === "compare" ? " cmp" : ""}`} data-col={c} title={r.spec.cellTips?.[c]}>
                        {fmt(r, c, "model", plan)}
                      </td>
                    ))}
                  </tr>
                  {showMeasRow && (
                    <tr className="meas" data-row={`${r.spec.key}:measured`}>
                      <th scope="row" className="stats-label-col">
                        <div className="stats-label">
                          <span className="stats-swatch hollow" aria-hidden />
                          <span className="stats-series">{t("stats.row.measured")}</span>
                        </div>
                      </th>
                      {cols.map((c) => (
                        <td key={c} className={`num${groupStart.has(c) && !compact ? " g-start" : ""}${COLUMN_GROUP[c] === "compare" ? " cmp" : ""}`} data-col={c}>
                          {COLUMN_GROUP[c] === "compare" ? "" : fmt(r, c, "measured", plan)}
                        </td>
                      ))}
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
