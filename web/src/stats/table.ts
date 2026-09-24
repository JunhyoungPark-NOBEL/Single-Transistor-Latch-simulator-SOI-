// Pure model of the statistics table (StatsTable.tsx): column/group definitions, per-row statistics and the
// model-vs-measured comparison, CSV export. Kept free of React so it can be unit-tested.
import type { ReactNode } from "react";
import { columnsToCsv } from "../utils/csv";
import { describe, isFiniteNum, ks2, type Describe, type Num } from "./describe";

export type StatsGroup = "centre" | "spread" | "tails" | "shape" | "sequence" | "counts" | "compare";
export type StatsColumn =
  | "mean" | "ci95" | "median"
  | "sd" | "iqr" | "cv"
  | "p05" | "p95" | "min" | "max"
  | "skew" | "kurt"
  | "lag1"
  | "n" | "censored"
  | "m_mean" | "m_sd" | "dmean" | "sd_ratio" | "ks_d" | "ks_p";

export const GROUP_COLUMNS: Record<StatsGroup, StatsColumn[]> = {
  centre: ["mean", "ci95", "median"],
  spread: ["sd", "iqr", "cv"],
  tails: ["p05", "p95", "min", "max"],
  shape: ["skew", "kurt"],
  sequence: ["lag1"],
  counts: ["n", "censored"],
  compare: ["m_mean", "m_sd", "dmean", "sd_ratio", "ks_d", "ks_p"],
};
/** The comparison sits next to centre/spread (Δmean, SD ratio relate to them); tails … counts follow. */
export const GROUP_ORDER: StatsGroup[] = ["centre", "spread", "compare", "tails", "shape", "sequence", "counts"];
export const COLUMN_GROUP = Object.fromEntries(
  GROUP_ORDER.flatMap((g) => GROUP_COLUMNS[g].map((c) => [c, g])),
) as Record<StatsColumn, StatsGroup>;
export const ALL_COLUMNS: StatsColumn[] = GROUP_ORDER.flatMap((g) => GROUP_COLUMNS[g]);

/** How a column is formatted: levels in the level unit (V), spreads in the spread unit (mV), plain numbers. */
export type CellKind = "level" | "spread" | "signed" | "pct" | "coef" | "ksd" | "p" | "ratio" | "int" | "cens";
export const COLUMN_KIND: Record<StatsColumn, CellKind> = {
  mean: "level", ci95: "spread", median: "level",
  sd: "spread", iqr: "spread", cv: "pct",
  p05: "level", p95: "level", min: "level", max: "level",
  skew: "coef", kurt: "coef", lag1: "coef",
  n: "int", censored: "cens",
  m_mean: "level", m_sd: "spread", dmean: "signed", sd_ratio: "ratio", ks_d: "ksd", ks_p: "p",
};

export interface StatsRow {
  key: string;
  /** row label (may contain markup, e.g. V<sub>LU</sub>) */
  label: ReactNode;
  /** plain-text label for CSV (defaults to `label` when it is a string, else `key`) */
  labelText?: string;
  /** small second line under the label */
  sub?: ReactNode;
  /** physical unit of `values` after `scale` (default "V") */
  unit?: string;
  /** one value per cycle/run; null/undefined/NaN = censored */
  values?: readonly Num[] | null;
  /** multiply values (and measured) by this before describing */
  scale?: number;
  /** total trials if `values` holds only the observed ones */
  n_total?: number;
  /** precomputed statistics (analytic distributions); fields given here override those computed from values */
  stats?: Partial<Describe> | null;
  /** measured sample for this row (overrides StatsTable's `measured[key]`) */
  measured?: readonly Num[] | null;
  color?: string;
  /** explanation shown on hover/focus of the row label */
  tip?: string;
  /** de-emphasised (e.g. an auxiliary analytic row) */
  secondary?: boolean;
}

export interface Comparison {
  dmean: number | null;
  sd_ratio: number | null;
  ks_d: number | null;
  ks_p: number | null;
}

export interface ComputedRow {
  spec: StatsRow;
  unit: string;
  d: Describe;
  m: Describe | null;
  cmp: Comparison | null;
}

export function emptyDescribe(): Describe {
  return {
    n: NaN, n_total: NaN, censored: NaN, mean: null, sd: null, se: null, ci95_half: null, cv: null, median: null,
    q1: null, q3: null, iqr: null, p05: null, p95: null, min: null, max: null, skewness: null, kurtosis_excess: null, lag1: null,
  };
}

const scaled = (v: readonly Num[] | null | undefined, s: number | undefined): Num[] | null =>
  v ? (s && s !== 1 ? v.map((x) => (isFiniteNum(x) ? x * s : null)) : [...v]) : null;

function withDerived(d: Describe): Describe {
  const o = { ...d };
  if (o.iqr === null && isFiniteNum(o.q1) && isFiniteNum(o.q3)) o.iqr = o.q3 - o.q1;
  if (o.cv === null && isFiniteNum(o.sd) && isFiniteNum(o.mean) && o.mean !== 0) o.cv = o.sd / Math.abs(o.mean);
  return o;
}

export function computeRow(spec: StatsRow, measuredFallback?: readonly Num[] | null): ComputedRow {
  const vals = scaled(spec.values, spec.scale);
  const base = vals ? describe(vals, { n_total: spec.n_total }) : emptyDescribe();
  const d = withDerived({ ...base, ...(spec.stats ?? {}) } as Describe);
  const measRaw = spec.measured !== undefined ? spec.measured : measuredFallback;
  const meas = scaled(measRaw, spec.scale);
  const m = meas && meas.some(isFiniteNum) ? describe(meas) : null;
  let cmp: Comparison | null = null;
  if (m) {
    const ks = vals ? ks2(vals, meas) : { D: null, p: null };
    cmp = {
      dmean: isFiniteNum(d.mean) && isFiniteNum(m.mean) ? d.mean - m.mean : null,
      sd_ratio: isFiniteNum(d.sd) && isFiniteNum(m.sd) && m.sd > 0 ? d.sd / m.sd : null,
      ks_d: ks.D,
      ks_p: ks.p,
    };
  }
  return { spec, unit: spec.unit ?? "V", d, m, cmp };
}

export function computeRows(rows: readonly StatsRow[], measured?: Record<string, readonly Num[] | null | undefined>): ComputedRow[] {
  return rows.map((r) => computeRow(r, measured?.[r.key]));
}

/** Raw (unformatted) value of a cell; `series` = the model row or its measured sub-row. */
export function cellValue(r: ComputedRow, col: StatsColumn, series: "model" | "measured" = "model"): number | null {
  const d = series === "model" ? r.d : r.m;
  if (!d) return null;
  if (COLUMN_GROUP[col] === "compare") {
    if (series === "measured" || !r.cmp) return null;
    if (col === "m_mean") return r.m?.mean ?? null;
    if (col === "m_sd") return r.m?.sd ?? null;
    return r.cmp[col as keyof Comparison];
  }
  switch (col) {
    case "ci95": return d.ci95_half;
    case "skew": return d.skewness;
    case "kurt": return d.kurtosis_excess;
    case "n": return Number.isFinite(d.n) ? d.n : null;
    case "censored": return Number.isFinite(d.censored) ? d.censored : null;
    default: return d[col as keyof Describe] as number | null;
  }
}

const labelText = (r: StatsRow) => r.labelText ?? (typeof r.label === "string" ? r.label : r.key);

/** CSV with every statistic in the row's physical unit (full precision), model and measured rows. */
export function statsCsv(rows: readonly ComputedRow[], names = { model: "model", measured: "measured" }): string {
  const out: Record<string, unknown[]> = {};
  const keys = [
    "quantity", "series", "unit", "n_total", "n", "censored", "mean", "ci95_half", "se", "median", "sd", "iqr", "q1", "q3",
    "cv", "p05", "p95", "min", "max", "skewness", "kurtosis_excess", "lag1", "dmean", "sd_ratio", "ks_D", "ks_p",
  ] as const;
  for (const k of keys) out[k] = [];
  const push = (r: ComputedRow, d: Describe, series: string, cmp: Comparison | null) => {
    out.quantity.push(labelText(r.spec));
    out.series.push(series);
    out.unit.push(r.unit);
    for (const k of ["n_total", "n", "censored"] as const) out[k].push(Number.isFinite(d[k]) ? d[k] : null);
    for (const k of ["mean", "ci95_half", "se", "median", "sd", "iqr", "q1", "q3", "cv", "p05", "p95", "min", "max", "skewness", "kurtosis_excess", "lag1"] as const) out[k].push(d[k]);
    out.dmean.push(cmp?.dmean ?? null);
    out.sd_ratio.push(cmp?.sd_ratio ?? null);
    out.ks_D.push(cmp?.ks_d ?? null);
    out.ks_p.push(cmp?.ks_p ?? null);
  };
  for (const r of rows) {
    push(r, r.d, r.spec.secondary && !r.spec.values ? `${names.model} (analytic)` : names.model, r.cmp);
    if (r.m) push(r, r.m, names.measured, null);
  }
  return columnsToCsv(keys.map((k) => ({ name: k, values: out[k] })));
}
