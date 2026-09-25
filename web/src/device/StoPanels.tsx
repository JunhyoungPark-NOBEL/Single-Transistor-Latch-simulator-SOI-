// Stochastic device panels: statistics summary (one line between the answer bar and the plots, table on
// demand), (a) I–V + MC traces (hero), (b) V_LU/V_LD histogram ⇄ CDF, (c) hazard and survival, (d) V_G curve
// mean ± σ (explicit Compute), (e) cycle series, (f) design map.
import type { Data, Layout, Shape } from "plotly.js";
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { finite } from "../api/guards";
import type { Arr, BranchesResult, Cdf, HazardResult, SweepMCResult, VgCurveStochasticResult } from "../api/types";
import { DetailsButton } from "../components/DetailsButton";
import { IconChevron } from "../components/icons";
import { Panel, type PanelMenuItem } from "../components/Panel";
import { useT, type T } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { DEV } from "../i18n/strings.device";
import { fill, UX } from "../i18n/strings.ux";
import { SubText } from "../plots/SubText";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { useIsAll } from "../state/layout";
import { loadDesignMap, loadMeasured, runVgStochastic } from "../state/runner";
import { useStore } from "../state/store";
import { describe, diffSeries, ecdf, ks2, meanCI } from "../stats/describe";
import { fmtP, fmtShare } from "../stats/format";
import { StatsTable, type StatsRow } from "../stats/StatsTable";
import { fmtDuration, isNum } from "../utils/format";
import { powerMW } from "../utils/payload";
import { isPaperReference, isStale, logRange, nums, pos, useDeviceKeys, useEntry, usePalette } from "./common";
import { measuredIvTraces, RangePopover, ROW_LEGEND, Seg } from "./DetPanels";

/** Readable engine name (never the raw id such as calibrated_lookup). */
export function engineName(t: T, engine: string | undefined): string {
  const k = `engine.${engine}` as keyof typeof DEV;
  return engine && DEV[k] ? t.l(DEV[k]) : (engine ?? "");
}

// ---------------------------------------------------------------- statistics summary
/** Linear interpolation of a quantile function (prob ascending) at probability p. */
function quantileAt(prob: Arr, v: Arr, p: number): number | null {
  let prev: [number, number] | null = null;
  for (let i = 0; i < prob.length; i++) {
    const a = prob[i];
    const b = v[i];
    if (!isNum(a) || !isNum(b)) continue;
    if (a >= p) {
      if (!prev) return b;
      const [pa, pb] = prev;
      return a === pa ? b : pb + ((b - pb) * (p - pa)) / (a - pa);
    }
    prev = [a, b];
  }
  return prev ? prev[1] : null;
}

/** "Measured record: 400 cycles at V_G = −1.8 V, 1.15 mW" (built from the run, not the server's label). */
function measuredCaption(t: T, n: number, vg: number, powerMw: number): string {
  const light = powerMw > 0 ? `${powerMw.toFixed(2)} mW` : t("stats.meas.dark");
  return t("stats.meas.caption", { n, vg: vg.toFixed(1).replace("-", "−"), light });
}

/**
 * Statistics of the stochastic run: one muted summary line (cycles · no latch-up · KS vs measured · seed ·
 * sweep · engine · runtime) with [표 보기]; the table opens compact (mean, σ, p5, p95, Δ측정, KS p) and
 * [모든 통계 열] restores the grouped table with the hazard row, the CI and the censoring line. In the 모두 보기
 * layout it starts expanded with every column.
 */
export function StatsPanel() {
  const t = useT();
  const c = usePalette();
  const isAll = useIsAll();
  const params = useStore((s) => s.params);
  const { entry, data } = useEntry<SweepMCResult>("sweep_mc");
  const { entry: he, data: hz } = useEntry<HazardResult>("hazard");
  const keys = useDeviceKeys();
  const [open, setOpen] = useState<boolean | null>(null);
  const [allCols, setAllCols] = useState<boolean | null>(null);
  const expanded = open ?? isAll;
  const full = allCols ?? isAll;
  const bodyId = useId();
  // show the analytic (hazard) row only when it belongs to the same parameter set as the MC result
  const hazardOk = !!hz && hz.fold_V !== null && isStale(he, keys.hazard) === isStale(entry, keys.sweep_mc);
  const lu = useMemo(() => (data ? describe(data.V_LU) : null), [data]);
  const ci = lu ? meanCI(lu) : null;
  const ciText = ci ? t("stats.meta.ci", { lo: ci.lo.toFixed(4), hi: ci.hi.toFixed(4) }) : undefined;
  const rows = useMemo<StatsRow[]>(() => {
    if (!data) return [];
    const m = data.measured;
    const out: StatsRow[] = [
      { key: "V_LU", label: <>V<sub>LU</sub></>, labelText: "V_LU", sub: t("stats.row.vlu"), unit: "V", values: data.V_LU, measured: m?.V_LU ?? null, color: c.hrs, cellTips: ciText ? { mean: ciText } : undefined },
      { key: "V_LD", label: <>V<sub>LD</sub></>, labelText: "V_LD", sub: t("stats.row.vld"), unit: "V", values: data.V_LD, measured: m?.V_LD ?? null, color: c.lrs },
      {
        key: "window", label: <>V<sub>LU</sub> − V<sub>LD</sub></>, labelText: "V_LU - V_LD", sub: t("stats.row.window"), unit: "V",
        values: diffSeries(data.V_LU, data.V_LD), measured: m?.V_LD ? diffSeries(m.V_LU, m.V_LD) : null, color: c.text2, tip: t("stats.row.window.tip"),
      },
    ];
    if (hazardOk && hz) {
      const st = hz.stats;
      const q = hz.quantiles;
      const atom = isNum(hz.fold_atom) && hz.fold_atom > 0.005 ? ` · ${t("stats.row.hazard.atom", { pct: fmtShare(hz.fold_atom) })}` : "";
      out.push({
        key: "hazard", label: <>V<sub>LU</sub></>, labelText: "V_LU (carrier noise only, hazard)", sub: `${t("stats.row.hazard")}${atom}`, unit: "V",
        secondary: true, fullOnly: true, color: c.unstable, tip: t("stats.row.hazard.tip"),
        stats: {
          mean: st.mean, sd: st.sd, median: st.median, p05: st.p05, p95: st.p95, min: st.min, max: st.max,
          q1: quantileAt(q.prob, q.v, 0.25), q3: quantileAt(q.prob, q.v, 0.75),
        },
      });
    }
    return out;
  }, [data, hz, hazardOk, c, t, ciText]);

  const m = data?.measured;
  const seed = data?.seed ?? params.stochastic.seed;
  const nTot = lu?.n_total ?? 0;
  const ks = useMemo(() => {
    if (!data?.measured) return null;
    const a = ks2(data.V_LU, data.measured.V_LU);
    const b = data.measured.V_LD ? ks2(data.V_LD, data.measured.V_LD) : null;
    if (a.p === null && (!b || b.p === null)) return null;
    return [a.p, b?.p ?? null].filter((p, i) => i === 0 || p !== null).map((p) => fmtP(p)).join(" / ");
  }, [data]);
  const running = entry?.status === "running" || entry?.status === "queued";
  const stale = isStale(entry, keys.sweep_mc);
  const err = entry?.status === "error" ? entry.error : null;
  const cens = lu?.censored ?? 0;
  const parts: { key: string; text: string; cls?: string; title?: string }[] = data && lu
    ? [
        { key: "n", text: t("stats.line.cycles", { n: nTot }) },
        { key: "cens", text: t("stats.line.noLatch", { n: cens, pct: fmtShare(nTot ? cens / nTot : 0) }), cls: cens > 0 ? "warn" : undefined, title: t("stats.line.noLatch.tip") },
        ...(ks ? [{ key: "ks", text: t("stats.line.ks", { p: ks }) }] : []),
        { key: "seed", text: t("stats.line.seed", { s: seed }) },
        { key: "sweep", text: t("stats.meta.sweep", { v: data.vd_max_V ?? params.sweep.vd_max_V, rate: data.rate_V_per_s ?? params.sweep.rate_V_per_s }) },
        { key: "engine", text: engineName(t, data.engine) },
        { key: "rt", text: `${fmtDuration(data.runtime_s)}${entry?.cached ? ` · ${t("cached")}` : ""}` },
      ]
    : [];
  const sentence = parts.map((x) => x.text).join(" · ");
  return (
    <section className={`stats-card${expanded && data ? " open" : ""}`} aria-labelledby="panel-stats-title" data-testid="panel-stats" aria-busy={running}>
      <div className="stats-line">
        <h3 className="stats-line-title" id="panel-stats-title">
          {t("stats.line.title")}
        </h3>
        {stale ? (
          <span className="badge stale" title={t("stale")}>
            {t.l(UX["badge.changed"])}
          </span>
        ) : err ? (
          <span className="badge err">{t.l(UX["status.error"])}</span>
        ) : null}
        <p className={`stats-meta${running ? " dim" : ""}`} data-testid="stats-meta" title={sentence || undefined}>
          {running && !data ? (
            <span>{t("stats.line.running")}</span>
          ) : parts.length ? (
            parts.map((x, i) => (
              <span key={x.key} className={x.cls} title={x.title} data-testid={x.key === "cens" ? "stats-line-censored" : undefined}>
                {i > 0 && <span className="sep"> · </span>}
                {x.text}
              </span>
            ))
          ) : (
            <span className="muted">{t("stats.line.empty")}</span>
          )}
        </p>
        <div className="stats-line-actions">
          {data && (
            <button type="button" className="btn sm ghost stats-expand" aria-expanded={expanded} aria-controls={bodyId} onClick={() => setOpen(!expanded)} data-testid="stats-expand">
              {expanded ? t("stats.collapse") : t("stats.expand")}
              <IconChevron size={13} className={`chev${expanded ? " up" : ""}`} />
            </button>
          )}
          <DetailsButton topic="sweep-mc" testId="details-panel-stats" />
        </div>
      </div>
      {err && (
        <div className="panel-foot">
          <div className="err-box" role="alert">
            <strong>{t("error")}:</strong> {err}
          </div>
        </div>
      )}
      {expanded && data && lu && (
        <div className={`stats-panel stats-body${running ? " dim" : ""}`} id={bodyId}>
          <StatsTable rows={rows} csvName="statistics_vlu_vld" allColumns={full} onAllColumnsChange={setAllCols} />
          {full && (
            <div className="stats-detail small muted">
              {ci && <span data-testid="stats-ci">{ciText}</span>}
              <span className={cens > 0 ? "warn" : undefined} data-testid="stats-censored">
                {t("stats.meta.censored", { n: cens, pct: fmtShare(nTot ? cens / nTot : 0) })}
              </span>
            </div>
          )}
          <div className="stats-foot small muted">
            {t("stats.foot.censoring")}
            {m && m.V_LU.length > 0 && <> {measuredCaption(t, m.V_LU.length, params.device.vg, powerMW(params.device))}</>}
          </div>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------- (a) I–V + MC sweeps
export function McIvPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const preset = useStore((s) => s.preset);
  const measured = useStore((s) => s.measured);
  const { entry, data } = useEntry<SweepMCResult>("sweep_mc");
  const { data: br } = useEntry<BranchesResult>("branches");
  const { sweep_mc: key } = useDeviceKeys();
  const [log, setLog] = useState(true);
  const [showMeas, setShowMeas] = useState(true);
  const [showBand, setShowBand] = useState(false);
  const [showTraces, setShowTraces] = useState(true);
  const measKind = isPaperReference(params.device) ? "paper" : preset === "photo" ? "photo" : null;
  useEffect(() => {
    if (showMeas && measKind && measured.status === "idle") void loadMeasured();
  }, [showMeas, measKind, measured.status]);
  const L = (k: keyof typeof DEV) => t.l(DEV[k]);

  const plot = useMemo(() => {
    if (!data) return undefined;
    const traces: Data[] = [];
    if (showMeas) traces.push(...measuredIvTraces(t, c, measured.data, measKind, powerMW(params.device), { band: showBand && measKind === "paper", name: L("leg.meas"), bandName: L("leg.measBand") }));
    // all MC traces in one trace separated by nulls (fast, one legend entry)
    const x: (number | null)[] = [];
    const y: (number | null)[] = [];
    if (showTraces)
      for (const tr of data.traces) {
        for (const part of [tr.up, tr.down]) {
          x.push(...nums(part.vd), null);
          y.push(...(log ? pos(part.id) : nums(part.id)), null);
        }
      }
    if (x.length) traces.push({ x, y, type: "scatter", mode: "lines", name: fill(L("leg.traces"), { n: data.traces.length }), line: { color: c.sto, width: 1 }, opacity: 0.38, hoverinfo: "skip", connectgaps: false });
    const hrs = data.centre.HRS.vd.length ? data.centre.HRS : br ? { vd: br.HRS.vd, id: br.HRS.id } : null;
    const lrs = data.centre.LRS.vd.length ? data.centre.LRS : br ? { vd: br.LRS.vd, id: br.LRS.id } : null;
    if (hrs) traces.push({ x: nums(hrs.vd), y: log ? pos(hrs.id) : nums(hrs.id), type: "scatter", mode: "lines", name: L("leg.hrs"), line: { color: c.hrs, width: 2.2 }, hovertemplate: `${HOVER_IV}<extra>HRS</extra>` });
    if (lrs) traces.push({ x: nums(lrs.vd), y: log ? pos(lrs.id) : nums(lrs.id), type: "scatter", mode: "lines", name: L("leg.lrs"), line: { color: c.lrs, width: 2.2 }, hovertemplate: `${HOVER_IV}<extra>LRS</extra>` });
    if (br?.latch) traces.push({ x: nums(br.unstable.vd), y: log ? pos(br.unstable.id) : nums(br.unstable.id), type: "scatter", mode: "lines", name: L("leg.unstable"), showlegend: false, line: { color: c.unstable, width: 1.2, dash: "dash" }, hoverinfo: "skip" });
    // per-cycle rug in a bottom strip (V_LU blue, V_LD red)
    const lu = finite(data.V_LU);
    const ld = finite(data.V_LD);
    traces.push(
      { x: lu, y: lu.map(() => 1), yaxis: "y2", type: "scatter", mode: "markers", name: t("axis.leg.rugLu"), showlegend: false, marker: { symbol: "line-ns-open", size: 14, color: c.hrs, line: { width: 1.2 } }, opacity: 0.6, hovertemplate: "V<sub>LU</sub> = %{x:.3f} V<extra></extra>" },
      { x: ld, y: ld.map(() => 0), yaxis: "y2", type: "scatter", mode: "markers", name: t("axis.leg.rugLd"), showlegend: false, marker: { symbol: "line-ns-open", size: 14, color: c.lrs, line: { width: 1.2 } }, opacity: 0.6, hovertemplate: "V<sub>LD</sub> = %{x:.3f} V<extra></extra>" },
    );
    const shapes: Partial<Shape>[] = [];
    for (const [v, col] of [[data.stats.LU.mean, c.hrs], [data.stats.LD.mean, c.lrs]] as const)
      if (isNum(v)) shapes.push({ type: "line", xref: "x", yref: "paper", x0: v, x1: v, y0: 0, y1: 1, line: { color: col, width: 1, dash: "dot" } });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vd") }, range: [0, params.sweep.vd_max_V + 0.15], anchor: "y2" },
      yaxis: { ...currentAxis(log, log ? t("axis.idAbs") : t("axis.id")), domain: [0.16, 1], ...(log ? { range: logRange(traces.filter((tr) => (tr as { yaxis?: string }).yaxis !== "y2").map((tr) => (tr as { y?: (number | null)[] }).y)) } : {}) },
      yaxis2: { domain: [0, 0.1], range: [-0.8, 1.8], showticklabels: false, showgrid: false, zeroline: false, ticks: "", showline: false, fixedrange: true },
      shapes,
      margin: { l: 64, r: 16, t: 30, b: 46 },
      legend: ROW_LEGEND,
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, br, log, showMeas, showBand, showTraces, measured.data, measKind, params.sweep.vd_max_V, params.device, c, t]);

  const menu: PanelMenuItem[] = [
    { kind: "check", id: "traces", label: L("menu.traces"), checked: showTraces, onChange: setShowTraces, testId: "mciv-traces" },
    ...(measKind ? [{ kind: "check" as const, id: "meas", label: L("menu.meas"), checked: showMeas, onChange: setShowMeas, testId: "mciv-meas" }] : []),
    ...(measKind === "paper" ? [{ kind: "check" as const, id: "band", label: L("menu.band"), checked: showBand, onChange: setShowBand, disabled: !showMeas }] : []),
  ];
  return (
    <Panel
      id="mc-iv"
      primary
      title={t("p.mciv")}
      desc={t("p.mciv.desc")}
      topic="sweep-mc"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="mc_iv"
      plot={plot}
      warnings={data?.warnings}
      toolbar={data ? <Seg label={t.l(DEV["menu.y"])} value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ v: "log", label: t("log") }, { v: "lin", label: t("lin") }]} /> : undefined}
      menu={data ? menu : undefined}
      foot={data ? fill(L("foot.engine"), { engine: engineName(t, data.engine), n: data.traces.length }) : undefined}
    />
  );
}

// ---------------------------------------------------------------- (b) distributions
function histFromEdges(values: number[], edges: number[]): number[] {
  const counts = new Array(Math.max(0, edges.length - 1)).fill(0);
  for (const v of values) {
    if (v < edges[0] || v > edges[edges.length - 1]) continue;
    let lo = 0;
    let hi = edges.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (v >= edges[mid]) lo = mid;
      else hi = mid;
    }
    counts[lo]++;
  }
  return counts;
}
/** Equal-width edges covering both samples (so model and measured share bins). */
function commonEdges(a: number[], b: number[], base: number[]): number[] {
  const all = [...a, ...b];
  if (!all.length) return base;
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const w = base.length > 1 ? base[1] - base[0] : (hi - lo) / 30 || 0.01;
  const n = Math.min(200, Math.max(5, Math.ceil((hi - lo) / w) + 1));
  return Array.from({ length: n + 1 }, (_, i) => lo - w / 2 + i * w);
}
/**
 * Model CDF normalised to all cycles: the server's `cdf` block when it is well formed (it also handles latch-downs
 * not reached by 0 V), else the ECDF of the values over every cycle (censored cycles keep it below 1).
 */
function modelCdf(block: Cdf | undefined, values: Arr): { v: number[]; p: number[] } {
  if (block && Array.isArray(block.v) && Array.isArray(block.p) && block.v.length === block.p.length && block.v.length > 0) return { v: block.v, p: block.p };
  const e = ecdf(values);
  return { v: e.v, p: e.p };
}

export function DistPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const { entry, data } = useEntry<SweepMCResult>("sweep_mc");
  const { sweep_mc: key } = useDeviceKeys();
  const [view, setView] = useState<"hist" | "cdf">("hist");
  const [showMeas, setShowMeas] = useState(true);
  const [which, setWhich] = useState<"both" | "LU" | "LD">("both");
  const plot = useMemo(() => {
    if (!data) return undefined;
    const traces: Data[] = [];
    const meas = showMeas ? data.measured : null;
    const series = [
      { k: "LU" as const, color: c.hrs, label: "V<sub>LU</sub>", raw: data.V_LU, model: finite(data.V_LU), meas: finite(meas?.V_LU), fold: data.centre.V_LU },
      { k: "LD" as const, color: c.lrs, label: "V<sub>LD</sub>", raw: data.V_LD, model: finite(data.V_LD), meas: finite(meas?.V_LD), fold: data.centre.V_LD },
    ].filter((s) => which === "both" || which === s.k);
    const shapes: Partial<Shape>[] = [];
    for (const s of series) {
      // the deterministic fold as a dotted line (the MC distribution scatters around it)
      if (isNum(s.fold)) shapes.push({ type: "line", xref: "x", yref: "paper", x0: s.fold, x1: s.fold, y0: 0, y1: 1, line: { color: s.color, width: 1.2, dash: "dot" } });
      if (view === "hist") {
        const edges = commonEdges(s.model, s.meas, data.hist[s.k].edges);
        const counts = histFromEdges(s.model, edges);
        const centers = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);
        const widths = counts.map((_, i) => edges[i + 1] - edges[i]);
        const nm = t("axis.leg.model", { s: s.label });
        traces.push({ x: centers, y: counts, width: widths, type: "bar", name: nm, marker: { color: s.color, line: { width: 0 } }, opacity: 0.8, hovertemplate: `${s.label} = %{x:.3f} V<br>n = %{y}<extra>${nm}</extra>` });
        if (s.meas.length) {
          const mc = histFromEdges(s.meas, edges);
          const scale = s.model.length / s.meas.length;
          const mn = t("axis.leg.meas", { s: s.label });
          traces.push({ x: centers, y: mc.map((v) => v * scale), type: "scatter", mode: "lines", line: { shape: "hvh", color: c.meas, width: 1.6 }, name: `${mn}${Math.abs(scale - 1) > 1e-9 ? ` (×${scale.toFixed(2)})` : ""}`, hovertemplate: `${s.label} = %{x:.3f} V<br>n = %{y:.1f}<extra>${mn}</extra>` });
        }
      } else {
        const cm = modelCdf(data.cdf?.[s.k], s.raw);
        const nm = t("axis.leg.model", { s: s.label });
        traces.push({ x: cm.v, y: cm.p, type: "scatter", mode: "lines", line: { shape: "hv", color: s.color, width: 2 }, name: nm, hovertemplate: `${s.label} = %{x:.3f} V<br>P = %{y:.3f}<extra>${nm}</extra>` });
        if (s.meas.length) {
          const mm = ecdf(s.meas);
          const mn = t("axis.leg.meas", { s: s.label });
          traces.push({ x: mm.v, y: mm.p, type: "scatter", mode: "lines", line: { shape: "hv", color: c.meas, width: 1.6, dash: "dot" }, name: mn, hovertemplate: `${s.label} = %{x:.3f} V<br>P = %{y:.3f}<extra>${mn}</extra>` });
        }
      }
    }
    const layout: Partial<Layout> = {
      barmode: "overlay",
      bargap: 0,
      xaxis: { title: { text: which === "LU" ? t("axis.vlu") : which === "LD" ? t("axis.vld") : t("axis.vSwitch") } },
      yaxis: view === "hist" ? { title: { text: t("axis.count") }, rangemode: "tozero" } : { title: { text: t("axis.cdf") }, range: [0, 1.02] },
      shapes,
      margin: { l: 56, r: 16, t: 40, b: 46 },
    };
    return { data: traces, layout };
  }, [data, view, showMeas, which, c, t]);
  const m = data?.measured;
  // censored cycles: the (all-cycle) CDF plateaus at the fraction that switched within the sweep
  const plateau = useMemo(() => {
    if (!data || view !== "cdf") return [];
    return (["LU", "LD"] as const)
      .filter((k) => which === "both" || which === k)
      .map((k) => ({ k, top: modelCdf(data.cdf?.[k], k === "LU" ? data.V_LU : data.V_LD).p.at(-1) ?? 1 }))
      .filter((x) => x.top < 0.9995);
  }, [data, view, which]);
  // two-sample KS of the model vs the measured record (finite values), shown in the footnote
  const ksLine = useMemo(() => {
    if (!data?.measured || !showMeas) return [];
    const out: { k: string; D: number; p: number | null }[] = [];
    for (const k of ["LU", "LD"] as const) {
      if (which !== "both" && which !== k) continue;
      const r = ks2(k === "LU" ? data.V_LU : data.V_LD, k === "LU" ? data.measured.V_LU : data.measured.V_LD);
      if (r.D !== null) out.push({ k, D: r.D, p: r.p });
    }
    return out;
  }, [data, showMeas, which]);
  const g = t.l(DEV["menu.series"]);
  const menu: PanelMenuItem[] = [
    { kind: "radio", id: "s-both", group: g, label: "V_LU + V_LD", checked: which === "both", onSelect: () => setWhich("both") },
    { kind: "radio", id: "s-lu", group: g, label: "V_LU", checked: which === "LU", onSelect: () => setWhich("LU") },
    { kind: "radio", id: "s-ld", group: g, label: "V_LD", checked: which === "LD", onSelect: () => setWhich("LD") },
    ...(data?.measured ? [{ kind: "check" as const, id: "meas", label: t.l(DEV["menu.meas"]), checked: showMeas, onChange: setShowMeas, testId: "dist-meas" }] : []),
  ];
  const bits: ReactNode[] = [];
  if (plateau.length)
    bits.push(
      <span key="pl" data-testid="dist-plateau">
        {plateau.map((x, i) => (
          <span key={x.k}>
            {i > 0 && " · "}V<sub>{x.k}</sub> {t("stats.dist.plateau", { pct: fmtShare(x.top) })}
          </span>
        ))}
      </span>,
    );
  if (ksLine.length)
    bits.push(
      <span key="ks" data-testid="dist-ks" title={m && m.V_LU.length > 0 ? measuredCaption(t, m.V_LU.length, params.device.vg, powerMW(params.device)) : undefined}>
        {t("stats.dist.ksHead")}{" "}
        {ksLine.map((x, i) => (
          <span key={x.k}>
            {i > 0 && " · "}V<sub>{x.k}</sub> {t("stats.dist.ksItem", { d: x.D.toFixed(3), p: fmtP(x.p) })}
          </span>
        ))}
      </span>,
    );
  if (data) bits.push(<span key="fold">{t.l(DEV["foot.detFold"])}</span>);
  return (
    <Panel
      id="dist"
      title={t("p.dist")}
      desc={t("p.dist.desc")}
      topic="sweep-mc"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName={`vlu_vld_${view}`}
      plot={plot}
      toolbar={data ? <Seg label={t.l(DEV["tab.dist"])} value={view} onChange={setView} options={[{ v: "hist", label: t("dist.hist") }, { v: "cdf", label: t("dist.cdf") }]} /> : undefined}
      menu={data ? menu : undefined}
      foot={bits.length ? <>{bits.map((b, i) => [i > 0 ? <span key={`s${i}`} className="sep"> · </span> : null, b])}</> : undefined}
    />
  );
}

// ---------------------------------------------------------------- (c) hazard
export function HazardPanel() {
  const t = useT();
  const c = usePalette();
  const { entry, data } = useEntry<HazardResult>("hazard");
  const { hazard: key } = useDeviceKeys();
  const plot = useMemo(() => {
    if (!data) return undefined;
    const v = nums(data.voltage);
    const traces: Data[] = [
      { x: v, y: pos(data.hazard), type: "scatter", mode: "lines", name: t("axis.leg.h"), line: { color: c.hrs, width: 2.2 }, hovertemplate: "V<sub>D</sub> = %{x:.4f} V<br>h = %{y:.3g} 1/s<extra></extra>" },
      { x: v, y: nums(data.survival), type: "scatter", mode: "lines", name: t("axis.leg.S"), yaxis: "y2", line: { color: c.text2, width: 2.2 }, hovertemplate: "V<sub>D</sub> = %{x:.4f} V<br>S = %{y:.4f}<extra></extra>" },
    ];
    const shapes: Partial<Shape>[] = [];
    const ann: NonNullable<Partial<Layout>["annotations"]> = [];
    if (isNum(data.fold_V)) {
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: data.fold_V, x1: data.fold_V, y0: 0, y1: 1, line: { color: c.hrs, width: 1.2, dash: "dash" } });
      ann.push({ x: data.fold_V, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: `fold ${data.fold_V.toFixed(3)} V`, showarrow: false, font: { size: 11, color: c.hrs } });
    }
    const st = data.stats;
    if (isNum(st.mean)) {
      shapes.push({ type: "line", xref: "x", yref: "y2", x0: st.mean, x1: st.mean, y0: 0, y1: 1, line: { color: c.hrs, width: 1, dash: "dot" } });
      ann.push({ x: st.mean, y: 0.5, xref: "x", yref: "y2", text: `⟨V<sub>LU</sub>⟩ = ${st.mean.toFixed(3)} V<br>σ = ${isNum(st.sd) ? (st.sd * 1e3).toFixed(1) : "—"} mV`, showarrow: false, xanchor: "right", xshift: -6, align: "right", font: { size: 11, color: c.text2 } });
    }
    // sweep rate: bottom-left of the survival strip (S = 1 there, below the fold)
    if (isNum(data.rate_V_per_s))
      ann.push({ x: 0.01, y: 0.04, xref: "paper", yref: "y2", xanchor: "left", yanchor: "bottom", text: t("axis.ann.rate", { r: data.rate_V_per_s }), showarrow: false, font: { size: 11, color: c.muted } });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vd") }, anchor: "y2" },
      yaxis: { type: "log", title: { text: t("axis.s.h") }, domain: [0.46, 1], exponentformat: "power", range: logRange([pos(data.hazard)], 0, 10) },
      yaxis2: { domain: [0, 0.38], title: { text: t("axis.s.S") }, range: [-0.03, 1.05] },
      shapes,
      annotations: ann,
      margin: { l: 58, r: 16, t: 40, b: 46 },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, c, t]);
  return (
    <Panel id="hazard" title={t.l(DEV["hazard.title"])} desc={t("p.hazard.desc")} topic="first-passage" entry={entry} hasData={!!data} currentKey={key} csvName="hazard" plot={plot} warnings={data?.warnings} />
  );
}

// ---------------------------------------------------------------- (d) V_G curve (stochastic)
export function VgStochPanel() {
  const t = useT();
  const c = usePalette();
  const range = useStore((s) => s.vgsRange);
  const setRange = useStore((s) => s.setVgsRange);
  const { entry, data } = useEntry<VgCurveStochasticResult>("vg_curve_stochastic");
  const { vg_curve_stochastic: key } = useDeviceKeys();
  const plot = useMemo(() => {
    if (!data) return undefined;
    const vg = nums(data.vg);
    const mean = nums(data.mean_VLU);
    const sd = nums(data.sd_VLU_mV).map((s) => (s == null ? null : s / 1e3));
    const up = mean.map((m, i) => (m == null || sd[i] == null ? null : m + (sd[i] as number)));
    const lo = mean.map((m, i) => (m == null || sd[i] == null ? null : m - (sd[i] as number)));
    const traces: Data[] = [
      { x: vg, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: vg, y: up, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.hrsSoft, name: "± σ", hoverinfo: "skip" },
      { x: vg, y: mean, type: "scatter", mode: "lines+markers", name: t("axis.leg.vluMean"), line: { color: c.hrs, width: 2.2 }, marker: { size: 6 }, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra></extra>" },
      { x: vg, y: nums(data.fold_centre_V), type: "scatter", mode: "lines", name: t("vgs.fold"), line: { color: c.hrs, width: 1.4, dash: "dash" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>V<sub>LU</sub> = %{y:.3f} V<extra>${t("vgs.fold")}</extra>` },
      { x: vg, y: nums(data.VLD_fold_V), type: "scatter", mode: "lines", name: t("axis.leg.vldFold"), line: { color: c.lrs, width: 1.4, dash: "dot" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>V<sub>LD</sub> = %{y:.3f} V<extra>${t("axis.leg.vldFold")}</extra>` },
      { x: vg, y: nums(data.sd_VLU_mV), type: "scatter", mode: "lines+markers", name: t("axis.leg.sdTotal"), yaxis: "y2", line: { color: c.hrs, width: 2 }, marker: { size: 5 }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ<sub>LU</sub> = %{y:.1f} mV<extra>${t("axis.leg.sdTotal")}</extra>` },
      { x: vg, y: nums(data.state_sd_mV), type: "scatter", mode: "lines", name: t("vgs.state"), yaxis: "y2", line: { color: c.categorical[1], width: 1.5, dash: "dash" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ = %{y:.1f} mV<extra>${t("vgs.state")}</extra>` },
      { x: vg, y: nums(data.noise_sd_mV), type: "scatter", mode: "lines", name: t("vgs.noise"), yaxis: "y2", line: { color: c.categorical[2], width: 1.5, dash: "dot" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ = %{y:.1f} mV<extra>${t("vgs.noise")}</extra>` },
    ];
    if (data.measured?.length) {
      traces.push(
        { x: data.measured.map((m) => m.vg), y: data.measured.map((m) => m.mean_V), error_y: { type: "data", array: data.measured.map((m) => m.sd_mV / 1e3), visible: true, color: c.meas, thickness: 1.2, width: 4 }, type: "scatter", mode: "markers", name: t("axis.leg.meanMeas"), marker: { color: c.meas, size: 8, symbol: "square" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra>${t("measured")}</extra>` },
        { x: data.measured.map((m) => m.vg), y: data.measured.map((m) => m.sd_mV), yaxis: "y2", type: "scatter", mode: "markers", name: t("axis.leg.sdMeas"), marker: { color: c.meas, size: 8, symbol: "square-open" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ<sub>LU</sub> = %{y:.1f} mV<extra>${t("measured")}</extra>` },
      );
    }
    // censoring at the sweep maximum (engines fix): stacked bars on a right-hand % axis of the σ strip
    const cens = censoredShares(data);
    if (cens) {
      traces.push(
        { x: vg, y: cens.noLatch.map((v) => (v == null ? null : 100 * v)), yaxis: "y3", type: "bar", name: t("axis.leg.noLatch"), marker: { color: c.unstable }, opacity: 0.35, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>%{y:.1f} %<extra>${t("axis.leg.noLatch")}</extra>` },
        { x: vg, y: cens.beyond.map((v) => (v == null ? null : 100 * v)), yaxis: "y3", type: "bar", name: t("axis.leg.beyond", { v: cens.vdMax }), marker: { color: c.warn }, opacity: 0.35, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>%{y:.1f} %<extra>${t("axis.leg.beyond", { v: cens.vdMax })}</extra>` },
      );
    }
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vg") }, anchor: "y2" },
      yaxis: { title: { text: t("axis.s.vluMean") }, domain: [0.45, 1] },
      yaxis2: { title: { text: t("axis.s.sigmaLu") }, domain: [0, 0.37], rangemode: "tozero" },
      margin: { l: 58, r: cens ? 50 : 16, t: 58, b: 46 },
      legend: { font: { size: 10.5 }, traceorder: "normal" },
    };
    if (cens) {
      layout.yaxis3 = { title: { text: t("axis.s.censored"), font: { size: 11 } }, overlaying: "y2", side: "right", range: [0, 100], showgrid: false, zeroline: false, ticksuffix: "", fixedrange: true };
      layout.barmode = "stack";
      layout.bargap = 0.35;
    }
    return { data: traces, layout, className: "plot tall" };
  }, [data, c, t]);
  const cens = data ? censoredShares(data) : null;
  const running = entry?.status === "running" || entry?.status === "queued";
  const stale = isStale(entry, key);
  const foot: ReactNode[] = [];
  if (cens && cens.peak)
    foot.push(
      <span key="c" data-testid="vgs-censored">
        {t("stats.vgs.foot", { max: fmtShare(cens.peak.total), vg: cens.peak.vg.toFixed(2).replace("-", "−"), nl: fmtShare(cens.peak.noLatch), bs: fmtShare(cens.peak.beyond) })}
      </span>,
    );
  if (stale && !running)
    foot.push(
      <span key="s" className="foot-action">
        {t.l(DEV["foot.staleVgs"])}{" "}
        <button type="button" className="linkish" onClick={() => void runVgStochastic()} data-testid="vgs-recompute">
          {t.l(DEV["foot.recompute"])}
        </button>
      </span>,
    );
  return (
    <Panel
      id="vg-sto"
      title={t("p.vgs")}
      desc={t("p.vgs.desc")}
      topic="local-states"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="vg_curve_stochastic"
      plot={plot}
      warnings={data?.warnings}
      empty={
        <>
          <span>{t("empty.compute")}</span>
          <button type="button" className="btn primary sm" onClick={() => void runVgStochastic()} data-testid="vgs-compute">
            {t("compute")}
          </button>
        </>
      }
      toolbar={<RangePopover t={t} range={range} setRange={setRange} onRun={() => void runVgStochastic()} maxN={61} runLabel={running ? t("loading") : data ? t.l(DEV["range.run"]) : t.l(DEV["range.compute"])} testId="vgs-range" />}
      foot={foot.length ? <>{foot}</> : undefined}
    />
  );
}

/**
 * Censored share per V_G of the stochastic V_G curve: no_latch_weight (no fold) + beyond_sweep_weight (V_LU above
 * the sweep maximum) = censored_weight. Null when the result carries no censoring or none exceeds 0.1 %.
 */
function censoredShares(d: VgCurveStochasticResult) {
  const nl = nums(d.no_latch_weight);
  const bs = d.beyond_sweep_weight ? nums(d.beyond_sweep_weight) : nl.map(() => 0);
  const tot = d.censored_weight ? nums(d.censored_weight) : nl.map((v, i) => (v == null ? null : v + (bs[i] ?? 0)));
  // only V_G values with a fold count (no_latch = 1 outside the latch window is shown by the fold curve)
  let peak: { vg: number; total: number; noLatch: number; beyond: number } | null = null;
  const vg = nums(d.vg);
  const inWindow = (i: number) => (nl[i] ?? 0) < 0.999;
  tot.forEach((v, i) => {
    const g = vg[i];
    if (v == null || g == null || !inWindow(i)) return;
    if (!peak || v > peak.total) peak = { vg: g, total: v, noLatch: nl[i] ?? 0, beyond: bs[i] ?? 0 };
  });
  const any = tot.some((v, i) => v != null && v > 1e-3 && inWindow(i));
  if (!any) return null;
  return { noLatch: nl, beyond: bs, total: tot, vdMax: d.vd_max_V ?? 0, peak: peak as { vg: number; total: number; noLatch: number; beyond: number } | null };
}

// ---------------------------------------------------------------- (e) cycle series
export function CyclePanel() {
  const t = useT();
  const c = usePalette();
  const { entry, data } = useEntry<SweepMCResult>("sweep_mc");
  const { sweep_mc: key } = useDeviceKeys();
  const plot = useMemo(() => {
    if (!data) return undefined;
    const idx = data.V_LU.map((_, i) => i + 1);
    const ax = data.state_axis;
    const unit = ax?.unit ?? "V";
    const toDisp = (v: number | null) => (v == null ? null : unit === "V" ? v * 1e3 : v);
    const traces: Data[] = [
      { x: idx, y: nums(data.V_LU), type: "scatter", mode: "lines+markers", name: "V<sub>LU</sub>", line: { color: c.hrs, width: 1 }, marker: { size: 4 }, hovertemplate: `${t("axis.h.cycle")}<br>V<sub>LU</sub> = %{y:.3f} V<extra></extra>` },
      { x: idx, y: nums(data.V_LD), type: "scatter", mode: "lines+markers", name: "V<sub>LD</sub>", line: { color: c.lrs, width: 1 }, marker: { size: 4 }, hovertemplate: `${t("axis.h.cycle")}<br>V<sub>LD</sub> = %{y:.3f} V<extra></extra>` },
    ];
    const cs = nums(data.cycle_state);
    const hasState = cs.some((v) => v != null && v !== 0);
    if (hasState)
      traces.push({ x: idx, y: cs.map(toDisp), yaxis: "y2", type: "scatter", mode: "lines", name: t("cyc.state"), line: { color: c.categorical[2], width: 1.4 }, hovertemplate: `${t("axis.h.cycle")}<br>δ = %{y:.3g} ${unit === "V" ? "mV" : unit}<extra>${t("cyc.state")}</extra>` });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.cycle") }, anchor: hasState ? "y2" : "y" },
      yaxis: { title: { text: hasState ? t("axis.s.vSwitch") : t("axis.vSwitch") }, domain: hasState ? [0.42, 1] : [0, 1] },
      margin: { l: 58, r: 16, t: 40, b: 46 },
    };
    if (hasState) layout.yaxis2 = { domain: [0, 0.32], title: { text: t("axis.s.delta", { u: unit === "V" ? "mV" : unit }) }, zeroline: true };
    return { data: traces, layout };
  }, [data, c, t]);
  const ax = data?.state_axis;
  return (
    <Panel
      id="cycles"
      title={t.l(DEV["cycles.title"])}
      desc={t("p.cycles.desc")}
      topic="local-states"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="cycle_series"
      plot={plot}
      foot={ax ? `${t("cyc.state")}: ${ax.label} · ${ax.mode} · σ = ${ax.unit === "V" ? `${(ax.sigma * 1e3).toFixed(1)} mV` : `${ax.sigma.toFixed(3)} ${ax.unit}`}` : undefined}
    />
  );
}

// ---------------------------------------------------------------- (f) design map
const DMAP_FIELDS = ["sigma_VLU_mV", "sigma_phi_mV", "latched_fraction", "sigma_VLU_sweep5p2V_mV", "expected_trap_count"] as const;
const DMAP_UNIT: Record<string, string> = { sigma_VLU_mV: "mV", sigma_phi_mV: "mV", latched_fraction: "", sigma_VLU_sweep5p2V_mV: "mV", expected_trap_count: "" };
/** Hover symbol of each design-map quantity (the colorbar title names it). */
const DMAP_SYM: Record<string, string> = { sigma_VLU_mV: "σ<sub>LU</sub>", sigma_phi_mV: "σ<sub>φ</sub>", latched_fraction: "f<sub>latch</sub>", sigma_VLU_sweep5p2V_mV: "σ<sub>LU</sub>", expected_trap_count: "⟨N⟩" };

/** 1e12 → "10<sup>12</sup>", 3e11 → "3×10<sup>11</sup>" (Plotly HTML). */
function pow10(v: number): string {
  const [m, e] = v.toExponential(0).split("e");
  const ex = String(Number(e)).replace("-", "−");
  return m === "1" ? `10<sup>${ex}</sup>` : `${m}×10<sup>${ex}</sup>`;
}

/** Colorbar ticks for a log₁₀ colour scale, labelled with the values themselves (1, 2, 5, 10 …). */
function logColorTicks(values: number[]): { tickvals: number[]; ticktext: string[] } {
  const pos = values.filter((v) => v > 0 && Number.isFinite(v));
  if (!pos.length) return { tickvals: [], ticktext: [] };
  const lo = Math.log10(Math.min(...pos));
  const hi = Math.log10(Math.max(...pos));
  const span = hi - lo;
  const mant = span >= 3 ? [1] : span >= 1 ? [1, 2, 5] : [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9];
  const tickvals: number[] = [];
  const ticktext: string[] = [];
  for (let d = Math.floor(lo); d <= Math.ceil(hi); d++)
    for (const m of mant) {
      const v = m * 10 ** d;
      const lv = Math.log10(v);
      if (lv < lo - 1e-9 || lv > hi + 1e-9) continue;
      tickvals.push(lv);
      ticktext.push(String(Number(v.toPrecision(2))));
    }
  return { tickvals, ticktext };
}

export function DesignMapPanel() {
  const t = useT();
  const c = usePalette();
  const dm = useStore((s) => s.designMap);
  const [field, setField] = useState<string>("sigma_VLU_mV");
  const [logc, setLogc] = useState(true);
  const [lines, setLines] = useState(true);
  useEffect(() => {
    if (dm.status === "idle") void loadDesignMap();
  }, [dm.status]);
  const d = dm.data;
  const plot = useMemo(() => {
    if (!d || !d.fields[field]) return undefined;
    const raw = d.fields[field];
    const z = raw.map((row) => row.map((v) => (v == null || !Number.isFinite(v) ? null : logc ? (v > 0 ? Math.log10(v) : null) : v)));
    const unit = DMAP_UNIT[field] ?? "";
    const sym = DMAP_SYM[field] ?? "";
    // hover shows the value itself (also with the log colour scale); the colorbar ticks too
    const text = raw.map((row) => row.map((v) => (v == null || !Number.isFinite(v) ? "—" : String(Number(v.toPrecision(3))))));
    const ticks = logc ? logColorTicks(raw.flat().filter((v): v is number => typeof v === "number")) : null;
    const traces: Data[] = [
      {
        type: "heatmap", x: d.length_nm, y: d.depth_fraction, z: z as never, text: text as never, colorscale: c.sequential.map((col, i, a) => [i / (a.length - 1), col]) as never,
        reversescale: field === "latched_fraction",
        colorbar: { title: { text: t(`axis.dmap.${field}` as StrKey), side: "right", font: { size: 11 } }, thickness: 12, outlinewidth: 0, tickfont: { size: 10.5, color: c.muted }, ...(ticks ? { tickmode: "array", tickvals: ticks.tickvals, ticktext: ticks.ticktext } : {}) } as never,
        hovertemplate: `L = %{x:.1f} nm<br>d = %{y:.2f}<br>${sym} = %{text}${unit ? ` ${unit}` : ""}<extra></extra>`,
        zsmooth: "best",
      } as Data,
    ];
    if (field === "sigma_phi_mV" && isNum(d.scalars.device_sigma_phi_mV)) {
      traces.push({
        type: "contour", x: d.length_nm, y: d.depth_fraction, z: z as never, showscale: false, hoverinfo: "skip",
        contours: { coloring: "none", start: logc ? Math.log10(d.scalars.device_sigma_phi_mV) : d.scalars.device_sigma_phi_mV, end: logc ? Math.log10(d.scalars.device_sigma_phi_mV) : d.scalars.device_sigma_phi_mV, size: 1, showlabels: false },
        line: { color: c.lrs, width: 2 }, name: t("axis.leg.deviceSigmaPhi", { v: d.scalars.device_sigma_phi_mV.toFixed(1) }), showlegend: true,
      } as Data);
    }
    const shapes: Partial<Shape>[] = [];
    const ann: NonNullable<Partial<Layout>["annotations"]> = [];
    if (lines && d.lines) {
      d.lines.Nt.forEach((nt, i) => {
        const L0 = d.lines!.L0_device[i];
        if (!isNum(L0)) return;
        shapes.push({ type: "line", xref: "x", yref: "paper", x0: L0, x1: L0, y0: 0, y1: 1, line: { color: c.lrs, width: 1.2, dash: "dash" } });
        ann.push({ x: Math.log10(L0), y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: pow10(nt), showarrow: false, font: { size: 10, color: c.lrs }, textangle: "-45" as never, xanchor: "left", xshift: -3 });
        const L50 = d.lines!.L0_50[i];
        if (isNum(L50)) shapes.push({ type: "line", xref: "x", yref: "paper", x0: L50, x1: L50, y0: 0, y1: 1, line: { color: c.text2, width: 1, dash: "dot" } });
      });
    }
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.dmap.L") }, type: "log" },
      yaxis: { title: { text: t("axis.dmap.d") } },
      shapes,
      annotations: ann,
      margin: { l: 58, r: 16, t: 52, b: 46 },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [d, field, logc, lines, c, t]);
  const available = DMAP_FIELDS.filter((f) => !d || d.fields[f]);
  return (
    <Panel
      id="design-map"
      title={t("p.dmap")}
      desc={t("p.dmap.desc")}
      topic="design-map"
      entry={{ status: dm.status === "loading" ? "running" : dm.status === "error" ? "error" : "done", progress: 0, message: t("dmap.loading"), error: dm.error }}
      hasData={!!plot}
      csvName={`design_map_${field}`}
      plot={plot ?? { data: [], layout: {} }}
      empty={dm.status === "error" ? dm.error : t("dmap.loading")}
      toolbar={
        <span className="tool-field">
          <label className="tb-label" htmlFor="dmap-field">{t("dmap.field")}</label>
          <select id="dmap-field" className="select" style={{ width: 170, height: 26 }} value={field} onChange={(e) => setField(e.target.value)} data-testid="dmap-field">
            {available.map((f) => (
              <option key={f} value={f}>{t(`dmap.${f}` as StrKey)}</option>
            ))}
          </select>
        </span>
      }
      menu={[
        { kind: "check", id: "logc", label: t("dmap.logc"), checked: logc, onChange: setLogc, testId: "dmap-logc" },
        { kind: "check", id: "lines", label: t("dmap.lines"), checked: lines, onChange: setLines, testId: "dmap-lines" },
      ]}
      foot={
        d?.lines && lines ? (
          <span>
            <span style={{ color: "var(--lrs)", fontWeight: 700 }}>╌╌</span> <SubText text={t("axis.dmap.footDevice", { v: isNum(d.scalars.device_sigma_phi_mV) ? d.scalars.device_sigma_phi_mV.toFixed(1) : "?" })} /> · <span style={{ fontWeight: 700 }}>┈┈</span>{" "}
            <SubText text={t("axis.dmap.foot50", { v: isNum(d.scalars.phi_50mV) ? d.scalars.phi_50mV.toFixed(1) : "?" })} /> — <SubText text={t("axis.dmap.footNt")} />
          </span>
        ) : undefined
      }
    />
  );
}
