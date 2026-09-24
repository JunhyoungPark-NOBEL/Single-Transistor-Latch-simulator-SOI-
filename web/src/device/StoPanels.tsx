// Stochastic device panels: statistics summary (full width, rendered under the KPI strip), (a) I–V + MC
// traces, (b) V_LU/V_LD histogram ⇄ CDF, (c) hazard and survival, (d) V_G curve mean ± σ (explicit Compute),
// (e) cycle series, (f) design map.
import type { Data, Layout, Shape } from "plotly.js";
import { useEffect, useMemo, useState } from "react";
import { finite } from "../api/guards";
import type { Arr, BranchesResult, Cdf, HazardResult, SweepMCResult, VgCurveStochasticResult } from "../api/types";
import { Panel } from "../components/Panel";
import { useT, type T } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { loadDesignMap, loadMeasured, runVgStochastic } from "../state/runner";
import { useStore } from "../state/store";
import { describe, diffSeries, ecdf, ks2, meanCI } from "../stats/describe";
import { fmtP, fmtShare } from "../stats/format";
import { StatsTable, type StatsRow } from "../stats/StatsTable";
import { isNum } from "../utils/format";
import { hazardPayload, powerMW, sweepMcPayload, vgStochPayload } from "../utils/payload";
import { isPaperReference, isStale, logRange, nums, pos, useCurrentKey, useEntry, usePalette } from "./common";
import { Check, insideLegend, measuredIvTraces, RangeInputs, Seg } from "./DetPanels";

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

export function StatsPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const { entry, data } = useEntry<SweepMCResult>("sweep_mc");
  const { entry: he, data: hz } = useEntry<HazardResult>("hazard");
  const key = useCurrentKey("sweep_mc", useMemo(() => sweepMcPayload(params), [params]));
  const hKey = useCurrentKey("hazard", useMemo(() => hazardPayload(params), [params]));
  // show the analytic (hazard) row only when it belongs to the same parameter set as the MC result
  const hazardOk = !!hz && hz.fold_V !== null && isStale(he, hKey) === isStale(entry, key);
  const rows = useMemo<StatsRow[]>(() => {
    if (!data) return [];
    const m = data.measured;
    const out: StatsRow[] = [
      { key: "V_LU", label: <>V<sub>LU</sub></>, labelText: "V_LU", sub: t("stats.row.vlu"), unit: "V", values: data.V_LU, measured: m?.V_LU ?? null, color: c.sto },
      { key: "V_LD", label: <>V<sub>LD</sub></>, labelText: "V_LD", sub: t("stats.row.vld"), unit: "V", values: data.V_LD, measured: m?.V_LD ?? null, color: c.down },
      {
        key: "window", label: <>V<sub>LU</sub> − V<sub>LD</sub></>, labelText: "V_LU - V_LD", sub: t("stats.row.window"), unit: "V",
        values: diffSeries(data.V_LU, data.V_LD), measured: m?.V_LD ? diffSeries(m.V_LU, m.V_LD) : null, color: c.det, tip: t("stats.row.window.tip"),
      },
    ];
    if (hazardOk && hz) {
      const st = hz.stats;
      const q = hz.quantiles;
      const atom = isNum(hz.fold_atom) && hz.fold_atom > 0.005 ? ` · ${t("stats.row.hazard.atom", { pct: fmtShare(hz.fold_atom) })}` : "";
      out.push({
        key: "hazard", label: <>V<sub>LU</sub></>, labelText: "V_LU (carrier noise only, hazard)", sub: `${t("stats.row.hazard")}${atom}`, unit: "V",
        secondary: true, color: c.unstable, tip: t("stats.row.hazard.tip"),
        stats: {
          mean: st.mean, sd: st.sd, median: st.median, p05: st.p05, p95: st.p95, min: st.min, max: st.max,
          q1: quantileAt(q.prob, q.v, 0.25), q3: quantileAt(q.prob, q.v, 0.75),
        },
      });
    }
    return out;
  }, [data, hz, hazardOk, c, t]);

  const lu = useMemo(() => (data ? describe(data.V_LU) : null), [data]);
  const ci = lu ? meanCI(lu) : null;
  const m = data?.measured;
  const seed = data?.seed ?? params.stochastic.seed;
  const nTot = lu?.n_total ?? 0;
  // re-running with a result on screen: dim the table instead of the Panel's 120 px progress overlay
  const running = entry?.status === "running" || entry?.status === "queued";
  return (
    <Panel
      id="stats"
      title={t("stats.panel.title")}
      desc={t("stats.panel.desc")}
      topic="sweep-mc"
      entry={running && data && entry ? { ...entry, status: "done" } : entry}
      hasData={!!data}
      currentKey={key}
      wide
      empty={t("stats.empty")}
      badges={data ? <span className="badge sto">{data.engine}</span> : undefined}
    >
      {data && lu && (
        <div className={`stats-panel${running ? " dim" : ""}`} aria-busy={running}>
          <div className="stats-meta" data-testid="stats-meta">
            <span className="stats-chip">{t("stats.meta.engine")} <b>{t(`stats.engine.${data.engine}` as StrKey)}</b></span>
            <span className="stats-chip"><b>{t("stats.meta.cycles", { n: nTot })}</b></span>
            <span className="stats-chip">{t("stats.meta.seed")} <b>{seed}</b></span>
            <span className="stats-chip">{t("stats.meta.sweep", { v: data.vd_max_V ?? params.sweep.vd_max_V, rate: data.rate_V_per_s ?? params.sweep.rate_V_per_s })}</span>
            <span className={`stats-chip${lu.censored > 0 ? " warn" : ""}`} data-testid="stats-censored">
              {t("stats.meta.censored", { n: lu.censored, pct: fmtShare(nTot ? lu.censored / nTot : 0) })}
            </span>
            {ci && (
              <span className="stats-chip" data-testid="stats-ci">
                {t("stats.meta.ci", { lo: ci.lo.toFixed(4), hi: ci.hi.toFixed(4) })}
              </span>
            )}
          </div>
          <StatsTable rows={rows} csvName="statistics_vlu_vld" />
          <div className="stats-foot small muted">
            {t("stats.foot.censoring")}
            {m && m.V_LU.length > 0 && <> {measuredCaption(t, m.V_LU.length, params.device.vg, powerMW(params.device))}</>}
          </div>
        </div>
      )}
    </Panel>
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
  const key = useCurrentKey("sweep_mc", useMemo(() => sweepMcPayload(params), [params]));
  const [log, setLog] = useState(true);
  const [showMeas, setShowMeas] = useState(false);
  const measKind = isPaperReference(params.device) ? "paper" : preset === "photo" ? "photo" : null;
  useEffect(() => {
    if (showMeas && measKind && measured.status === "idle") void loadMeasured();
  }, [showMeas, measKind, measured.status]);

  const plot = useMemo(() => {
    if (!data) return undefined;
    const traces: Data[] = [];
    if (showMeas) traces.push(...measuredIvTraces(t, c, measured.data, measKind, powerMW(params.device)));
    // all MC traces in one trace separated by nulls (fast, one legend entry)
    const x: (number | null)[] = [];
    const y: (number | null)[] = [];
    for (const tr of data.traces) {
      for (const part of [tr.up, tr.down]) {
        x.push(...nums(part.vd), null);
        y.push(...(log ? pos(part.id) : nums(part.id)), null);
      }
    }
    if (x.length) traces.push({ x, y, type: "scatter", mode: "lines", name: `${t("iv.traces")} (${data.traces.length})`, line: { color: c.sto, width: 1 }, opacity: 0.38, hoverinfo: "skip", connectgaps: false });
    const hrs = data.centre.HRS.vd.length ? data.centre.HRS : br ? { vd: br.HRS.vd, id: br.HRS.id } : null;
    const lrs = data.centre.LRS.vd.length ? data.centre.LRS : br ? { vd: br.LRS.vd, id: br.LRS.id } : null;
    if (hrs) traces.push({ x: nums(hrs.vd), y: log ? pos(hrs.id) : nums(hrs.id), type: "scatter", mode: "lines", name: `HRS · ${t("iv.centre")}`, line: { color: c.hrs, width: 2.2 }, hovertemplate: `${HOVER_IV}<extra>HRS</extra>` });
    if (lrs) traces.push({ x: nums(lrs.vd), y: log ? pos(lrs.id) : nums(lrs.id), type: "scatter", mode: "lines", name: `LRS · ${t("iv.centre")}`, line: { color: c.lrs, width: 2.2 }, hovertemplate: `${HOVER_IV}<extra>LRS</extra>` });
    if (br?.latch) traces.push({ x: nums(br.unstable.vd), y: log ? pos(br.unstable.id) : nums(br.unstable.id), type: "scatter", mode: "lines", name: t("iv.unstable"), line: { color: c.unstable, width: 1.4, dash: "dash" }, hoverinfo: "skip" });
    // per-cycle rug in a bottom strip
    const lu = finite(data.V_LU);
    const ld = finite(data.V_LD);
    traces.push(
      { x: lu, y: lu.map(() => 1), yaxis: "y2", type: "scatter", mode: "markers", name: `V<sub>LU</sub> ${t("iv.rug")}`, marker: { symbol: "line-ns-open", size: 14, color: c.sto, line: { width: 1.2 } }, opacity: 0.6, hovertemplate: "V<sub>LU</sub> = %{x:.3f} V<extra></extra>" },
      { x: ld, y: ld.map(() => 0), yaxis: "y2", type: "scatter", mode: "markers", name: `V<sub>LD</sub> ${t("iv.rug")}`, marker: { symbol: "line-ns-open", size: 14, color: c.down, line: { width: 1.2 } }, opacity: 0.6, hovertemplate: "V<sub>LD</sub> = %{x:.3f} V<extra></extra>" },
    );
    const shapes: Partial<Shape>[] = [];
    for (const [v, col] of [[data.stats.LU.mean, c.sto], [data.stats.LD.mean, c.down]] as const)
      if (isNum(v)) shapes.push({ type: "line", xref: "x", yref: "paper", x0: v, x1: v, y0: 0, y1: 1, line: { color: col, width: 1, dash: "dot" } });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "V<sub>D</sub> (V)" }, range: [0, params.sweep.vd_max_V + 0.15], anchor: "y2" },
      yaxis: { ...currentAxis(log), domain: [0.16, 1], ...(log ? { range: logRange(traces.filter((tr) => (tr as { yaxis?: string }).yaxis !== "y2").map((tr) => (tr as { y?: (number | null)[] }).y)) } : {}) },
      yaxis2: { domain: [0, 0.1], range: [-0.8, 1.8], showticklabels: false, showgrid: false, zeroline: false, ticks: "", showline: false, fixedrange: true },
      shapes,
      margin: { l: 64, r: 16, t: 16, b: 46 },
      legend: insideLegend(c, "tl"),
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, br, log, showMeas, measured.data, measKind, params.sweep.vd_max_V, params.device, c, t]);

  return (
    <Panel
      id="mc-iv"
      title={t("p.mciv")}
      desc={t("p.mciv.desc")}
      topic="sweep-mc"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="mc_iv"
      plot={plot}
      warnings={data?.warnings}
      badges={data ? <span className="badge sto">{data.engine}</span> : undefined}
      toolbar={
        <>
          <Seg label="y" value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ v: "log", label: t("log") }, { v: "lin", label: t("lin") }]} />
          {measKind && <Check checked={showMeas} onChange={setShowMeas} label={t("measured")} />}
        </>
      }
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
  const key = useCurrentKey("sweep_mc", useMemo(() => sweepMcPayload(params), [params]));
  const [view, setView] = useState<"hist" | "cdf">("hist");
  const [showMeas, setShowMeas] = useState(true);
  const [which, setWhich] = useState<"both" | "LU" | "LD">("both");
  const plot = useMemo(() => {
    if (!data) return undefined;
    const traces: Data[] = [];
    const meas = showMeas ? data.measured : null;
    const series = [
      { k: "LU" as const, color: c.sto, label: "V<sub>LU</sub>", raw: data.V_LU, model: finite(data.V_LU), meas: finite(meas?.V_LU) },
      { k: "LD" as const, color: c.down, label: "V<sub>LD</sub>", raw: data.V_LD, model: finite(data.V_LD), meas: finite(meas?.V_LD) },
    ].filter((s) => which === "both" || which === s.k);
    for (const s of series) {
      if (view === "hist") {
        const edges = commonEdges(s.model, s.meas, data.hist[s.k].edges);
        const counts = histFromEdges(s.model, edges);
        const centers = counts.map((_, i) => (edges[i] + edges[i + 1]) / 2);
        const widths = counts.map((_, i) => edges[i + 1] - edges[i]);
        traces.push({ x: centers, y: counts, width: widths, type: "bar", name: `${s.label} ${t("model")}`, marker: { color: s.color, line: { color: c.surface, width: 0.5 } }, opacity: 0.8, hovertemplate: `${s.label} %{x:.3f} V<br>n = %{y}<extra></extra>` });
        if (s.meas.length) {
          const mc = histFromEdges(s.meas, edges);
          const scale = s.model.length / s.meas.length;
          traces.push({ x: centers, y: mc.map((v) => v * scale), type: "scatter", mode: "lines", line: { shape: "hvh", color: c.meas, width: 1.6 }, name: `${s.label} ${t("measured")}${Math.abs(scale - 1) > 1e-9 ? ` (×${scale.toFixed(2)})` : ""}`, hovertemplate: `${t("measured")} %{x:.3f} V<br>%{y:.1f}<extra></extra>` });
        }
      } else {
        const cm = modelCdf(data.cdf?.[s.k], s.raw);
        traces.push({ x: cm.v, y: cm.p, type: "scatter", mode: "lines", line: { shape: "hv", color: s.color, width: 2 }, name: `${s.label} ${t("model")}`, hovertemplate: `%{x:.3f} V<br>P = %{y:.3f}<extra></extra>` });
        if (s.meas.length) {
          const mm = ecdf(s.meas);
          traces.push({ x: mm.v, y: mm.p, type: "scatter", mode: "lines", line: { shape: "hv", color: c.meas, width: 1.6, dash: "dot" }, name: `${s.label} ${t("measured")}`, hovertemplate: `%{x:.3f} V<br>P = %{y:.3f}<extra></extra>` });
        }
      }
    }
    const layout: Partial<Layout> = {
      barmode: "overlay",
      bargap: 0,
      xaxis: { title: { text: "V (V)" } },
      yaxis: view === "hist" ? { title: { text: "counts" }, rangemode: "tozero" } : { title: { text: "P(V ≤ x)" }, range: [0, 1.02] },
      margin: { l: 56, r: 16, t: 40, b: 46 },
    };
    return { data: traces, layout };
  }, [data, view, showMeas, which, c, t]);
  const m = data?.measured;
  // two-sample KS of the model vs the measured record (finite values), shown under the plot
  // censored cycles: the (all-cycle) CDF plateaus at the fraction that switched within the sweep
  const plateau = useMemo(() => {
    if (!data || view !== "cdf") return [];
    return (["LU", "LD"] as const)
      .filter((k) => which === "both" || which === k)
      .map((k) => ({ k, top: modelCdf(data.cdf?.[k], k === "LU" ? data.V_LU : data.V_LD).p.at(-1) ?? 1 }))
      .filter((x) => x.top < 0.9995);
  }, [data, view, which]);
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
      toolbar={
        <>
          <Seg label="view" value={view} onChange={setView} options={[{ v: "hist", label: t("dist.hist") }, { v: "cdf", label: t("dist.cdf") }]} />
          <Seg label="series" value={which} onChange={setWhich} options={[{ v: "both", label: "LU + LD" }, { v: "LU", label: "LU" }, { v: "LD", label: "LD" }]} />
          {data?.measured && <Check checked={showMeas} onChange={setShowMeas} label={t("measured")} />}
        </>
      }
    >
      {plateau.length > 0 && (
        <div className="panel-foot small muted" data-testid="dist-plateau">
          <span>
            {plateau.map((x, i) => (
              <span key={x.k}>
                {i > 0 && " · "}V<sub>{x.k}</sub> {t("stats.dist.plateau", { pct: fmtShare(x.top) })}
              </span>
            ))}
          </span>
        </div>
      )}
      {m && m.V_LU.length > 0 && (
        <div className="panel-foot small muted">
          {ksLine.length > 0 && (
            <span data-testid="dist-ks">
              {ksLine.map((x, i) => (
                <span key={x.k}>
                  {i > 0 && " · "}V<sub>{x.k}</sub> {t("stats.dist.ks", { d: x.D.toFixed(3), p: fmtP(x.p) })}
                </span>
              ))}
            </span>
          )}
          <span>{measuredCaption(t, m.V_LU.length, params.device.vg, powerMW(params.device))}</span>
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- (c) hazard
export function HazardPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const { entry, data } = useEntry<HazardResult>("hazard");
  const key = useCurrentKey("hazard", useMemo(() => hazardPayload(params), [params]));
  const plot = useMemo(() => {
    if (!data) return undefined;
    const v = nums(data.voltage);
    const traces: Data[] = [
      { x: v, y: pos(data.hazard), type: "scatter", mode: "lines", name: "h(V<sub>D</sub>)", line: { color: c.sto, width: 2.2 }, hovertemplate: "V<sub>D</sub> = %{x:.4f} V<br>h = %{y:.3~s}/s<extra></extra>" },
      { x: v, y: nums(data.survival), type: "scatter", mode: "lines", name: "S(V<sub>D</sub>)", yaxis: "y2", line: { color: c.det, width: 2.2 }, hovertemplate: "V<sub>D</sub> = %{x:.4f} V<br>S = %{y:.4f}<extra></extra>" },
    ];
    const shapes: Partial<Shape>[] = [];
    const ann: NonNullable<Partial<Layout>["annotations"]> = [];
    if (isNum(data.fold_V)) {
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: data.fold_V, x1: data.fold_V, y0: 0, y1: 1, line: { color: c.hrs, width: 1.2, dash: "dash" } });
      ann.push({ x: data.fold_V, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: `fold ${data.fold_V.toFixed(3)} V`, showarrow: false, font: { size: 11, color: c.hrs } });
    }
    const st = data.stats;
    if (isNum(st.mean)) {
      shapes.push({ type: "line", xref: "x", yref: "y2", x0: st.mean, x1: st.mean, y0: 0, y1: 1, line: { color: c.sto, width: 1, dash: "dot" } });
      ann.push({ x: st.mean, y: 0.5, xref: "x", yref: "y2", text: `⟨V<sub>LU</sub>⟩ = ${st.mean.toFixed(3)} V<br>σ = ${isNum(st.sd) ? (st.sd * 1e3).toFixed(1) : "—"} mV`, showarrow: false, xanchor: "right", xshift: -6, align: "right", font: { size: 11, color: c.text2 } });
    }
    const layout: Partial<Layout> = {
      xaxis: { title: { text: `V<sub>D</sub> (V) — ${data.rate_V_per_s} V/s` }, anchor: "y2" },
      yaxis: { type: "log", title: { text: "h (1/s)" }, domain: [0.46, 1], exponentformat: "power", range: logRange([pos(data.hazard)], 0, 10) },
      yaxis2: { domain: [0, 0.38], title: { text: "S" }, range: [-0.03, 1.05] },
      shapes,
      annotations: ann,
      margin: { l: 58, r: 16, t: 40, b: 46 },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, c]);
  return (
    <Panel id="hazard" title={t("p.hazard")} desc={t("p.hazard.desc")} topic="first-passage" entry={entry} hasData={!!data} currentKey={key} csvName="hazard" plot={plot} warnings={data?.warnings} />
  );
}

// ---------------------------------------------------------------- (d) V_G curve (stochastic)
export function VgStochPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const range = useStore((s) => s.vgsRange);
  const setRange = useStore((s) => s.setVgsRange);
  const { entry, data } = useEntry<VgCurveStochasticResult>("vg_curve_stochastic");
  const key = useCurrentKey("vg_curve_stochastic", useMemo(() => vgStochPayload(params, range), [params, range]));
  const plot = useMemo(() => {
    if (!data) return undefined;
    const vg = nums(data.vg);
    const mean = nums(data.mean_VLU);
    const sd = nums(data.sd_VLU_mV).map((s) => (s == null ? null : s / 1e3));
    const up = mean.map((m, i) => (m == null || sd[i] == null ? null : m + (sd[i] as number)));
    const lo = mean.map((m, i) => (m == null || sd[i] == null ? null : m - (sd[i] as number)));
    const traces: Data[] = [
      { x: vg, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: vg, y: up, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.stoSoft, name: "± σ", hoverinfo: "skip" },
      { x: vg, y: mean, type: "scatter", mode: "lines+markers", name: t("vgs.mean"), line: { color: c.sto, width: 2.2 }, marker: { size: 6 }, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra></extra>" },
      { x: vg, y: nums(data.fold_centre_V), type: "scatter", mode: "lines", name: t("vgs.fold"), line: { color: c.hrs, width: 1.4, dash: "dash" }, hovertemplate: "fold %{y:.3f} V<extra></extra>" },
      { x: vg, y: nums(data.VLD_fold_V), type: "scatter", mode: "lines", name: "V<sub>LD</sub> fold", line: { color: c.lrs, width: 1.4, dash: "dot" }, hovertemplate: "V<sub>LD</sub> %{y:.3f} V<extra></extra>" },
      { x: vg, y: nums(data.sd_VLU_mV), type: "scatter", mode: "lines+markers", name: t("vgs.sd"), yaxis: "y2", line: { color: c.sto, width: 2 }, marker: { size: 5 }, hovertemplate: "σ<sub>LU</sub> = %{y:.1f} mV<extra></extra>" },
      { x: vg, y: nums(data.state_sd_mV), type: "scatter", mode: "lines", name: t("vgs.state"), yaxis: "y2", line: { color: c.categorical[1], width: 1.5, dash: "dash" }, hovertemplate: "state %{y:.1f} mV<extra></extra>" },
      { x: vg, y: nums(data.noise_sd_mV), type: "scatter", mode: "lines", name: t("vgs.noise"), yaxis: "y2", line: { color: c.categorical[2], width: 1.5, dash: "dot" }, hovertemplate: "noise %{y:.1f} mV<extra></extra>" },
    ];
    if (data.measured?.length) {
      traces.push(
        { x: data.measured.map((m) => m.vg), y: data.measured.map((m) => m.mean_V), error_y: { type: "data", array: data.measured.map((m) => m.sd_mV / 1e3), visible: true, color: c.meas, thickness: 1.2, width: 4 }, type: "scatter", mode: "markers", name: t("measured"), marker: { color: c.meas, size: 8, symbol: "square" }, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>%{y:.3f} V<extra>measured</extra>" },
        { x: data.measured.map((m) => m.vg), y: data.measured.map((m) => m.sd_mV), yaxis: "y2", type: "scatter", mode: "markers", name: `σ ${t("measured")}`, marker: { color: c.meas, size: 8, symbol: "square-open" }, hovertemplate: "σ = %{y:.1f} mV<extra>measured</extra>" },
      );
    }
    // censoring at the sweep maximum (engines fix): stacked bars on a right-hand % axis of the σ strip
    const cens = censoredShares(data);
    if (cens) {
      traces.push(
        { x: vg, y: cens.noLatch.map((v) => (v == null ? null : 100 * v)), yaxis: "y3", type: "bar", name: t("stats.vgs.noLatch"), marker: { color: c.unstable }, opacity: 0.35, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>${t("stats.vgs.noLatch")}: %{y:.1f} %<extra></extra>` },
        { x: vg, y: cens.beyond.map((v) => (v == null ? null : 100 * v)), yaxis: "y3", type: "bar", name: t("stats.vgs.beyond", { v: cens.vdMax }), marker: { color: c.warn }, opacity: 0.35, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>${t("stats.vgs.beyond", { v: cens.vdMax })}: %{y:.1f} %<extra></extra>` },
      );
    }
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "V<sub>G</sub> (V)" }, anchor: "y2" },
      yaxis: { title: { text: "V<sub>LU</sub> (V)" }, domain: [0.45, 1] },
      yaxis2: { title: { text: "σ (mV)" }, domain: [0, 0.37], rangemode: "tozero" },
      margin: { l: 58, r: cens ? 50 : 16, t: 58, b: 46 },
      legend: { font: { size: 10.5 } },
    };
    if (cens) {
      layout.yaxis3 = { title: { text: t("stats.vgs.axis"), font: { size: 11 } }, overlaying: "y2", side: "right", range: [0, 100], showgrid: false, zeroline: false, ticksuffix: "", fixedrange: true };
      layout.barmode = "stack";
      layout.bargap = 0.35;
    }
    return { data: traces, layout, className: "plot tall" };
  }, [data, c, t]);
  const cens = data ? censoredShares(data) : null;
  const running = entry?.status === "running" || entry?.status === "queued";
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
      toolbar={<RangeInputs range={range} setRange={setRange} onRun={() => void runVgStochastic()} maxN={61} label={running ? t("loading") : data ? t("recompute") : t("compute")} />}
    >
      {cens && cens.peak && (
        <div className="panel-foot small muted" data-testid="vgs-censored">
          {t("stats.vgs.foot", { max: fmtShare(cens.peak.total), vg: cens.peak.vg.toFixed(2).replace("-", "−"), nl: fmtShare(cens.peak.noLatch), bs: fmtShare(cens.peak.beyond) })}
        </div>
      )}
    </Panel>
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
  const params = useStore((s) => s.params);
  const { entry, data } = useEntry<SweepMCResult>("sweep_mc");
  const key = useCurrentKey("sweep_mc", useMemo(() => sweepMcPayload(params), [params]));
  const plot = useMemo(() => {
    if (!data) return undefined;
    const idx = data.V_LU.map((_, i) => i + 1);
    const ax = data.state_axis;
    const unit = ax?.unit ?? "V";
    const toDisp = (v: number | null) => (v == null ? null : unit === "V" ? v * 1e3 : v);
    const traces: Data[] = [
      { x: idx, y: nums(data.V_LU), type: "scatter", mode: "lines+markers", name: "V<sub>LU</sub>", line: { color: c.sto, width: 1 }, marker: { size: 4 }, hovertemplate: "cycle %{x}<br>V<sub>LU</sub> = %{y:.3f} V<extra></extra>" },
      { x: idx, y: nums(data.V_LD), type: "scatter", mode: "lines+markers", name: "V<sub>LD</sub>", line: { color: c.down, width: 1 }, marker: { size: 4 }, hovertemplate: "cycle %{x}<br>V<sub>LD</sub> = %{y:.3f} V<extra></extra>" },
    ];
    const cs = nums(data.cycle_state);
    const hasState = cs.some((v) => v != null && v !== 0);
    if (hasState)
      traces.push({ x: idx, y: cs.map(toDisp), yaxis: "y2", type: "scatter", mode: "lines", name: t("cyc.state"), line: { color: c.categorical[2], width: 1.4 }, hovertemplate: `cycle %{x}<br>δ = %{y:.3g} ${unit === "V" ? "mV" : unit}<extra></extra>` });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "cycle" }, anchor: hasState ? "y2" : "y" },
      yaxis: { title: { text: "V (V)" }, domain: hasState ? [0.42, 1] : [0, 1] },
      margin: { l: 58, r: 16, t: 40, b: 46 },
    };
    if (hasState) layout.yaxis2 = { domain: [0, 0.32], title: { text: `δ (${unit === "V" ? "mV" : unit})` }, zeroline: true };
    return { data: traces, layout };
  }, [data, c, t]);
  const ax = data?.state_axis;
  return (
    <Panel id="cycles" title={t("p.cycles")} desc={t("p.cycles.desc")} topic="local-states" entry={entry} hasData={!!data} currentKey={key} csvName="cycle_series" plot={plot}>
      {ax && (
        <div className="panel-foot small muted">
          {t("cyc.state")}: {ax.label} · {ax.mode} · σ = {ax.unit === "V" ? `${(ax.sigma * 1e3).toFixed(1)} mV` : `${ax.sigma.toFixed(3)} ${ax.unit}`}
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- (f) design map
const DMAP_FIELDS = ["sigma_VLU_mV", "sigma_phi_mV", "latched_fraction", "sigma_VLU_sweep5p2V_mV", "expected_trap_count"] as const;
const DMAP_UNIT: Record<string, string> = { sigma_VLU_mV: "mV", sigma_phi_mV: "mV", latched_fraction: "", sigma_VLU_sweep5p2V_mV: "mV", expected_trap_count: "" };

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
    const z = d.fields[field].map((row) => row.map((v) => (v == null || !Number.isFinite(v) ? null : logc ? (v > 0 ? Math.log10(v) : null) : v)));
    const unit = DMAP_UNIT[field] ?? "";
    const traces: Data[] = [
      {
        type: "heatmap", x: d.length_nm, y: d.depth_fraction, z: z as never, colorscale: c.sequential.map((col, i, a) => [i / (a.length - 1), col]) as never,
        reversescale: field === "latched_fraction",
        colorbar: { title: { text: `${logc ? "log₁₀ " : ""}${t(`dmap.${field}` as StrKey)}${unit ? ` (${unit})` : ""}`, side: "right", font: { size: 11 } }, thickness: 12, outlinewidth: 0, tickfont: { size: 10.5, color: c.muted } } as never,
        hovertemplate: `L = %{x:.1f} nm<br>depth = %{y:.2f}<br>${logc ? "log₁₀ " : ""}%{z:.3g}<extra></extra>`,
        zsmooth: "best",
      } as Data,
    ];
    if (field === "sigma_phi_mV" && isNum(d.scalars.device_sigma_phi_mV)) {
      traces.push({
        type: "contour", x: d.length_nm, y: d.depth_fraction, z: z as never, showscale: false, hoverinfo: "skip",
        contours: { coloring: "none", start: logc ? Math.log10(d.scalars.device_sigma_phi_mV) : d.scalars.device_sigma_phi_mV, end: logc ? Math.log10(d.scalars.device_sigma_phi_mV) : d.scalars.device_sigma_phi_mV, size: 1, showlabels: false },
        line: { color: c.lrs, width: 2 }, name: `σ_φ = ${d.scalars.device_sigma_phi_mV.toFixed(1)} mV (device)`, showlegend: true,
      } as Data);
    }
    const shapes: Partial<Shape>[] = [];
    const ann: NonNullable<Partial<Layout>["annotations"]> = [];
    if (lines && d.lines) {
      d.lines.Nt.forEach((nt, i) => {
        const L0 = d.lines!.L0_device[i];
        if (!isNum(L0)) return;
        shapes.push({ type: "line", xref: "x", yref: "paper", x0: L0, x1: L0, y0: 0, y1: 1, line: { color: c.lrs, width: 1.2, dash: "dash" } });
        ann.push({ x: Math.log10(L0), y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: `N<sub>t</sub>=${nt.toExponential(0).replace("e+", "e")}`, showarrow: false, font: { size: 10, color: c.lrs }, textangle: "-30" as never });
        const L50 = d.lines!.L0_50[i];
        if (isNum(L50)) shapes.push({ type: "line", xref: "x", yref: "paper", x0: L50, x1: L50, y0: 0, y1: 1, line: { color: c.text2, width: 1, dash: "dot" } });
      });
    }
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "local-region size L (nm)" }, type: "log" },
      yaxis: { title: { text: "depth fraction" } },
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
        <>
          <label className="tb-label" htmlFor="dmap-field">{t("dmap.field")}</label>
          <select id="dmap-field" className="select" style={{ width: 170, height: 26 }} value={field} onChange={(e) => setField(e.target.value)}>
            {available.map((f) => (
              <option key={f} value={f}>{t(`dmap.${f}` as StrKey)}</option>
            ))}
          </select>
          <Check checked={logc} onChange={setLogc} label={t("dmap.logc")} />
          <Check checked={lines} onChange={setLines} label={t("dmap.lines")} />
        </>
      }
    >
      {d?.lines && lines && (
        <div className="panel-foot small muted">
          <span>
            <span style={{ color: "var(--lrs)", fontWeight: 700 }}>╌╌</span> L₀ (σ_φ = device {isNum(d.scalars.device_sigma_phi_mV) ? d.scalars.device_sigma_phi_mV.toFixed(0) : "?"} mV) · <span style={{ fontWeight: 700 }}>┈┈</span> L₀ ({isNum(d.scalars.phi_50mV) ? d.scalars.phi_50mV.toFixed(1) : "?"} mV) — N<sub>t</sub> {t("dmap.lines")}
          </span>
        </div>
      )}
    </Panel>
  );
}
