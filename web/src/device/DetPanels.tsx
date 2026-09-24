// Deterministic device panels: (a) I–V branches, (b) current components, (c) body-charge balance,
// (d) V_G curve of the folds.
import type { Data, Layout, Shape } from "plotly.js";
import { useEffect, useMemo, useRef, useState } from "react";
import type { BranchesResult, ChargeBalanceResult, MeasuredData, VgCurveResult } from "../api/types";
import { Panel } from "../components/Panel";
import { useT, type T } from "../i18n";
import { currentAxis, HOVER_IV, type PlotPalette } from "../plots/theme";
import { loadMeasured, runChargeBalance, runVgCurve } from "../state/runner";
import { useStore } from "../state/store";
import { isNum } from "../utils/format";
import { branchesPayload, chargeBalancePayload, midFold, powerMW, vgCurvePayload } from "../utils/payload";
import { abs, arrowIndices, isPaperReference, logRange, nums, pick, pos, rangeWithin, useCurrentKey, useEntry, usePalette } from "./common";

export function Seg<V extends string>({ value, options, onChange, label }: { value: V; options: { v: V; label: string }[]; onChange: (v: V) => void; label: string }) {
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.v} type="button" role="radio" aria-checked={value === o.v} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Check({ checked, onChange, label, testId }: { checked: boolean; onChange: (v: boolean) => void; label: string; testId?: string }) {
  return (
    <button type="button" className="btn sm" aria-pressed={checked} onClick={() => onChange(!checked)} data-testid={testId} style={checked ? { borderColor: "var(--accent)", color: "var(--accent-strong)", background: "var(--accent-soft)" } : undefined}>
      <span aria-hidden style={{ width: 8, height: 8, borderRadius: 2, background: checked ? "var(--accent)" : "var(--border-strong)" }} />
      {label}
    </button>
  );
}

/** Measured overlay traces for I–V panels: paper-device median + 10–90 % band, or photo light I–V curves. */
export function measuredIvTraces(t: T, c: PlotPalette, m: MeasuredData | undefined, kind: "paper" | "photo" | null, powerNow: number): Data[] {
  if (!m || !kind) return [];
  if (kind === "paper" && m.paper_iv) {
    const p = m.paper_iv;
    return [
      { x: p.vd_up, y: pos(p.p10_up), type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup: "meas" },
      { x: p.vd_up, y: pos(p.p90_up), type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.measBand, name: t("iv.measBand"), hoverinfo: "skip", legendgroup: "meas" },
      { x: p.vd_up, y: pos(p.median_up), type: "scatter", mode: "lines", line: { color: c.meas, width: 1.2, dash: "dot" }, name: `${t("iv.measMedian")} ↑↓`, legendgroup: "meas", hovertemplate: `${HOVER_IV}<extra>${t("measured")} ↑</extra>` },
      { x: p.vd_down, y: pos(p.median_down), type: "scatter", mode: "lines", line: { color: c.meas, width: 1.2, dash: "dash" }, name: `${t("iv.measMedian")} ↓`, legendgroup: "meas", showlegend: false, hovertemplate: `${HOVER_IV}<extra>${t("measured")} ↓</extra>` },
    ];
  }
  if (kind === "photo" && m.light_iv.length) {
    let best = 0;
    m.light_iv.forEach((cv, i) => {
      if (cv.power_mW != null && Math.abs(cv.power_mW - powerNow) < Math.abs((m.light_iv[best].power_mW ?? 1e9) - powerNow)) best = i;
    });
    return m.light_iv.map((cv, i) => ({
      x: cv.vd,
      y: pos(cv.id),
      type: "scatter" as const,
      mode: "lines" as const,
      line: { color: c.meas, width: i === best ? 1.6 : 0.8, dash: i === best ? ("solid" as const) : ("dot" as const) },
      opacity: i === best ? 0.9 : 0.35,
      name: `${t("iv.light")} ${cv.label}`,
      legendgroup: "meas",
      showlegend: i === best,
      hovertemplate: `${HOVER_IV}<extra>${cv.label}</extra>`,
    }));
  }
  return [];
}

/** Legend inside the top-left corner of an I–V plot (empty region for log |I_D|). */
export function insideLegend(c: PlotPalette, corner: "tl" | "br" = "br", yBottom = 0.03): Partial<Layout>["legend"] {
  const pos = corner === "tl" ? { x: 0.015, y: 0.985, xanchor: "left" as const, yanchor: "top" as const } : { x: 0.985, y: yBottom, xanchor: "right" as const, yanchor: "bottom" as const };
  return { orientation: "v", ...pos, bgcolor: c.surface.startsWith("#") ? `${c.surface}e6` : c.surface, bordercolor: c.border, borderwidth: 1, font: { size: 10.5 }, tracegroupgap: 0, itemwidth: 30 };
}

function foldAnnotations(f: BranchesResult["folds"], c: PlotPalette): Partial<Layout>["annotations"] {
  const out: NonNullable<Partial<Layout>["annotations"]> = [];
  if (isNum(f.V_LU) && isNum(f.I_LU))
    out.push({ x: f.V_LU, y: Math.log10(f.I_LU), xref: "x", yref: "y", text: `V<sub>LU</sub> = ${f.V_LU.toFixed(3)} V`, showarrow: true, arrowhead: 0, arrowcolor: c.hrs, ax: -58, ay: 26, font: { color: c.hrs, size: 11.5 }, bgcolor: c.surface, bordercolor: c.hrs, borderwidth: 1, borderpad: 3 });
  if (isNum(f.V_LD) && isNum(f.I_LD))
    out.push({ x: f.V_LD, y: Math.log10(f.I_LD), xref: "x", yref: "y", text: `V<sub>LD</sub> = ${f.V_LD.toFixed(3)} V`, showarrow: true, arrowhead: 0, arrowcolor: c.lrs, ax: -62, ay: -28, font: { color: c.lrs, size: 11.5 }, bgcolor: c.surface, bordercolor: c.lrs, borderwidth: 1, borderpad: 3 });
  return out;
}

// ---------------------------------------------------------------- (a) I–V branches
export function IvPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const preset = useStore((s) => s.preset);
  const measured = useStore((s) => s.measured);
  const { entry, data } = useEntry<BranchesResult>("branches");
  const key = useCurrentKey("branches", useMemo(() => branchesPayload(params), [params]));
  const [log, setLog] = useState(true);
  const [showMeas, setShowMeas] = useState(true);
  const [showSweep, setShowSweep] = useState(true);
  const measKind = isPaperReference(params.device) ? "paper" : preset === "photo" ? "photo" : null;
  useEffect(() => {
    if (showMeas && measKind && measured.status === "idle") void loadMeasured();
  }, [showMeas, measKind, measured.status]);

  const plot = useMemo(() => {
    if (!data) return undefined;
    const traces: Data[] = [];
    if (showMeas) traces.push(...measuredIvTraces(t, c, measured.data, measKind, powerMW(params.device)));
    const br = (curve: BranchesResult["HRS"], name: string, color: string, dash: "solid" | "dash", width: number): Data => ({
      x: nums(curve.vd), y: log ? pos(curve.id) : nums(curve.id), type: "scatter", mode: "lines", name, line: { color, width, dash }, hovertemplate: `${HOVER_IV}<extra>${name}</extra>`,
    });
    if (!data.latch) traces.push(br(data.full, t("branch"), c.unstable, "solid", 2));
    traces.push(br(data.HRS, t("iv.hrs"), c.hrs, "solid", 2.4), br(data.unstable, t("iv.unstable"), c.unstable, "dash", 1.6), br(data.LRS, t("iv.lrs"), c.lrs, "solid", 2.4));
    if (showSweep) {
      for (const [dir, xy, color, name] of [
        ["up", data.double_sweep.up, c.up, t("iv.up")],
        ["down", data.double_sweep.down, c.down, t("iv.down")],
      ] as const) {
        const x = nums(xy.vd);
        const y = log ? pos(xy.id) : nums(xy.id);
        traces.push({ x, y, type: "scatter", mode: "lines", line: { color, width: 1.1 }, opacity: 0.85, name, legendgroup: dir, hovertemplate: `${HOVER_IV}<extra>${name}</extra>` });
        const idx = arrowIndices(xy.vd, xy.id, Math.max(4, Math.floor(x.length / 9)));
        traces.push({
          x: pick(x, idx), y: pick(y, idx), type: "scatter", mode: "markers", legendgroup: dir, showlegend: false, hoverinfo: "skip",
          marker: { symbol: "arrow", size: 9, color, angleref: "previous", line: { width: 0 } } as never,
        });
      }
    }
    const f = data.folds;
    if (data.latch && isNum(f.V_LU) && isNum(f.V_LD)) {
      traces.push({
        x: [f.V_LU, f.V_LD], y: [f.I_LU, f.I_LD], type: "scatter", mode: "markers", name: "folds", showlegend: false,
        marker: { symbol: "diamond", size: 10, color: [c.hrs, c.lrs], line: { color: c.surface, width: 1.5 } },
        hovertemplate: "fold: %{x:.4f} V<br>%{y:.3~s}A<extra></extra>",
      });
    }
    const yr = log ? logRange(traces.map((tr) => (tr as { y?: (number | null)[] }).y)) : undefined;
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "V<sub>D</sub> (V)" }, range: [0, params.sweep.vd_max_V + 0.15], zeroline: false },
      yaxis: { ...currentAxis(log), ...(yr ? { range: yr } : {}) },
      annotations: log && data.latch ? foldAnnotations(f, c) : [],
      margin: { l: 64, r: 16, t: 16, b: 46 },
      legend: insideLegend(c, "tl"),
    };
    return { data: traces, layout };
  }, [data, log, showMeas, showSweep, measured.data, measKind, c, t, params.sweep.vd_max_V, params.device]);

  return (
    <Panel
      id="iv"
      title={t("p.iv")}
      desc={t("p.iv.desc")}
      topic="charge-balance"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="iv_branches"
      plot={plot}
      warnings={data?.warnings}
      toolbar={
        <>
          <Seg label="y" value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ v: "log", label: t("log") }, { v: "lin", label: t("lin") }]} />
          <Check checked={showSweep} onChange={setShowSweep} label={`${t("iv.up")}/${t("iv.down")}`} />
          {measKind && <Check checked={showMeas} onChange={setShowMeas} label={t("measured")} testId="toggle-measured" />}
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- (b) current components
export const COMPONENTS: { key: keyof BranchesResult["HRS"]["comp"]; loss: boolean }[] = [
  { key: "ii_total", loss: false },
  { key: "btbt_junction", loss: false },
  { key: "gidl", loss: false },
  { key: "photo", loss: false },
  { key: "channel", loss: false },
  { key: "loss_bulk_srh", loss: true },
  { key: "loss_diffusion", loss: true },
  { key: "loss_junction_srh", loss: true },
];

export function ComponentsPanel() {
  const t = useT();
  const c = usePalette();
  const { entry, data } = useEntry<BranchesResult>("branches");
  const [branch, setBranch] = useState<"HRS" | "LRS" | "full">("HRS");
  const [log, setLog] = useState(true);
  const plot = useMemo(() => {
    if (!data) return undefined;
    const cv = data[branch] ?? data.full;
    const traces: Data[] = COMPONENTS.map((k, i) => {
      const name = t(`comp.${k.key}` as never);
      return {
        x: nums(cv.vd), y: log ? pos(abs(cv.comp?.[k.key])) : abs(cv.comp?.[k.key]), type: "scatter", mode: "lines", name,
        line: { color: c.categorical[i], width: k.loss ? 1.6 : 2, dash: k.loss ? "dash" : "solid" },
        hovertemplate: `V<sub>D</sub> = %{x:.3f} V<br>%{y:.3~s}A<extra>${name}</extra>`,
      } as Data;
    });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: `V<sub>D</sub> (V) — ${branch === "full" ? t("all") : branch}` } },
      yaxis: { ...currentAxis(log, "|I| (A)"), ...(log ? { range: logRange(traces.map((tr) => (tr as { y?: (number | null)[] }).y), 1e-18, 12) } : {}) },
      legend: { orientation: "h", y: 1.01, yanchor: "bottom", x: 0, font: { size: 11 } },
      margin: { l: 64, r: 16, t: 62, b: 46 },
    };
    return { data: traces, layout };
  }, [data, branch, log, c, t]);
  return (
    <Panel
      id="components"
      title={t("p.comp")}
      desc={t("p.comp.desc")}
      topic="impact-ionization"
      entry={entry}
      hasData={!!data}
      csvName="current_components"
      plot={plot}
      toolbar={
        <>
          <Seg label={t("branch")} value={branch} onChange={setBranch} options={[{ v: "HRS", label: "HRS" }, { v: "LRS", label: "LRS" }, { v: "full", label: t("all") }]} />
          <Seg label="y" value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ v: "log", label: t("log") }, { v: "lin", label: t("lin") }]} />
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- (c) body-charge balance
export function ChargeBalancePanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const cbVd = useStore((s) => s.cbVd);
  const setCbVd = useStore((s) => s.setCbVd);
  const { entry, data } = useEntry<ChargeBalanceResult>("charge_balance");
  const { data: br } = useEntry<BranchesResult>("branches");
  const [xq, setXq] = useState<"u" | "Q">("u");
  const autoVd = midFold(br?.folds.V_LU, br?.folds.V_LD, 0.8 * params.sweep.vd_max_V);
  const vd = cbVd ?? data?.vd ?? autoVd;
  const key = useCurrentKey("charge_balance", useMemo(() => chargeBalancePayload(params, cbVd ?? data?.vd ?? autoVd), [params, cbVd, data?.vd, autoVd]));
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const onVd = (v: number) => {
    setCbVd(v);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => void runChargeBalance(v), 350);
  };
  useEffect(() => () => clearTimeout(timer.current), []);

  const plot = useMemo(() => {
    if (!data) return undefined;
    const x = xq === "u" ? nums(data.u) : nums(data.Q_C).map((q) => (q == null ? null : q * 1e15));
    const xTitle = xq === "u" ? "u (V) — source–body bias" : "Q<sub>B</sub> (fC)";
    const xr = (u: number, q: number) => (xq === "u" ? u : q * 1e15);
    const hv = xq === "u" ? "u = %{x:.3f} V" : "Q = %{x:.3f} fC";
    const traces: Data[] = [
      { x, y: pos(data.generation_A), type: "scatter", mode: "lines", name: t("cb.G"), line: { color: c.categorical[0], width: 2 }, hovertemplate: `${hv}<br>G = %{y:.3~s}A<extra></extra>` },
      { x, y: pos(data.loss_A), type: "scatter", mode: "lines", name: t("cb.L"), line: { color: c.categorical[1], width: 2, dash: "dash" }, hovertemplate: `${hv}<br>L = %{y:.3~s}A<extra></extra>` },
      { x, y: nums(data.potential), type: "scatter", mode: "lines", name: t("cb.U"), yaxis: "y2", line: { color: c.categorical[6], width: 2 }, hovertemplate: `${hv}<br>U = %{y:.3g}<extra></extra>` },
    ];
    const stable = data.roots.filter((r) => r.kind === "stable");
    const unstable = data.roots.filter((r) => r.kind === "unstable");
    const potAt = (u: number) => {
      const us = data.u;
      let best = 0;
      for (let i = 0; i < us.length; i++) if (Math.abs((us[i] ?? 1e9) - u) < Math.abs((us[best] ?? 1e9) - u)) best = i;
      return data.potential[best];
    };
    if (stable.length)
      traces.push({ x: stable.map((r) => xr(r.u, r.Q_C)), y: stable.map((r) => potAt(r.u)), yaxis: "y2", type: "scatter", mode: "markers", name: t("cb.stable"), marker: { size: 11, color: c.det, line: { color: c.surface, width: 1.5 } }, hovertemplate: `${t("cb.stable")}<br>${hv}<br>I<sub>D</sub> = %{customdata:.3~s}A<extra></extra>`, customdata: stable.map((r) => r.id) as never });
    if (unstable.length)
      traces.push({ x: unstable.map((r) => xr(r.u, r.Q_C)), y: unstable.map((r) => potAt(r.u)), yaxis: "y2", type: "scatter", mode: "markers", name: t("cb.unstable"), marker: { size: 11, color: c.surface, line: { color: c.warn, width: 2 } }, hovertemplate: `${t("cb.unstable")}<br>${hv}<extra></extra>` });
    // U(x) grows by 10⁴–10⁵ k_BT far from the wells: frame the region around the roots
    const uRoots = data.roots.map((r) => potAt(r.u)).filter((v): v is number => typeof v === "number");
    let uRange: [number, number] | undefined;
    if (uRoots.length >= 2) {
      const lo = Math.min(...uRoots);
      const hi = Math.max(...uRoots);
      const d = hi - lo || 1;
      uRange = [lo - 0.35 * d, hi + 0.8 * d];
    } else if (uRoots.length === 1) {
      const ru = data.roots.map((r) => xr(r.u, r.Q_C));
      uRange = rangeWithin(x, nums(data.potential), ru[0] - 0.15 * (xq === "u" ? 1 : 1), ru[0] + 0.15);
    }
    const shapes: Partial<Shape>[] = data.roots.map((r) => ({
      type: "line", xref: "x", yref: "paper", x0: xr(r.u, r.Q_C), x1: xr(r.u, r.Q_C), y0: 0, y1: 1, line: { color: r.kind === "stable" ? c.det : c.warn, width: 1, dash: "dot" },
    }));
    const layout: Partial<Layout> = {
      grid: undefined,
      xaxis: { title: { text: xTitle }, anchor: "y2" },
      yaxis: { ...currentAxis(true, "G, L (A)"), domain: [0.44, 1], range: logRange([pos(data.generation_A), pos(data.loss_A)], 1e-17, 12) },
      yaxis2: { domain: [0, 0.36], title: { text: "U (k<sub>B</sub>T)" }, zeroline: true, range: uRange },
      shapes,
      margin: { l: 64, r: 16, t: 40, b: 46 },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, xq, c, t]);

  const vmax = params.sweep.vd_max_V;
  return (
    <Panel
      id="charge-balance"
      title={t("p.cb")}
      desc={t("p.cb.desc")}
      topic="charge-balance"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="charge_balance"
      plot={plot}
      warnings={data?.warnings}
      toolbar={
        <>
          <div className="slider-inline">
            <label className="tb-label mono" htmlFor="cb-vd">V<sub>D</sub> = {vd.toFixed(3)} V</label>
            <input id="cb-vd" type="range" min={0.05} max={vmax} step={0.005} value={Math.min(vmax, vd)} onChange={(e) => onVd(Number(e.target.value))} aria-label="V_D" data-testid="cb-vd" />
          </div>
          <button type="button" className="btn sm" onClick={() => { setCbVd(null); void runChargeBalance(autoVd); }} title={t("cb.auto")} aria-pressed={cbVd == null}>
            {t("cb.auto")}
          </button>
          <Seg label={t("cb.x")} value={xq} onChange={setXq} options={[{ v: "u", label: "u" }, { v: "Q", label: "Q_B" }]} />
        </>
      }
    >
      {data && (
        <div className="panel-foot small muted mono">
          {data.roots.length} roots · {data.roots.map((r) => `${r.kind === "stable" ? "●" : "○"} u=${r.u.toFixed(3)} V`).join("  ")}
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- (d) V_G curve
export function VgPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const range = useStore((s) => s.vgRange);
  const setRange = useStore((s) => s.setVgRange);
  const { entry, data } = useEntry<VgCurveResult>("vg_curve");
  const key = useCurrentKey("vg_curve", useMemo(() => vgCurvePayload(params, range), [params, range]));
  const vgNow = params.device.vg;
  const plot = useMemo(() => {
    if (!data) return undefined;
    const vg = nums(data.vg);
    const traces: Data[] = [
      { x: vg, y: nums(data.V_LD), type: "scatter", mode: "lines+markers", name: "V<sub>LD</sub>", line: { color: c.lrs, width: 2 }, marker: { size: 4 }, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>V<sub>LD</sub> = %{y:.4f} V<extra></extra>" },
      { x: vg, y: nums(data.V_LU), type: "scatter", mode: "lines+markers", name: "V<sub>LU</sub>", line: { color: c.hrs, width: 2 }, marker: { size: 4 }, fill: "tonexty", fillcolor: c.detSoft, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>V<sub>LU</sub> = %{y:.4f} V<extra></extra>" },
    ];
    const shapes: Partial<Shape>[] = [];
    const ann: NonNullable<Partial<Layout>["annotations"]> = [];
    const w = data.window;
    if (isNum(w.vg_low) && isNum(w.vg_high)) {
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: w.vg_low, x1: w.vg_high, y0: 0, y1: 1, fillcolor: c.detSoft, opacity: 0.35, line: { width: 0 }, layer: "below" });
      ann.push({ x: (w.vg_low + w.vg_high) / 2, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: `${t("vg.window")}: ${w.vg_low.toFixed(2)} … ${w.vg_high.toFixed(2)} V`, showarrow: false, font: { size: 11, color: c.det } });
    }
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: vgNow, x1: vgNow, y0: 0, y1: 1, line: { color: c.text2, width: 1.2, dash: "dash" } });
    ann.push({ x: vgNow, y: 0.97, xref: "x", yref: "paper", yanchor: "top", text: `${t("vg.current")} ${vgNow.toFixed(2)} V`, showarrow: false, xanchor: "left", xshift: 4, font: { size: 11, color: c.text2 }, bgcolor: c.surface });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "V<sub>G</sub> (V)" } },
      yaxis: { title: { text: "fold V<sub>D</sub> (V)" } },
      shapes,
      annotations: ann,
      margin: { l: 58, r: 16, t: 46, b: 46 },
      legend: { x: 1, xanchor: "right", y: 1.08 },
    };
    return { data: traces, layout };
  }, [data, vgNow, c, t]);
  return (
    <Panel
      id="vg"
      title={t("p.vg")}
      desc={t("p.vg.desc")}
      topic="channel"
      entry={entry}
      hasData={!!data}
      currentKey={key}
      csvName="vg_curve"
      plot={plot}
      warnings={data?.warnings}
      toolbar={<RangeInputs range={range} setRange={setRange} onRun={() => void runVgCurve()} maxN={61} />}
    />
  );
}

export function RangeInputs({ range, setRange, onRun, maxN, label }: { range: { min: number; max: number; n: number }; setRange: (r: Partial<{ min: number; max: number; n: number }>) => void; onRun: () => void; maxN: number; label?: string }) {
  const t = useT();
  const num = (s: string, d: number) => {
    const v = Number(s.replace(",", ".").replace("−", "-"));
    return Number.isFinite(v) ? v : d;
  };
  return (
    <span className="inline-inputs">
      <span className="tb-label">{t("vg.range")}</span>
      <input className="input" aria-label="V_G min" defaultValue={range.min} onBlur={(e) => setRange({ min: num(e.target.value, range.min) })} />
      <span className="muted">…</span>
      <input className="input" aria-label="V_G max" defaultValue={range.max} onBlur={(e) => setRange({ max: num(e.target.value, range.max) })} />
      <input className="input" style={{ width: 48 }} aria-label={t("vg.points")} defaultValue={range.n} onBlur={(e) => setRange({ n: Math.max(2, Math.min(maxN, Math.round(num(e.target.value, range.n)))) })} />
      <span className="tb-label">{t("vg.points")}</span>
      <button type="button" className="btn sm" onClick={onRun}>{label ?? t("recompute")}</button>
    </span>
  );
}

