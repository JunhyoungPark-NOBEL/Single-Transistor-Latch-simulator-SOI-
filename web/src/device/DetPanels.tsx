// Deterministic device panels: (a) I–V curve (hero), (b) current components, (c) body-charge balance,
// (d) V_G curve of the folds. Each panel shows at most one control; view toggles live in its ⋯ menu.
import type { Data, Layout, Shape } from "plotly.js";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Arr, BranchesResult, ChargeBalanceResult, MeasuredData, VgCurveResult } from "../api/types";
import { selectMoreTab } from "../components/MoreCard";
import { NO_CSV, Panel, type PanelMenuItem } from "../components/Panel";
import { ToolPopover } from "../components/PanelMenu";
import { useT, type T } from "../i18n";
import { DEV } from "../i18n/strings.device";
import { fill } from "../i18n/strings.ux";
import { signed, subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { currentAxis, HOVER_IV, type PlotPalette } from "../plots/theme";
import { useIsAll } from "../state/layout";
import { usePrevRun } from "../state/prevRuns";
import { runChargeBalance, runVgCurve } from "../state/runner";
import { useStore } from "../state/store";
import { isNum } from "../utils/format";
import { chargeBalancePayload, midFold, round } from "../utils/payload";
import { abs, arrowIndices, interpAt, logRange, nums, pick, pos, rangeWithin, useCurrentKey, useDeviceKeys, useEntry, usePalette } from "./common";

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

/** [로그 | 선형] segmented control of a current axis. */
function LogSeg({ t, log, setLog }: { t: T; log: boolean; setLog: (v: boolean) => void }) {
  return <Seg label={t.l(DEV["menu.y"])} value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ v: "log", label: t("log") }, { v: "lin", label: t("lin") }]} />;
}

/** ⋯ radio pair y-axis 로그 | 선형. */
export function logMenu(t: T, log: boolean, setLog: (v: boolean) => void): PanelMenuItem[] {
  const g = t.l(DEV["menu.y"]);
  return [
    { kind: "radio", id: "y-log", group: g, label: t("log"), checked: log, onSelect: () => setLog(true) },
    { kind: "radio", id: "y-lin", group: g, label: t("lin"), checked: !log, onSelect: () => setLog(false) },
  ];
}

export interface MeasuredOpts {
  /** Draw the 10–90 % band of the reference record (default true). */
  band?: boolean;
  /** Preserve signed/zero measured currents on a linear axis. */
  log?: boolean;
  /** Legend name of the measured median / curve (default "측정 중앙값 ↑↓"). */
  name?: string;
  /** Legend name of the band. */
  bandName?: string;
  /** Illumination records: draw only the curve nearest to the current power (the 간단히 layout). The other
   *  light powers / V_G values would otherwise sit unlabelled across the model traces. */
  onlyBest?: boolean;
}

/** Measured overlay traces for I–V panels: paper-device median (+ 10–90 % band), or photo light I–V curves. */
export function measuredIvTraces(t: T, c: PlotPalette, m: MeasuredData | undefined, kind: "paper" | "photo" | null, powerNow: number, opts: MeasuredOpts = {}): Data[] {
  if (!m || !kind) return [];
  const Y = opts.log === false ? nums : pos;
  if (kind === "paper" && m.paper_iv) {
    const p = m.paper_iv;
    const name = opts.name ?? `${t("iv.measMedian")} ↑↓`;
    const out: Data[] = [];
    if (opts.band ?? true)
      out.push(
        { x: p.vd_up, y: Y(p.p10_up), type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup: "meas" },
        { x: p.vd_up, y: Y(p.p90_up), type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: c.measBand, name: opts.bandName ?? t("iv.measBand"), hoverinfo: "skip", legendgroup: "meas" },
      );
    out.push(
      { x: p.vd_up, y: Y(p.median_up), type: "scatter", mode: "lines", line: { color: c.meas, width: 1.2, dash: "dot" }, name, legendgroup: "meas", hovertemplate: `${HOVER_IV}<extra>${t("measured")} ↑</extra>` },
      { x: p.vd_down, y: Y(p.median_down), type: "scatter", mode: "lines", line: { color: c.meas, width: 1.2, dash: "dash" }, name: `${name} ↓`, legendgroup: "meas", showlegend: false, hovertemplate: `${HOVER_IV}<extra>${t("measured")} ↓</extra>` },
    );
    return out;
  }
  if (kind === "photo" && m.light_iv.length) {
    let best = 0;
    m.light_iv.forEach((cv, i) => {
      if (cv.power_mW != null && Math.abs(cv.power_mW - powerNow) < Math.abs((m.light_iv[best].power_mW ?? 1e9) - powerNow)) best = i;
    });
    const curves = opts.onlyBest ? [m.light_iv[best]] : m.light_iv;
    if (opts.onlyBest) best = 0;
    return curves.map((cv, i) => ({
      x: cv.vd,
      y: Y(cv.id),
      type: "scatter" as const,
      mode: "lines" as const,
      line: { color: c.meas, width: i === best ? 1.6 : 0.8, dash: i === best ? ("solid" as const) : ("dot" as const) },
      opacity: i === best ? 0.9 : 0.35,
      name: opts.name ?? `${t("iv.light")} ${cv.label}`,
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

/** One horizontal legend row above the plot (at most a few entries: HRS, LRS, 측정, 이전). */
export const ROW_LEGEND: Partial<Layout>["legend"] = { orientation: "h", x: 0, xanchor: "left", y: 1.0, yanchor: "bottom", font: { size: 11.5 }, itemwidth: 30, tracegroupgap: 0 };

type Ann = NonNullable<Partial<Layout>["annotations"]>[number];

// ---------------------------------------------------------------- I_D–V_D voltage sweep
export function IvPanel() {
  const t = useT();
  const c = usePalette();
  const params = useStore((s) => s.params);
  const { entry, data } = useEntry<BranchesResult>("branches");
  const prev = usePrevRun<BranchesResult>("branches");
  const { branches: key } = useDeviceKeys();
  const [log, setLog] = useState(true);
  const [showPrev, setShowPrev] = useState(false);
  const isAll = useIsAll();
  const L = (k: keyof typeof DEV) => t.l(DEV[k]);
  const ghost = showPrev && prev && prev.dataKey !== entry?.dataKey ? prev.data : undefined;
  const [offer, setOffer] = useState<{ vd: number; x: number; y: number } | null>(null);
  const vmax = params.sweep.vd_max_V;

  const plot = useMemo(() => {
    if (!data) return undefined;
    const Y = (a: Arr) => log ? pos(a) : nums(a);
    const traces: Data[] = [];
    if (ghost) {
      const x: (number | null)[] = [];
      const y: (number | null)[] = [];
      for (const cv of [ghost.double_sweep.up, ghost.double_sweep.down]) {
        x.push(...nums(cv.vd), null);
        y.push(...Y(cv.id), null);
      }
      traces.push({ x, y, type: "scatter", mode: "lines", name: L("leg.prev"), line: { color: c.ghost, width: 1.5, dash: "dot" }, meta: NO_CSV as never, connectgaps: false, hovertemplate: `${HOVER_IV}<extra>${L("leg.prev")}</extra>` });
    }
    for (const [dir, xy, color, name] of [
      ["up", data.double_sweep.up, c.hrs, L("leg.up")],
      ["down", data.double_sweep.down, c.lrs, L("leg.down")],
    ] as const) {
      const x = nums(xy.vd);
      const y = Y(xy.id);
      traces.push({ x, y, type: "scatter", mode: "lines", line: { color, width: 2.4 }, name, legendgroup: dir, hovertemplate: `${HOVER_IV}<extra>${name}</extra>` });
      const idx = arrowIndices(xy.vd, xy.id, Math.max(4, Math.floor(x.length / 6)));
      traces.push({ x: pick(x, idx), y: pick(y, idx), type: "scatter", mode: "markers", legendgroup: dir, showlegend: false, hoverinfo: "skip", meta: NO_CSV as never,
        marker: { symbol: "arrow", size: 8, color, angleref: "previous", line: { width: 0 } } as never });
    }
    const yr = log ? logRange(traces.map(tr => (tr as { y?: (number | null)[] }).y)) : undefined;
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vd") }, range: [0, (data.vd_max_V ?? vmax) + 0.05], zeroline: false },
      yaxis: { ...currentAxis(log, log ? t("axis.idAbs") : t("axis.id")), ...(yr ? { range: yr } : {}) },
      margin: { l: 66, r: 20, t: 34, b: 48 },
      legend: ROW_LEGEND,
    };
    return { data: traces, layout };
  }, [data, ghost, log, c, t, vmax]);

  return <Panel
    id="iv" primary title={L("iv.title")} topic="charge-balance"
    entry={entry} hasData={!!data} currentKey={key} csvName="idvd_sweep" plot={plot}
    warnings={data?.warnings}
    toolbar={data ? <LogSeg t={t} log={log} setLog={setLog} /> : undefined}
    menu={data ? [{ kind: "check", id: "prev", label: L("menu.prev"), checked: showPrev, onChange: setShowPrev, testId: "iv-prev" }] : undefined}
    onPlotClick={isAll && data ? ({ x, clientX, clientY }) => (x >= 0.05 && x <= vmax ? setOffer({ vd: round(x, 3), x: clientX, y: clientY }) : setOffer(null)) : undefined}
  >
    {offer && <CbOffer t={t} {...offer} onClose={() => setOffer(null)} />}
  </Panel>;
}

/** Small floating offer at the clicked point of the I–V plot: open the charge balance at that V_D. */
function CbOffer({ t, vd, x, y, onClose }: { t: T; vd: number; x: number; y: number; onClose: () => void }) {
  const setCbVd = useStore((s) => s.setCbVd);
  const isAll = useIsAll();
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ left: x + 10, top: y + 10 });
  useLayoutEffect(() => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    setPos({ left: Math.max(8, Math.min(x + 10, window.innerWidth - r.width - 8)), top: y + 10 + r.height > window.innerHeight - 8 ? y - r.height - 10 : y + 10 });
  }, [x, y]);
  useEffect(() => {
    const down = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("pointerdown", down, true);
    document.addEventListener("keydown", key);
    window.addEventListener("scroll", onClose, true);
    window.addEventListener("resize", onClose);
    return () => {
      document.removeEventListener("pointerdown", down, true);
      document.removeEventListener("keydown", key);
      window.removeEventListener("scroll", onClose, true);
      window.removeEventListener("resize", onClose);
    };
  }, [onClose]);
  const go = () => {
    setCbVd(vd);
    void runChargeBalance(vd);
    if (isAll) document.querySelector('[data-testid="panel-charge-balance"]')?.scrollIntoView({ behavior: "smooth", block: "center" });
    else selectMoreTab("device-det", "charge-balance");
    onClose();
  };
  return createPortal(
    <div ref={ref} className="cb-offer" style={pos} role="group" aria-label={t.l(DEV["cbOffer.aria"])} data-testid="iv-cb-offer">
      <span className="cb-offer-v">
        <SubText text={subs(fill(t.l(DEV["cbOffer.label"]), { v: vd.toFixed(3) }))} />
      </span>
      <button type="button" className="btn sm" onClick={go} data-testid="iv-cb-offer-go">
        {/* one flex item: .btn is inline-flex with a gap, which would split the text around the subscript */}
        <span>
          <SubText text={subs(t.l(DEV["cbOffer.go"]))} /> →
        </span>
      </button>
    </div>,
    document.body,
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
/** Components drawn at first (largest peak |I| on the shown branch); the others start hidden (legend click). */
const TOP_COMPONENTS = 4;

export function ComponentsPanel() {
  const t = useT();
  const c = usePalette();
  const { entry, data } = useEntry<BranchesResult>("branches");
  const { branches: key } = useDeviceKeys();
  const [branch, setBranch] = useState<"HRS" | "LRS" | "full">("HRS");
  const [log, setLog] = useState(true);
  const plot = useMemo(() => {
    if (!data) return undefined;
    // no latch (or an empty branch): HRS/LRS are empty — show the whole traced locus instead of a blank plot
    const used = branch !== "full" && data.latch && (data[branch]?.vd?.length ?? 0) > 0 ? branch : "full";
    const cv = data[used] ?? data.full;
    const peak = COMPONENTS.map((k) => Math.max(0, ...abs(cv.comp?.[k.key]).map((v) => (v != null && Number.isFinite(v) ? v : 0))));
    const top = new Set([...peak.keys()].sort((a, b) => peak[b] - peak[a]).slice(0, TOP_COMPONENTS).filter((i) => peak[i] > 0));
    const traces: Data[] = COMPONENTS.map((k, i) => {
      const name = t(`comp.${k.key}` as never);
      return {
        x: nums(cv.vd), y: log ? pos(abs(cv.comp?.[k.key])) : abs(cv.comp?.[k.key]), type: "scatter", mode: "lines", name,
        visible: top.has(i) ? true : "legendonly",
        line: { color: c.categorical[i], width: k.loss ? 1.6 : 2, dash: k.loss ? "dash" : "solid" },
        hovertemplate: `V<sub>D</sub> = %{x:.3f} V<br>|I| = %{y:.3~s}A<extra>${name}</extra>`,
      } as Data;
    });
    const shown = traces.filter((_, i) => top.has(i)).map((tr) => (tr as { y?: (number | null)[] }).y);
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vd") } },
      yaxis: { ...currentAxis(log, t("axis.icomp")), ...(log ? { range: logRange(shown.length ? shown : traces.map((tr) => (tr as { y?: (number | null)[] }).y), 1e-18, 12) } : {}) },
      // which branch the components follow (the whole traced locus when there is no latch)
      annotations: [{ x: 0.01, y: 0.99, xref: "paper", yref: "paper", xanchor: "left", yanchor: "top", showarrow: false, text: used === "full" ? t("axis.ann.full") : t("axis.ann.branch", { b: used }), font: { size: 11, color: c.text2 }, bgcolor: c.surface, borderpad: 2 }],
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
      currentKey={key}
      csvName="current_components"
      plot={plot}
      toolbar={data ? <Seg label={t("branch")} value={branch} onChange={setBranch} options={[{ v: "HRS", label: "HRS" }, { v: "LRS", label: "LRS" }, { v: "full", label: t("all") }]} /> : undefined}
      menu={data ? logMenu(t, log, setLog) : undefined}
      foot={data ? t.l(DEV["foot.components"]) : undefined}
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
  const [timer] = useState<{ id?: ReturnType<typeof setTimeout> }>({});
  const onVd = (v: number) => {
    setCbVd(v);
    clearTimeout(timer.id);
    timer.id = setTimeout(() => void runChargeBalance(v), 350);
  };
  useEffect(() => () => clearTimeout(timer.id), [timer]);

  const plot = useMemo(() => {
    if (!data) return undefined;
    const x = xq === "u" ? nums(data.u) : nums(data.Q_C).map((q) => (q == null ? null : q * 1e15));
    const xTitle = xq === "u" ? t("axis.cb.u") : t("axis.cb.q");
    const xr = (u: number, q: number) => (xq === "u" ? u : q * 1e15);
    const hv = xq === "u" ? "u = %{x:.3f} V" : "Q<sub>B</sub> = %{x:.3f} fC";
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
      traces.push({ x: stable.map((r) => xr(r.u, r.Q_C)), y: stable.map((r) => potAt(r.u)), yaxis: "y2", type: "scatter", mode: "markers", name: t("cb.stable"), marker: { size: 11, color: c.text, line: { color: c.surface, width: 1.5 } }, hovertemplate: `${t("cb.stable")}<br>${hv}<br>I<sub>D</sub> = %{customdata:.3~s}A<extra></extra>`, customdata: stable.map((r) => r.id) as never });
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
      uRange = rangeWithin(x, nums(data.potential), ru[0] - 0.15, ru[0] + 0.15);
    }
    const shapes: Partial<Shape>[] = data.roots.map((r) => ({
      type: "line", xref: "x", yref: "paper", x0: xr(r.u, r.Q_C), x1: xr(r.u, r.Q_C), y0: 0, y1: 1, line: { color: r.kind === "stable" ? c.text2 : c.warn, width: 1, dash: "dot" },
    }));
    const layout: Partial<Layout> = {
      grid: undefined,
      xaxis: { title: { text: xTitle }, anchor: "y2" },
      yaxis: { ...currentAxis(true, t("axis.s.gl")), domain: [0.44, 1], range: logRange([pos(data.generation_A), pos(data.loss_A)], 1e-17, 12) },
      yaxis2: { domain: [0, 0.36], title: { text: t("axis.s.u") }, zeroline: true, range: uRange },
      shapes,
      margin: { l: 64, r: 16, t: 40, b: 46 },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, xq, c, t]);

  const vmax = params.sweep.vd_max_V;
  const g = t.l(DEV["menu.x"]);
  const menu: PanelMenuItem[] = [
    { kind: "radio", id: "x-u", group: g, label: "u", checked: xq === "u", onSelect: () => setXq("u") },
    { kind: "radio", id: "x-q", group: g, label: "Q_B", checked: xq === "Q", onSelect: () => setXq("Q") },
    {
      kind: "check", id: "auto", label: t.l(DEV["menu.cbAuto"]), checked: cbVd == null, testId: "cb-auto",
      onChange: (on) => {
        if (on) {
          setCbVd(null);
          void runChargeBalance(autoVd);
        } else setCbVd(vd);
      },
    },
  ];
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
        data ? (
          <div className="slider-inline">
            <label className="tb-label mono" htmlFor="cb-vd">
              V<sub>D</sub> = {vd.toFixed(3)} V
            </label>
            <input id="cb-vd" type="range" min={0.05} max={vmax} step={0.005} value={Math.min(vmax, vd)} onChange={(e) => onVd(Number(e.target.value))} aria-label="V_D" data-testid="cb-vd" />
          </div>
        ) : undefined
      }
      menu={data ? menu : undefined}
      foot={data ? `${t("axis.cb.roots", { n: data.roots.length })} · ${data.roots.map((r) => `${r.kind === "stable" ? "●" : "○"} u = ${r.u.toFixed(3)} V`).join("  ")}` : undefined}
    />
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
  // canonical V_G in the payload: moving V_G only moves the "현재 V_G" marker (no stale badge, no re-run)
  const { vg_curve: key } = useDeviceKeys();
  const vgNow = params.device.vg;
  // the sweep peak: a fold above it exists in the model, but the 0 → V_D,max sweep never reaches it
  const vmax = params.sweep.vd_max_V;
  const plot = useMemo(() => {
    if (!data) return undefined;
    const vg = nums(data.vg);
    const vlu = nums(data.V_LU);
    const above = vlu.map((v) => v != null && v > vmax + 1e-9);
    const traces: Data[] = [
      { x: vg, y: nums(data.V_LD), type: "scatter", mode: "lines+markers", name: "V<sub>LD</sub>", line: { color: c.lrs, width: 2 }, marker: { size: 4 }, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>V<sub>LD</sub> = %{y:.4f} V<extra></extra>" },
      {
        x: vg, y: vlu, type: "scatter", mode: "lines+markers", name: "V<sub>LU</sub>", line: { color: c.hrs, width: 2 }, fill: "tonexty", fillcolor: c.neutralSoft,
        // points above V_D,max: hollow grey markers (the sweep turns back before them)
        marker: { size: above.map((a) => (a ? 5 : 4)), color: above.map((a) => (a ? c.surface : c.hrs)), line: { width: above.map((a) => (a ? 1.2 : 0)), color: c.muted } } as never,
        hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>V<sub>LU</sub> = %{y:.4f} V<extra></extra>",
      },
    ];
    const shapes: Partial<Shape>[] = [];
    const ann: Ann[] = [];
    const w = data.window;
    if (isNum(w.vg_low) && isNum(w.vg_high)) {
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: w.vg_low, x1: w.vg_high, y0: 0, y1: 1, fillcolor: c.neutralSoft, opacity: 0.6, line: { width: 0 }, layer: "below" });
      ann.push({ x: (w.vg_low + w.vg_high) / 2, y: 1, xref: "x", yref: "paper", yanchor: "bottom", text: `${t("vg.window")}: ${signed(w.vg_low, 2)} … ${signed(w.vg_high, 2)} V`, showarrow: false, font: { size: 11, color: c.text2 } });
    }
    // V_D,max: dotted line with its label at the left end
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: vmax, y1: vmax, line: { color: c.warn, width: 1.2, dash: "dot" } });
    ann.push({ x: 0.01, y: vmax, xref: "paper", yref: "y", xanchor: "left", yanchor: "bottom", text: subs(fill(t.l(DEV["foot.vdmax"]), { v: String(vmax) })), showarrow: false, font: { size: 11, color: c.warn } });
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: vgNow, x1: vgNow, y0: 0, y1: 1, line: { color: c.text2, width: 1.2, dash: "dash" } });
    // "설정 V_G" midway between the two curves at the set V_G (an empty area), else at the top
    const luNow = interpAt(data.vg, data.V_LU, vgNow);
    const ldNow = interpAt(data.vg, data.V_LD, vgNow);
    const mid = isNum(luNow) && isNum(ldNow) ? (luNow + ldNow) / 2 : null;
    ann.push({
      x: vgNow, xref: "x", ...(mid !== null ? { y: mid, yref: "y" as const, yanchor: "middle" as const } : { y: 0.97, yref: "paper" as const, yanchor: "top" as const }),
      text: t("axis.ann.setVg", { v: signed(vgNow, 2) }), showarrow: false, xanchor: "left", xshift: 4, font: { size: 11, color: c.text2 },
    });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vg") } },
      yaxis: { title: { text: t("axis.foldV") } },
      shapes,
      annotations: ann,
      margin: { l: 58, r: 16, t: 46, b: 46 },
      legend: { x: 1, xanchor: "right", y: 1.08 },
    };
    return { data: traces, layout };
  }, [data, vgNow, vmax, c, t]);
  const vgs = nums(data?.vg).filter((v): v is number => v != null);
  // the V_G ranges whose fold the current sweep reaches (V_LU ≤ V_D,max), as contiguous segments
  const reach = useMemo(() => {
    if (!data) return null;
    const vg = nums(data.vg);
    const vlu = nums(data.V_LU);
    const seg: [number, number][] = [];
    let open: [number, number] | null = null;
    vg.forEach((g, i) => {
      const v = vlu[i];
      const ok = g != null && v != null && v <= vmax + 1e-9 && data.latch?.[i] !== false;
      if (ok) open = open ? [open[0], g] : [g, g];
      else if (open) {
        seg.push(open);
        open = null;
      }
    });
    if (open) seg.push(open);
    const any = vlu.some((v) => v != null && v > vmax + 1e-9);
    return { seg, any };
  }, [data, vmax]);
  const rangeText = vgs.length ? fill(t.l(DEV["foot.range"]), { min: signed(vgs[0], 1), max: signed(vgs[vgs.length - 1], 1), n: vgs.length }) : undefined;
  const reachText =
    reach && reach.any
      ? reach.seg.length
        ? fill(t.l(DEV["foot.vgReach"]), { v: vmax, r: reach.seg.map(([a, b]) => (a === b ? `${signed(a, 2)} V` : `${signed(a, 2)} … ${signed(b, 2)} V`)).join(", ") })
        : fill(t.l(DEV["foot.vgReach.none"]), { v: vmax })
      : null;
  const foot = rangeText ? (
    <>
      {reachText && (
        <>
          <span data-testid="vg-reach">
            <SubText text={subs(reachText)} />
          </span>
          <span className="sep"> · </span>
        </>
      )}
      {rangeText}
    </>
  ) : undefined;
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
      toolbar={<RangePopover t={t} range={range} setRange={setRange} onRun={() => void runVgCurve()} maxN={61} runLabel={t.l(DEV["range.run"])} testId="vg-range" />}
      foot={foot}
    />
  );
}

type VgRangeT = { min: number; max: number; n: number };
const parseNum = (s: string, d: number) => {
  const v = Number(s.trim().replace(",", ".").replace("−", "-"));
  return s.trim() !== "" && Number.isFinite(v) ? v : d;
};

/** ⚙ 범위 popover: V_G min / max / points and one (re)compute button. */
export function RangePopover({ t, range, setRange, onRun, maxN, runLabel, testId }: { t: T; range: VgRangeT; setRange: (r: Partial<VgRangeT>) => void; onRun: () => void; maxN: number; runLabel: string; testId?: string }) {
  return (
    <ToolPopover label={t.l(DEV["range.button"])} title={t.l(DEV["range.title"])} testId={testId}>
      {(close) => <RangeForm t={t} range={range} setRange={setRange} maxN={maxN} runLabel={runLabel} onRun={() => { onRun(); close(); }} testId={testId} />}
    </ToolPopover>
  );
}

function RangeForm({ t, range, setRange, onRun, maxN, runLabel, testId }: { t: T; range: VgRangeT; setRange: (r: Partial<VgRangeT>) => void; onRun: () => void; maxN: number; runLabel: string; testId?: string }) {
  const [min, setMin] = useState(String(range.min));
  const [max, setMax] = useState(String(range.max));
  const [n, setN] = useState(String(range.n));
  const commit = () => {
    const lo = parseNum(min, range.min);
    const hi = parseNum(max, range.max);
    const next = { min: Math.min(lo, hi), max: Math.max(lo, hi), n: Math.max(2, Math.min(maxN, Math.round(parseNum(n, range.n)))) };
    if (next.min === next.max) return;
    setRange(next);
  };
  return (
    <form
      className="range-form"
      onSubmit={(e) => {
        e.preventDefault();
        commit();
        onRun();
      }}
    >
      <label>
        <span>{t.l(DEV["range.min"])}</span>
        <input className="input" inputMode="decimal" value={min} onChange={(e) => setMin(e.target.value)} onBlur={commit} aria-label="V_G min" data-testid={testId ? `${testId}-min` : undefined} />
      </label>
      <label>
        <span>{t.l(DEV["range.max"])}</span>
        <input className="input" inputMode="decimal" value={max} onChange={(e) => setMax(e.target.value)} onBlur={commit} aria-label="V_G max" data-testid={testId ? `${testId}-max` : undefined} />
      </label>
      <label>
        <span>{t.l(DEV["range.n"])}</span>
        <input className="input" inputMode="numeric" value={n} onChange={(e) => setN(e.target.value)} onBlur={commit} aria-label={t("vg.points")} data-testid={testId ? `${testId}-n` : undefined} />
      </label>
      <button type="submit" className="btn sm primary range-run" data-testid={testId ? `${testId}-run` : undefined}>
        {runLabel}
      </button>
    </form>
  );
}
