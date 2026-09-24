// LTspice-style waveform viewer for the schematic run: traces picked by clicking the schematic (chips to
// remove), shared time axis with voltage / current / charge / state subplots, log |I|, stochastic runs
// overlay + mean ± SD band from `envelopes`, a draggable time cursor with a slider and a readout, summary
// cards, per-STL event statistics, per-run distributions (shared stats module) and solver information.
import type { Data, Layout } from "plotly.js";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { iKey, parseProbe, valueAt, type CustomCircuitResult, type Envelope } from "../api/circuitCustom";
import type { Arr, Signal } from "../api/types";
import { Notices, Panel } from "../components/Panel";
import { IconX } from "../components/icons";
import { Kpi } from "../device/KpiStrip";
import { logRange, usePalette } from "../device/common";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { currentAxis } from "../plots/theme";
import { describe, histogram, StatsTable, type StatsRow } from "../stats";
import { useStore, type ResultEntry } from "../state/store";
import { fmtDuration, fmtInt, isNum, siPrefix } from "../utils/format";
import { splitUnit } from "../circuit/summary";
import { runSchematic, type SchematicRunData } from "./run";
import { fmtSI } from "./si";
import { useSch } from "./store";
import { Segmented, Switch } from "./ui";

const AXES = ["voltage", "current", "charge", "state", "logic"] as const;
type AxisKind = (typeof AXES)[number];
export const axisOf = (s: Pick<Signal, "axis" | "unit">): AxisKind =>
  (s.axis as AxisKind) ?? (s.unit === "V" ? "voltage" : s.unit === "A" ? "current" : s.unit === "C" ? "charge" : s.unit === "1" ? "logic" : "state");

export function timeScale(tEnd: number): [number, string] {
  if (!(tEnd > 0)) return [1, "s"];
  const [f, p] = siPrefix(tEnd);
  return [1 / f, `${p}s`];
}

const num = (a: Arr | undefined) => (a ?? []).map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));

export function findSignal(res: CustomCircuitResult, key: string, run = 0): Signal | undefined {
  return res.runs[run]?.signals.find((s) => s.key === key);
}
export function findEnvelope(res: CustomCircuitResult, key: string): Envelope | undefined {
  return res.envelopes?.find((e) => e.key === key);
}

export function allSignalKeys(res: CustomCircuitResult): string[] {
  const keys = res.runs[0]?.signals.map((s) => s.key) ?? [];
  const order = (k: string) => (k.startsWith("V(") ? 0 : k.startsWith("I(") ? 1 : 2);
  return [...keys].sort((a, b) => order(a) - order(b) || a.localeCompare(b, "en", { numeric: true }));
}

/** Value of a signal at time t: run 0 (or the selected run) or the ensemble mean. */
export function signalAt(res: CustomCircuitResult, key: string, t: number, source: "run0" | "mean"): number | null {
  if (source === "mean") {
    const env = findEnvelope(res, key);
    if (env) return valueAt(env.t, env.mean, t);
  }
  const r = res.runs[0];
  const s = r?.signals.find((x) => x.key === key);
  return r && s ? valueAt(r.t, s.values, t) : null;
}

function traceLabel(key: string) {
  return key;
}

function WaveViewer({ res, entry, stale }: { res: CustomCircuitResult; entry: ResultEntry | undefined; stale: boolean }) {
  const t = useT();
  const c = usePalette();
  const traces = useSch((s) => s.traces);
  const logI = useSch((s) => s.logI);
  const showRuns = useSch((s) => s.showRuns);
  const showBand = useSch((s) => s.showBand);
  const cursorT = useSch((s) => s.cursorT);
  const set = useSch((s) => s.set);
  const sto = res.mode === "stochastic";
  const t0 = res.runs[0]?.t[0] ?? 0;
  const tEnd = res.runs[0]?.t[res.runs[0].t.length - 1] ?? 0;
  const [ts, tu] = timeScale(typeof tEnd === "number" ? tEnd : 0);
  const tsRef = useRef(ts);
  tsRef.current = ts;
  const keys = useMemo(() => allSignalKeys(res), [res]);
  const present = traces.filter((k) => keys.includes(k));
  const missing = traces.filter((k) => !keys.includes(k));

  const colorOf = useCallback((k: string) => c.categorical[Math.max(0, traces.indexOf(k)) % c.categorical.length], [c, traces]);

  const base = useMemo(() => {
    const meta = present.map((k) => ({ k, s: findSignal(res, k)! }));
    const axes = AXES.filter((a) => meta.some((m) => axisOf(m.s) === a));
    const n = Math.max(1, axes.length);
    const gap = 0.06;
    const h = (1 - gap * (n - 1)) / n;
    const data: Data[] = [];
    const layout: Partial<Layout> = {
      margin: { l: 70, r: 18, t: 30, b: 44 },
      hovermode: "x unified",
      xaxis: { title: { text: `${t("c.t")} (${tu})` }, anchor: (n > 1 ? `y${n}` : "y") as never, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: c.muted, spikedash: "dot" },
      showlegend: false,
    };
    axes.forEach((ax, i) => {
      const yName = i === 0 ? "yaxis" : `yaxis${i + 1}`;
      const yRef = i === 0 ? "y" : `y${i + 1}`;
      const top = 1 - i * (h + gap);
      const inAx = meta.filter((m) => axisOf(m.s) === ax);
      const unit = inAx[0]?.s.unit ?? "";
      const cur = ax === "current";
      const log = cur && logI;
      const vals = (a: Arr) => (log ? a.map((v) => (typeof v === "number" && v !== 0 ? Math.abs(v) : null)) : num(a));
      const allY: (number | null)[][] = [];
      for (const { k, s } of inAx) {
        const col = colorOf(k);
        const env = sto ? findEnvelope(res, k) : undefined;
        const hover = `%{y:.4~s}${unit === "1" ? "" : unit}`;
        if (sto && showRuns) {
          res.runs.forEach((r, ri) => {
            const sig = r.signals.find((x) => x.key === k);
            if (!sig) return;
            const y = vals(sig.values);
            allY.push(y);
            data.push({
              x: num(r.t).map((v) => (v == null ? null : v * ts)), y, type: "scatter", mode: "lines", xaxis: "x", yaxis: yRef as never,
              line: { color: col, width: env ? 0.8 : 1.2 }, opacity: env ? 0.35 : 0.8, name: `${k} · #${r.run}`, hoverinfo: ri === 0 && !env ? "y+name" : "skip",
              hovertemplate: ri === 0 && !env ? `${hover}<extra>${k}</extra>` : undefined,
            } as Data);
          });
        }
        if (sto && env) {
          const tt = num(env.t).map((v) => (v == null ? null : v * ts));
          const mean = env.mean.map((v) => (typeof v === "number" ? v : null));
          const sd = env.sd.map((v) => (typeof v === "number" ? v : 0));
          if (showBand) {
            const hi = mean.map((m, i) => (m == null ? null : log ? Math.abs(m) + sd[i] : m + sd[i]));
            const lo = mean.map((m, i) => (m == null ? null : log ? Math.max(Math.abs(m) - sd[i], Math.abs(m) * 1e-3) || null : m - sd[i]));
            data.push({ x: tt, y: lo, type: "scatter", mode: "lines", line: { width: 0, color: col }, xaxis: "x", yaxis: yRef as never, hoverinfo: "skip", showlegend: false } as Data);
            data.push({ x: tt, y: hi, type: "scatter", mode: "lines", line: { width: 0, color: col }, fill: "tonexty", fillcolor: hexA(col, 0.18), xaxis: "x", yaxis: yRef as never, hoverinfo: "skip", showlegend: false } as Data);
            allY.push(lo, hi);
          }
          const my = log ? mean.map((m) => (m == null || m === 0 ? null : Math.abs(m))) : mean;
          allY.push(my);
          data.push({ x: tt, y: my, type: "scatter", mode: "lines", line: { color: col, width: 2 }, xaxis: "x", yaxis: yRef as never, name: `${k} ${t("schematic.res.mean")}`, hovertemplate: `${hover}<extra>${k} (${t("schematic.res.mean")})</extra>` } as Data);
        } else if (!sto || !showRuns) {
          const y = vals(s.values);
          allY.push(y);
          data.push({
            x: num(res.runs[0].t).map((v) => (v == null ? null : v * ts)), y, type: "scatter", mode: "lines", xaxis: "x", yaxis: yRef as never,
            line: { color: col, width: 1.8, shape: ax === "logic" ? "hv" : "linear" }, name: k, hovertemplate: `${hover}<extra>${k}</extra>`,
          } as Data);
        }
      }
      (layout as Record<string, unknown>)[yName] = {
        domain: [Math.max(0, top - h), top],
        ...(cur ? currentAxis(log, log ? "|I| (A)" : "I (A)") : {}),
        ...(log ? { range: logRange(allY) } : {}),
        title: { text: cur ? (log ? "|I| (A)" : "I (A)") : `${t(`c.axis.${ax}` as StrKey)}${unit && unit !== "1" ? ` (${unit})` : ""}`, font: { size: 11 } },
        ...(ax === "charge" ? { tickformat: "~s", ticksuffix: "C", exponentformat: "SI" } : {}),
        ...(ax === "voltage" || ax === "state" ? { tickformat: "~s", exponentformat: "SI" } : {}),
      };
    });
    // latch events of run 0 as small markers along the top
    const evs = res.events.filter((e) => e.run === 0 && (e.kind === "latch_up" || e.kind === "latch_down")).slice(0, 80);
    layout.annotations = evs.map((e) => ({
      x: e.t * ts, xref: "x", y: 1, yref: "paper", yanchor: "bottom", showarrow: false, text: e.kind === "latch_up" ? "▲" : "▼",
      font: { size: 9, color: e.kind === "latch_up" ? c.lrs : c.hrs }, hovertext: `${e.cell ?? ""} ${e.kind} · t = ${fmtSI(e.t, "s", 4)}${isNum(e.v_d) ? ` · V_D = ${e.v_d.toFixed(3)} V` : ""}`,
    })) as Layout["annotations"];
    return { data, layout, n };
  }, [present, res, logI, showRuns, showBand, sto, ts, tu, c, t, colorOf]);

  const layout = useMemo(() => {
    if (cursorT == null) return base.layout;
    return {
      ...base.layout,
      shapes: [{ type: "line", xref: "x", yref: "paper", x0: cursorT * ts, x1: cursorT * ts, y0: 0, y1: 1, line: { color: c.text2, width: 1.5, dash: "solid" }, layer: "above", editable: true }],
    } as Partial<Layout>;
  }, [base.layout, cursorT, ts, c]);

  // Plotly events (click / drag the cursor line): the graph div is created by Panel → hook it from the DOM
  const hasPlot = present.length > 0;
  useEffect(() => {
    if (!hasPlot) return;
    const hook = () => {
      const gd = document.querySelector("[data-testid=panel-sch-waves] .js-plotly-plot") as (HTMLElement & { on?: (ev: string, fn: (d: never) => void) => void; __schHooked?: boolean }) | null;
      if (!gd || gd.__schHooked || typeof gd.on !== "function") return;
      gd.__schHooked = true;
      gd.on("plotly_relayout", (ev: Record<string, unknown>) => {
        const x = ev["shapes[0].x0"];
        if (typeof x === "number") useSch.getState().setCursor(x / tsRef.current);
      });
      gd.on("plotly_click", (ev: { points?: { x?: number }[] }) => {
        const x = ev.points?.[0]?.x;
        if (typeof x === "number") useSch.getState().setCursor(x / tsRef.current);
      });
    };
    hook();
    const id = setInterval(hook, 400);
    return () => clearInterval(id);
  }, [hasPlot]);

  const plot = present.length ? { data: base.data, layout, className: base.n > 2 ? "plot tall sch-plot" : "plot sch-plot" } : undefined;
  const t0n = typeof t0 === "number" ? t0 : 0;
  const tEn = typeof tEnd === "number" ? tEnd : 0;
  const cur = cursorT ?? tEn;
  const source = useSch((s) => s.annotateSource);
  return (
    <Panel
      id="sch-waves"
      wide
      title={t("schematic.res.title")}
      desc={t("schematic.res.desc")}
      topic="circuit-element"
      entry={entry}
      hasData
      stale={stale}
      csvName="schematic_waveforms"
      badges={
        <>
          <span className={`badge ${sto ? "sto" : "det"}`}>{sto ? t("schematic.res.mode.sto") : t("schematic.res.mode.det")}{sto ? ` · ${t("schematic.res.nRuns", { n: res.runs.length })}` : ""}</span>
        </>
      }
      plot={plot}
      empty={<span>{t("schematic.res.noTraces")}</span>}
      toolbar={
        <div className="trace-bar" data-testid="trace-bar">
          {present.map((k) => (
            <span key={k} className="trace-chip" style={{ "--chip": colorOf(k) } as React.CSSProperties} data-testid={`trace-${k}`}>
              <i aria-hidden />
              <span className="mono">{traceLabel(k)}</span>
              <button type="button" aria-label={t("schematic.res.remove", { k })} title={t("schematic.res.remove", { k })} onClick={() => useSch.getState().removeTrace(k)}>
                <IconX size={11} />
              </button>
            </span>
          ))}
          {missing.map((k) => (
            <span key={k} className="trace-chip missing" title={t("schematic.res.notSaved")}>
              <span className="mono">{k}</span>
              <button type="button" aria-label={t("schematic.res.remove", { k })} onClick={() => useSch.getState().removeTrace(k)}>
                <IconX size={11} />
              </button>
            </span>
          ))}
          <select className="select trace-add" value="" onChange={(e) => e.target.value && useSch.getState().addTrace(e.target.value)} aria-label={t("schematic.res.addTrace")} data-testid="trace-add">
            <option value="">{t("schematic.res.addTrace")}</option>
            {keys.filter((k) => !traces.includes(k)).map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          <span className="spacer" />
          <label className="tb-toggle">
            <Switch on={logI} onChange={(v) => set({ logI: v })} label={t("schematic.res.logI")} testId="toggle-logi" />
            {t("schematic.res.logI")}
          </label>
          {sto && (
            <>
              <label className="tb-toggle">
                <Switch on={showRuns} onChange={(v) => set({ showRuns: v })} label={t("schematic.res.runs")} />
                {t("schematic.res.runs")}
              </label>
              <label className="tb-toggle">
                <Switch on={showBand} onChange={(v) => set({ showBand: v })} label={t("schematic.res.band")} />
                {t("schematic.res.band")}
              </label>
            </>
          )}
        </div>
      }
    >
      {present.length === 0 && (
        <div className="panel-foot">
          <div className="sch-notraces" data-testid="sch-notraces">{t("schematic.res.noTraces")}</div>
        </div>
      )}
      {present.length > 0 && tEn > t0n && (
        <div className="cursor-bar" data-testid="cursor-bar">
          <span className="cursor-label">{t("schematic.res.cursor")}</span>
          <input
            type="range"
            min={t0n}
            max={tEn}
            step={(tEn - t0n) / 1000}
            value={cur}
            onChange={(e) => useSch.getState().setCursor(Number(e.target.value))}
            aria-label={t("schematic.res.cursor")}
            data-testid="cursor-slider"
          />
          <span className="cursor-time mono" data-testid="cursor-time">
            {(cur * ts).toPrecision(5)} {tu}
          </span>
          {sto && (
            <Segmented
              value={source}
              label={t("schematic.res.source")}
              onChange={(v) => set({ annotateSource: v })}
              options={[
                { v: "run0", label: t("schematic.res.run0") },
                { v: "mean", label: t("schematic.res.mean") },
              ]}
            />
          )}
          <span className="small muted cursor-hint">{t("schematic.res.cursorHint")}</span>
        </div>
      )}
      {present.length > 0 && (
        <div className="readout" data-testid="readout">
          {present.map((k) => {
            const v = signalAt(res, k, cur, sto ? source : "run0");
            const s = findSignal(res, k);
            return (
              <span key={k} className="readout-item">
                <i style={{ background: colorOf(k) }} aria-hidden />
                <span className="mono">{k}</span>
                <strong className="mono">{v == null ? "—" : fmtSI(v, s?.unit === "1" ? "" : s?.unit ?? "", 4)}</strong>
              </span>
            );
          })}
        </div>
      )}
      <div className="panel-foot">
        <details className="signs">
          <summary>{t("schematic.res.signs")}</summary>
          <p>{t("schematic.res.signsBody")}</p>
        </details>
      </div>
    </Panel>
  );
}

function hexA(hex: string, a: number): string {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function SummaryCards({ res, stale }: { res: CustomCircuitResult; stale: boolean }) {
  const t = useT();
  if (!res.summary.length) return null;
  return (
    <div className={`kpis wrap${stale ? " stale" : ""}`} data-testid="sch-summary">
      {res.summary.map((s, i) => {
        const v = splitUnit(s.value, s.unit);
        const sp = isNum(s.spread) ? splitUnit(s.spread, s.unit, s.unit === "V" ? "mV" : undefined) : null;
        return <Kpi key={s.key} id={`sch-${s.key}`} label={t.l(s.label)} value={v.value} unit={v.unit} sub={sp ? `± ${sp.value} ${sp.unit}` : undefined} color={i === 0 ? "var(--accent)" : "var(--border-strong)"} />;
      })}
    </div>
  );
}

interface CellStats {
  cell: string;
  lu: number[];
  ld: number[];
  firstT: (number | null)[];
  firstV: (number | null)[];
}

function cellStats(res: CustomCircuitResult, nRuns: number, cells: string[]): CellStats[] {
  return cells.map((cell) => {
    const lu = new Array(nRuns).fill(0);
    const ld = new Array(nRuns).fill(0);
    const firstT: (number | null)[] = new Array(nRuns).fill(null);
    const firstV: (number | null)[] = new Array(nRuns).fill(null);
    for (const e of res.events) {
      if ((e.cell ?? "") !== cell || e.run < 0 || e.run >= nRuns) continue;
      if (e.kind === "latch_up") {
        lu[e.run]++;
        if (firstT[e.run] == null) {
          firstT[e.run] = e.t;
          firstV[e.run] = isNum(e.v_d) ? e.v_d : null;
        }
      } else if (e.kind === "latch_down") ld[e.run]++;
    }
    return { cell, lu, ld, firstT, firstV };
  });
}

function EventsTable({ res, nRuns, cells, stale }: { res: CustomCircuitResult; nRuns: number; cells: string[]; stale: boolean }) {
  const t = useT();
  const rows = useMemo(() => cellStats(res, nRuns, cells), [res, nRuns, cells]);
  const sto = res.mode === "stochastic";
  if (!cells.length) return null;
  const ms = (xs: number[]) => {
    const d = describe(xs);
    return sto ? `${d.mean == null ? "—" : d.mean.toFixed(2)} ± ${d.sd == null ? "—" : d.sd.toFixed(2)}` : fmtInt(xs[0]);
  };
  const tstat = (xs: (number | null)[], unit: string) => {
    const d = describe(xs);
    if (d.n === 0) return "—";
    if (!sto) return unit === "V" ? `${(d.mean ?? 0).toFixed(3)} V` : fmtSI(d.mean, unit, 4);
    return `${unit === "V" ? `${(d.mean ?? 0).toFixed(3)} V` : fmtSI(d.mean, unit, 4)} ± ${d.sd == null ? "—" : unit === "V" ? `${(d.sd * 1e3).toFixed(1)} mV` : fmtSI(d.sd, unit, 3)}`;
  };
  return (
    <Panel id="sch-events" wide={sto || !res.trajectory} title={t("schematic.res.events")} desc={t("schematic.res.eventsDesc")} topic="stochastic-events" hasData stale={stale}>
      <div className="panel-foot">
        <div className="table-wrap">
          <table className="table compact" data-testid="sch-events-table">
            <thead>
              <tr>
                <th>{t("schematic.res.cell")}</th>
                <th className="num">
                  {t("schematic.res.lu")}
                  {sto ? ` (${t("schematic.res.perRun")})` : ""}
                </th>
                <th className="num">
                  {t("schematic.res.ld")}
                  {sto ? ` (${t("schematic.res.perRun")})` : ""}
                </th>
                {sto && <th className="num">{t("schematic.res.pAny")}</th>}
                <th className="num">{t("schematic.res.firstLu")}</th>
                <th className="num">{t("schematic.res.vdLu")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.cell}>
                  <td className="mono">{r.cell}</td>
                  <td className="num">{ms(r.lu)}</td>
                  <td className="num">{ms(r.ld)}</td>
                  {sto && <td className="num">{((r.lu.filter((x) => x > 0).length / Math.max(1, nRuns)) * 100).toFixed(0)} %</td>}
                  <td className="num">{tstat(r.firstT, "s")}</td>
                  <td className="num">{tstat(r.firstV, "V")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  );
}

function DistributionsPanel({ res, stale }: { res: CustomCircuitResult; stale: boolean }) {
  const t = useT();
  const c = usePalette();
  const dists = (res.distributions ?? []).filter((d) => d.values.some((v) => isNum(v)));
  const [k, setK] = useState(0);
  const d = dists[Math.min(k, dists.length - 1)];
  const plot = useMemo(() => {
    if (!d) return undefined;
    const [f, p] = d.unit && d.unit !== "1" && d.unit !== "V" ? siPrefix(Math.max(...d.values.filter(isNum).map(Math.abs), 0) || 1) : [1, ""];
    const xs = d.values.map((v) => (isNum(v) ? v / f : null));
    const n = xs.filter(isNum).length;
    // Freedman–Diaconis, but never fewer than ~√n bins (FD collapses to one bin when the IQR is 0)
    const h = histogram(xs, "fd", { minBins: Math.min(40, Math.max(8, Math.ceil(Math.sqrt(n) * 1.5))), maxBins: 60 });
    const centers = h.counts.map((_, i) => (h.edges[i] + h.edges[i + 1]) / 2);
    return {
      data: [{ x: centers, y: h.counts, type: "bar", width: h.width * 0.92, marker: { color: c.sto, line: { color: c.surface, width: 1 } }, opacity: 0.85, name: t.l(d.label), hovertemplate: `%{x:.4g} ${p}${d.unit === "1" ? "" : d.unit}: %{y}<extra></extra>` } as Data],
      layout: { xaxis: { title: { text: `${t.l(d.label)}${d.unit && d.unit !== "1" ? ` (${p}${d.unit})` : ""}` } }, yaxis: { title: { text: "counts" } }, bargap: 0.02, showlegend: false } as Partial<Layout>,
      className: "plot short",
    };
  }, [d, c, t]);
  // per-run metrics first; the values of every signal at t_stop ("end:<key>") only on request
  const [withEnd, setWithEnd] = useState(false);
  const nEnd = dists.filter((x) => x.key.startsWith("end:")).length;
  const rows: StatsRow[] = useMemo(
    () => dists.filter((x) => withEnd || !x.key.startsWith("end:")).map((x) => ({ key: x.key, label: t.l(x.label), labelText: t.l(x.label), unit: x.unit === "1" ? "" : x.unit, values: x.values })),
    [dists, t, withEnd],
  );
  if (!dists.length) return null;
  return (
    <Panel
      id="sch-dist"
      wide
      title={t("schematic.res.dist")}
      desc={t("schematic.res.distDesc")}
      topic="stochastic-events"
      hasData
      stale={stale}
      csvName="schematic_distribution"
      plot={plot}
      toolbar={
        dists.length > 1 ? (
          <select className="select" style={{ width: 260, height: 26 }} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label={t("schematic.res.dist")} data-testid="sch-dist-select">
            {dists.map((x, i) => (
              <option key={x.key} value={i}>
                {t.l(x.label)}
              </option>
            ))}
          </select>
        ) : undefined
      }
    >
      <div className="panel-foot">
        <StatsTable
          rows={rows}
          csvName="schematic_statistics"
          testId="sch-stats-table"
          caption={
            nEnd > 0 ? (
              <label className="tb-toggle">
                <Switch on={withEnd} onChange={setWithEnd} label={t("schematic.res.withEnd", { n: nEnd })} testId="toggle-end-stats" />
                {t("schematic.res.withEnd", { n: nEnd })}
              </label>
            ) : undefined
          }
        />
      </div>
    </Panel>
  );
}

function TrajectoryPanel({ res, stale }: { res: CustomCircuitResult; stale: boolean }) {
  const t = useT();
  const c = usePalette();
  const tr = res.trajectory as { vd: Arr; id: Arr; cell?: string } | undefined;
  const plot = useMemo(() => {
    if (!tr || !tr.vd?.length) return undefined;
    const y = tr.id.map((v) => (typeof v === "number" && v !== 0 ? Math.abs(v) : null));
    const cell = tr.cell ?? "X1";
    return {
      data: [{ x: num(tr.vd), y, type: "scatter", mode: "lines", line: { color: c.sto, width: 1.6 }, name: cell, hovertemplate: `V<sub>DS</sub> = %{x:.3f} V<br>|I<sub>D</sub>| = %{y:.3~s}A<extra>${cell}</extra>` } as Data],
      layout: { xaxis: { title: { text: `V<sub>DS</sub> (V) · ${cell}` } }, yaxis: { ...currentAxis(true, "|I<sub>D</sub>| (A)"), range: logRange([y]) }, showlegend: false } as Partial<Layout>,
      className: "plot short",
    };
  }, [tr, c]);
  if (!plot) return null;
  return <Panel id="sch-traj" title={t("schematic.res.traj")} desc={t("schematic.res.trajDesc")} topic="circuit-element" hasData stale={stale} csvName="schematic_trajectory" plot={plot} />;
}

export function Results({ entry, stale, cells }: { entry: ResultEntry | undefined; stale: boolean; cells: string[] }) {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const data = entry?.data as SchematicRunData | undefined;
  const res = data?.result;
  if (!res) {
    return (
      <Panel
        id="sch-waves"
        wide
        title={t("schematic.res.title")}
        desc={t("schematic.res.desc")}
        topic="circuit-element"
        entry={entry}
        hasData={false}
        plot={{ data: [], layout: {} }}
        empty={
          <>
            <span>{t("schematic.res.empty")}</span>
            <button type="button" className="btn primary sm" onClick={() => void runSchematic()} data-testid="sch-empty-run">
              ▶ {t("run.circuit")}
            </button>
          </>
        }
      />
    );
  }
  const nRuns = res.mode === "stochastic" ? Math.max(res.runs.length, ...res.events.map((e) => e.run + 1), data?.request.stochastic?.n_runs ?? 0) : 1;
  const ss = res.solver_stats;
  return (
    <>
      <SummaryCards res={res} stale={stale} />
      {data?.demoFallback && (
        <div className="callout warn" role="status" data-testid="sch-demo-fallback">
          {t("schematic.run.demoFallback")}
        </div>
      )}
      <WaveViewer res={res} entry={entry} stale={stale} />
      <div className="grid">
        <EventsTable res={res} nRuns={nRuns} cells={cells} stale={stale} />
        <TrajectoryPanel res={res} stale={stale} />
        {res.mode === "stochastic" && <DistributionsPanel res={res} stale={stale} />}
      </div>
      <div className="sch-solver small muted mono" data-testid="sch-solver">
        {t("schematic.res.solver", { steps: fmtInt(ss?.steps), rej: fmtInt(ss?.rejected), newton: fmtInt(ss?.newton_iters), time: fmtDuration(ss?.runtime_s ?? res.runtime_s) })}
        {mode !== res.mode && <span className="badge stale">{res.mode}</span>}
      </div>
      {res.warnings?.length > 0 && <Notices items={res.warnings} />}
    </>
  );
}

/** Annotation values at the cursor: node voltages by net name + every signal by key. */
export function annotationsAt(res: CustomCircuitResult, t: number, source: "run0" | "mean"): { v: Map<string, number>; sig: Map<string, number>; t: number } {
  const v = new Map<string, number>();
  const sig = new Map<string, number>();
  for (const s of res.runs[0]?.signals ?? []) {
    const val = signalAt(res, s.key, t, source);
    if (val == null) continue;
    sig.set(s.key, val);
    const p = parseProbe(s.key);
    if (p?.type === "V") v.set(p.node, val);
  }
  // STL drain current from I(X.s) when only the source terminal was saved
  for (const [k, val] of [...sig]) {
    const p = parseProbe(k);
    if (p?.type === "I" && p.terminal === "s" && !sig.has(iKey(p.el, "d"))) sig.set(iKey(p.el, "d"), -val);
  }
  return { v, sig, t };
}
