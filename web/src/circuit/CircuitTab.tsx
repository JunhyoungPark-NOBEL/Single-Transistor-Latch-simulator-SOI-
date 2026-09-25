// Circuit tab: bench selector cards, and a generic rendering of CircuitResult (§4): schematic, waveforms
// grouped by axis on shared-x stacked subplots, I–V trajectory over the steady-state branches, summary
// cards, distributions, sweeps, events, solver stats and warnings.
import type { Data, Layout, Shape } from "plotly.js";
import { lazy, Suspense, useMemo, useState, type ReactNode } from "react";
import type { BenchId, BranchesResult, CircuitResult, Signal } from "../api/types";
import { finite } from "../api/guards";
import { Notices, Panel } from "../components/Panel";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { BENCH_ORDER, BENCHES } from "../params/benches";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { useStore } from "../state/store";
import { fmtDuration, fmtInt, fmtSig, isNum, siPrefix } from "../utils/format";
import { splitUnit } from "./summary";
import { AXES, circuitAxisTitle, siTicks, stackClass, sweepAxisTitle, type AxisKind } from "./axes";
import { subs, withUnit } from "../plots/labels";

export { splitUnit };
import { circuitPayload } from "../utils/payload";
import { Kpi } from "../device/KpiStrip";
import { isStale, logRange, nums, pos, useCurrentKey, useEntry, usePalette } from "../device/common";
import { Check, Seg } from "../device/DetPanels";
import { BenchIcon } from "./BenchIcons";
import { Schematic } from "./Schematic";
import { useCircuitView, type CircuitView } from "./view";
import "./circuit.css";

const axisOf = (s: Signal): AxisKind =>
  (s.axis as AxisKind) ?? (s.unit === "V" ? "voltage" : s.unit === "A" ? "current" : s.unit === "C" ? "charge" : s.unit === "1" ? "logic" : "state");

/** Default netlists shown before the first run (element names as server/compute/circuit/benches.py). */
function defaultSchematic(bench: BenchId, vg: number): CircuitResult["schematic"] {
  const cell = (k: number, d: string, g: string) => [
    { kind: "STL", name: `X${k}`, nodes: [d, g, "0"] },
    { kind: "V", name: `VG${k}`, nodes: [g, "0"], value: `${vg} V (DC)` },
  ];
  if (bench === "pbit")
    return {
      nodes: ["clk", "d", "g", "0"],
      elements: [
        { kind: "V", name: "Vclk", nodes: ["clk", "0"], value: "clock" },
        { kind: "R", name: "RL", nodes: ["clk", "d"], value: "100 kΩ" },
        { kind: "C", name: "Cd", nodes: ["d", "0"], value: "2 fF" },
        ...cell(1, "d", "g"),
        { kind: "CMP", name: "CMP", nodes: ["d"], value: "bit = [v_D < v_th]" },
      ],
    };
  if (bench === "coupled")
    return {
      nodes: ["src", "d1", "d2", "g1", "g2", "0"],
      elements: [
        { kind: "V", name: "Vsrc", nodes: ["src", "0"], value: "ramp / pulses" },
        { kind: "R", name: "Rs1", nodes: ["src", "d1"], value: "100 kΩ" },
        { kind: "R", name: "Rs2", nodes: ["src", "d2"], value: "100 kΩ" },
        { kind: "R", name: "Rc", nodes: ["d1", "d2"], value: "1 MΩ" },
        { kind: "C", name: "Cd1", nodes: ["d1", "0"], value: "2 fF" },
        { kind: "C", name: "Cd2", nodes: ["d2", "0"], value: "2 fF" },
        ...cell(1, "d1", "g1"),
        ...cell(2, "d2", "g2"),
      ],
    };
  return {
    nodes: ["src", "d", "g", "0"],
    elements: [
      { kind: "V", name: "Vsrc", nodes: ["src", "0"], value: bench === "pulse" ? "pulses" : "triangle" },
      { kind: "R", name: "Rs", nodes: ["src", "d"], value: "1 kΩ" },
      { kind: "C", name: "Cd", nodes: ["d", "0"], value: "2 fF" },
      ...cell(1, "d", "g"),
    ],
  };
}

function BenchPicker() {
  const t = useT();
  const bench = useStore((s) => s.params.circuit.bench);
  const setParam = useStore((s) => s.setParam);
  return (
    <div className="bench-cards" role="radiogroup" aria-label={t("c.bench")} data-testid="bench-picker">
      {BENCH_ORDER.map((id) => (
        <button key={id} type="button" role="radio" aria-checked={bench === id} className="bench-card" onClick={() => setParam(["circuit", "bench"], id)} data-testid={`bench-${id}`}>
          <BenchIcon bench={id} />
          <span className="bench-title">{t(BENCHES[id].title)}</span>
          <span className="bench-desc">{t(BENCHES[id].desc)}</span>
        </button>
      ))}
    </div>
  );
}

function timeScale(t: (number | null)[]): [number, string] {
  const m = Math.max(...finite(t).map(Math.abs), 0);
  if (!m) return [1, "s"];
  const [f, p] = siPrefix(m);
  return [1 / f, `${p}s`];
}

function WaveformPanel({ res, entry, currentKey }: { res: CircuitResult | undefined; entry: ReturnType<typeof useEntry>["entry"]; currentKey: string }) {
  const t = useT();
  const c = usePalette();
  const [run, setRun] = useState<string>("0");
  const [logI, setLogI] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  const plot = useMemo(() => {
    if (!res || !res.runs.length) return undefined;
    const runs = run === "all" ? res.runs : res.runs.filter((r) => String(r.run) === run).slice(0, 1);
    const present = AXES.filter((a) => res.runs.some((r) => r.signals.some((s) => axisOf(s) === a)));
    const n = present.length;
    const gap = 0.05;
    const h = (1 - gap * (n - 1)) / n;
    const [ts, tu] = timeScale(res.runs[0].t);
    const traces: Data[] = [];
    let k = 0; // colour index shared by all signals (identity is kept across subplots)
    const layout: Partial<Layout> = { margin: { l: 70, r: 16, t: 40, b: 46 }, xaxis: { title: { text: t("axis.time", { u: tu }) }, anchor: `y${n > 1 ? n : ""}` as never } };
    present.forEach((ax, i) => {
      const yName = i === 0 ? "yaxis" : `yaxis${i + 1}`;
      const top = 1 - i * (h + gap);
      const unit = res.runs[0].signals.find((s) => axisOf(s) === ax)?.unit ?? "";
      const cur = ax === "current";
      (layout as Record<string, unknown>)[yName] = {
        domain: [Math.max(0, top - h), top],
        ...(cur ? { ...currentAxis(logI, ""), ...(logI ? { range: logRange(runs.flatMap((r) => r.signals.filter((x) => axisOf(x) === "current").map((x) => pos(x.values.map((v) => (v == null ? null : Math.abs(v))))))) } : {}) } : ax === "logic" ? {} : siTicks(unit)),
        title: { text: circuitAxisTitle(t, ax, unit, logI), font: { size: 11 } },
        ...(ax === "logic" ? { range: [-0.2, 1.2], dtick: 1 } : {}),
      };
      for (const r of runs) {
        for (const s of r.signals.filter((x) => axisOf(x) === ax)) {
          const color = c.categorical[k % c.categorical.length];
          k++;
          traces.push({
            x: nums(r.t).map((v) => (v == null ? null : v * ts)),
            y: cur && logI ? pos(s.values.map((v) => (v == null ? null : Math.abs(v)))) : nums(s.values),
            type: "scatter", mode: "lines", yaxis: i === 0 ? "y" : (`y${i + 1}` as never), xaxis: "x",
            name: `${subs(t.l(s.label))}${runs.length > 1 ? ` · #${r.run}` : ""}`,
            legendgroup: s.key, showlegend: runs.length === 1 || r === runs[0],
            line: { color: runs.length > 1 ? c.categorical[r.run % c.categorical.length] : color, width: runs.length > 1 ? 1 : 1.8, shape: ax === "logic" ? "hv" : "linear" },
            opacity: runs.length > 1 ? 0.7 : 1,
            hovertemplate: `t = %{x:.4g} ${tu}<br>%{y:.4~s}${s.unit === "1" ? "" : s.unit}<extra>${subs(t.l(s.label))}</extra>`,
          });
        }
      }
    });
    if (showEvents) {
      const evs = res.events.filter((e) => run === "all" || String(e.run) === run).slice(0, 200);
      layout.shapes = evs.map((e) => ({
        type: "line", xref: "x", yref: "paper", x0: e.t * ts, x1: e.t * ts, y0: 0, y1: 1,
        line: { color: e.kind === "latch_up" ? c.lrs : e.kind === "latch_down" ? c.hrs : c.muted, width: 1, dash: "dot" },
      })) as Partial<Shape>[];
    }
    return { data: traces, layout, className: stackClass(n) };
  }, [res, run, logI, showEvents, c, t]);
  return (
    <Panel
      id="waves"
      wide
      currentKey={currentKey}
      title={t("c.waves")}
      desc={t("c.waves.desc")}
      topic="circuit-element"
      entry={entry}
      hasData={!!res}
      csvName="waveforms"
      plot={plot}
      toolbar={
        <>
          {res && res.runs.length > 1 && (
            <>
              <label className="tb-label" htmlFor="run-sel">{t("c.run")}</label>
              <select id="run-sel" className="select" style={{ width: 130, height: 26 }} value={run} onChange={(e) => setRun(e.target.value)}>
                {res.runs.map((r) => (
                  <option key={r.run} value={String(r.run)}>#{r.run}</option>
                ))}
                <option value="all">{t("c.overlay")}</option>
              </select>
            </>
          )}
          <Seg label="I" value={logI ? "log" : "lin"} onChange={(v) => setLogI(v === "log")} options={[{ v: "log", label: `I ${t("log")}` }, { v: "lin", label: `I ${t("lin")}` }]} />
          <Check checked={showEvents} onChange={setShowEvents} label={t("c.events")} />
        </>
      }
    />
  );
}

function TrajectoryPanel({ res, entry, currentKey }: { res: CircuitResult | undefined; entry: ReturnType<typeof useEntry>["entry"]; currentKey: string }) {
  const t = useT();
  const c = usePalette();
  const { data: br } = useEntry<BranchesResult>("circuit_branches");
  const plot = useMemo(() => {
    if (!res?.trajectory) return undefined;
    const traces: Data[] = [];
    if (br) {
      traces.push(
        { x: nums(br.HRS.vd), y: pos(br.HRS.id), type: "scatter", mode: "lines", name: t("iv.hrs"), line: { color: c.hrs, width: 2 }, hovertemplate: `${HOVER_IV}<extra>HRS</extra>` },
        { x: nums(br.unstable.vd), y: pos(br.unstable.id), type: "scatter", mode: "lines", name: t("iv.unstable"), line: { color: c.unstable, width: 1.4, dash: "dash" }, hoverinfo: "skip" },
        { x: nums(br.LRS.vd), y: pos(br.LRS.id), type: "scatter", mode: "lines", name: t("iv.lrs"), line: { color: c.lrs, width: 2 }, hovertemplate: `${HOVER_IV}<extra>LRS</extra>` },
      );
    }
    traces.push({ x: nums(res.trajectory.vd), y: pos(res.trajectory.id), type: "scatter", mode: "lines", name: t("axis.leg.run0"), line: { color: c.categorical[6], width: 1.6 }, hovertemplate: `${HOVER_IV}<extra>${t("axis.leg.run0")}</extra>` });
    return { data: traces, layout: { xaxis: { title: { text: t("axis.vd") } }, yaxis: { ...currentAxis(true, t("axis.idAbs")), range: logRange(traces.map((tr) => (tr as { y?: (number | null)[] }).y)) } } as Partial<Layout> };
  }, [res, br, c, t]);
  return <Panel id="trajectory" title={t("c.traj")} desc={t("c.traj.desc")} topic="circuit-element" entry={entry} hasData={!!plot} currentKey={currentKey} csvName="trajectory" plot={plot ?? { data: [], layout: {} }} />;
}

function DistributionsPanel({ res, stale }: { res: CircuitResult; stale: boolean }) {
  const t = useT();
  const c = usePalette();
  const [k, setK] = useState(0);
  const dists = (res.distributions ?? []).filter((d) => finite(d.values).length >= 3);
  const d = dists[Math.min(k, dists.length - 1)];
  const plot = useMemo(() => {
    if (!d) return undefined;
    return {
      data: [{ x: finite(d.values), type: "histogram", marker: { color: c.sto, line: { color: c.surface, width: 1 } }, opacity: 0.85, name: subs(t.l(d.label)), nbinsx: 30, hovertemplate: `%{x} ${d.unit === "1" ? "" : d.unit}<br>n = %{y}<extra></extra>` } as Data],
      layout: { xaxis: { title: { text: withUnit(subs(t.l(d.label)), d.unit) } }, yaxis: { title: { text: t("axis.count") } }, bargap: 0.02 } as Partial<Layout>,
    };
  }, [d, c, t]);
  if (!dists.length) return null;
  return (
    <Panel id="c-dist" title={t("c.dist")} topic="stochastic-events" hasData={!!plot} stale={stale} csvName="circuit_distribution" plot={plot}
      toolbar={dists.length > 1 ? (
        <select className="select" style={{ width: 200, height: 26 }} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label={t("c.dist")}>
          {dists.map((x, i) => <option key={x.key} value={i}>{t.l(x.label)}</option>)}
        </select>
      ) : undefined}
    />
  );
}

function SweepsPanel({ res, stale }: { res: CircuitResult; stale: boolean }) {
  const t = useT();
  const c = usePalette();
  const [k, setK] = useState(0);
  const sw = res.sweeps ?? [];
  const s = sw[Math.min(k, sw.length - 1)];
  const plot = useMemo(() => {
    if (!s) return undefined;
    return {
      data: [{
        x: nums(s.x), y: nums(s.y), type: "scatter", mode: "lines+markers", name: t.l(s.label), line: { color: c.sto, width: 2 }, marker: { size: 7 },
        error_y: s.y_err ? { type: "data", array: nums(s.y_err) as number[], visible: true, color: c.sto, thickness: 1.2, width: 4 } : undefined,
        hovertemplate: `${subs(s.x_label)} = %{x:.4g} ${s.x_unit === "1" ? "" : s.x_unit}<br>${subs(s.y_label)} = %{y:.4g} ${s.y_unit === "1" ? "" : s.y_unit}<extra></extra>`,
      } as Data],
      layout: { xaxis: { title: { text: sweepAxisTitle(t, s.x_label, s.x_unit) } }, yaxis: { title: { text: sweepAxisTitle(t, s.y_label, s.y_unit) } } } as Partial<Layout>,
    };
  }, [s, c, t]);
  if (!sw.length) return null;
  return (
    <Panel id="c-sweeps" title={s ? t.l(s.label) : t("c.sweeps")} topic="circuit-element" hasData={!!plot} stale={stale} csvName="circuit_sweep" plot={plot}
      toolbar={sw.length > 1 ? (
        <select className="select" style={{ width: 220, height: 26 }} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label={t("c.sweeps")}>
          {sw.map((x, i) => <option key={x.key} value={i}>{t.l(x.label)}</option>)}
        </select>
      ) : undefined}
    />
  );
}

function EventsPanel({ res, stale }: { res: CircuitResult; stale: boolean }) {
  const t = useT();
  const [ts, tu] = timeScale(res.runs[0]?.t ?? []);
  const ss = res.solver_stats;
  return (
    <Panel id="c-events" title={`${t("c.events")} · ${t("c.solver")}`} topic="numerics" hasData stale={stale}>
      <div className="panel-foot">
        <div className="row small mono" style={{ flexWrap: "wrap", gap: 12 }} data-testid="solver-stats">
          <span>{t("c.steps")}: <strong>{fmtInt(ss?.steps)}</strong></span>
          <span>{t("c.rejected")}: <strong>{fmtInt(ss?.rejected)}</strong></span>
          <span>{t("c.newton")}: <strong>{fmtInt(ss?.newton_iters)}</strong></span>
          <span>{t("kpi.runtime")}: <strong>{fmtDuration(ss?.runtime_s ?? res.runtime_s)}</strong></span>
        </div>
        {res.events.length === 0 ? (
          <div className="small muted">{t("c.noevents")}</div>
        ) : (
          <div className="table-wrap events">
            <table className="table">
              <thead>
                <tr>
                  <th className="num">{t("c.run")}</th>
                  <th>{t("c.kind")}</th>
                  <th className="num">t ({tu})</th>
                  <th className="num">{t("c.value")}</th>
                  <th className="num">v_src (V)</th>
                  <th className="num">v_D (V)</th>
                </tr>
              </thead>
              <tbody>
                {res.events.slice(0, 300).map((e, i) => (
                  <tr key={i}>
                    <td className="num">{e.run}</td>
                    <td><span className={`badge ${e.kind === "latch_up" ? "err" : e.kind === "latch_down" ? "det" : ""}`}>{e.kind}</span></td>
                    <td className="num">{(e.t * ts).toPrecision(5)}</td>
                    <td className="num">{isNum(e.value) ? e.value.toPrecision(4) : "—"}</td>
                    <td className="num">{isNum(e.v_src) ? e.v_src.toFixed(3) : "—"}</td>
                    <td className="num">{isNum(e.v_d) ? e.v_d.toFixed(3) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      {res.warnings?.length > 0 && <Notices items={res.warnings} />}
    </Panel>
  );
}

const SchematicView = lazy(() => import("../schematic/SchematicView"));

function ViewSwitch() {
  const t = useT();
  const view = useCircuitView((s) => s.view);
  const setView = useCircuitView((s) => s.setView);
  const opts: { v: CircuitView; title: StrKey; sub: StrKey; icon: ReactNode }[] = [
    {
      v: "schematic", title: "schematic.view.editor", sub: "schematic.view.editorSub",
      icon: <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M3 6h5l2-3 3 6 2-3h6" /><path d="M6 6v6m0 4v5M3 12h6M3 16h6M18 6v15" /><path d="M14 21h8" /></svg>,
    },
    {
      v: "benches", title: "schematic.view.benches", sub: "schematic.view.benchesSub",
      icon: <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>,
    },
  ];
  return (
    <div className="view-switch" role="tablist" aria-label={t("schematic.view.aria")} data-testid="circuit-view-switch">
      {opts.map((o) => (
        <button key={o.v} type="button" role="tab" aria-selected={view === o.v} className="view-tab" onClick={() => setView(o.v)} data-testid={`circuit-view-${o.v}`}>
          {o.icon}
          <span className="view-tab-text">
            <strong>{t(o.title)}</strong>
            <span>{t(o.sub)}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

export function CircuitTab() {
  const view = useCircuitView((s) => s.view);
  const t = useT();
  return (
    <>
      <ViewSwitch />
      {view === "schematic" ? (
        <Suspense fallback={<div className="skeleton-plot" aria-label={t("schematic.loading")} style={{ height: 540 }} />}>
          <SchematicView />
        </Suspense>
      ) : (
        <BenchesView />
      )}
    </>
  );
}

function BenchesView() {
  const t = useT();
  const params = useStore((s) => s.params);
  const mode = useStore((s) => s.mode);
  const { entry, data: res } = useEntry<CircuitResult>("circuit");
  const key = useCurrentKey("circuit", useMemo(() => circuitPayload(params, mode), [params, mode]));
  const bench = params.circuit.bench;
  const sch = res && res.bench === bench ? res.schematic : defaultSchematic(bench, params.device.vg);
  const running = entry?.status === "running" || entry?.status === "queued";
  const stale = isStale(entry, key);
  return (
    <>
      <BenchPicker />
      {res && res.bench === bench && res.summary.length > 0 && (
        <div className={`kpis wrap${stale ? " stale" : ""}`} data-testid="circuit-summary" title={stale ? t("stale") : undefined}>
          {res.summary.map((s, i) => {
            const v = splitUnit(s.value, s.unit);
            const sp = isNum(s.spread) ? splitUnit(s.spread, s.unit, s.unit === "V" ? "mV" : undefined) : null;
            return (
              <Kpi key={s.key} id={`c-${s.key}`} label={t.l(s.label)} value={v.value} unit={v.unit} sub={sp ? `± ${sp.value} ${sp.unit}` : undefined} color={i === 0 ? "var(--accent)" : "var(--border-strong)"} />
            );
          })}
        </div>
      )}
      <div className="grid">
        <Panel id="schematic" wide title={t("c.schematic")} desc={t("c.schematic.desc")} topic="circuit-element" hasData entry={running ? entry : undefined} currentKey={key}
          badges={res && res.bench === bench ? <span className="badge">{res.mode}</span> : <span className="badge">{t(BENCHES[bench].title)}</span>}>
          <div className="panel-foot schematic">
            <Schematic nodes={sch.nodes} elements={sch.elements} title={t(BENCHES[bench].title)} />
          </div>
          {res && res.bench === bench && res.bench_params && (
            <div className="panel-foot">
              <div className="chips" data-testid="resolved-params" aria-label="resolved bench parameters">
                {Object.entries(res.bench_params)
                  .filter(([, v]) => typeof v === "number" || typeof v === "string")
                  .map(([k, v]) => (
                    <span key={k} className="chip" title={k}>
                      {k} = {typeof v === "number" ? (Number.isInteger(v) ? String(v) : fmtSig(v, 4)) : String(v)}
                    </span>
                  ))}
              </div>
            </div>
          )}
        </Panel>
        <WaveformPanel res={res} entry={entry} currentKey={key} />
        <TrajectoryPanel res={res} entry={entry} currentKey={key} />
        {res && <DistributionsPanel res={res} stale={stale} />}
        {res && <SweepsPanel res={res} stale={stale} />}
        {res && <EventsPanel res={res} stale={stale} />}
      </div>
    </>
  );
}
