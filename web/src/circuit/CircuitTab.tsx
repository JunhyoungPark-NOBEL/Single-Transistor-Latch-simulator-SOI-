// Circuit tab: [회로도 편집기 | 빠른 벤치] sub-view switch and the quick benches, a generic rendering of
// CircuitResult (§4). 간단히 layout: compact bench chips, a one-line bench description, the result summary (≤ 4
// cells) next to a schematic thumbnail, the waveform hero (source V, drain V, drain I in two subplots; the
// other signals from "신호 ▾") with the solver statistics as its footnote, and one tabbed analysis card
// (I–V 궤적 | 분포 | 스윕 | 사건 표). 모두 보기 layout: the bench cards, every metric, the full-size schematic
// and every panel in a grid, as before.
import type { Data, Layout, Shape } from "plotly.js";
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { BenchId, BranchesResult, CircuitResult, Signal } from "../api/types";
import { finite } from "../api/guards";
import { LayoutToggle } from "../components/LayoutToggle";
import { entryStatus, FocusLayout, type MoreTab } from "../components/MoreCard";
import { Notices, Panel, type PanelMenuItem } from "../components/Panel";
import { IconCheck, IconChevron } from "../components/icons";
import { Modal } from "../devices/Modal";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { BENCH_ORDER, BENCHES } from "../params/benches";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { useIsAll } from "../state/layout";
import { useStore, type ResultEntry } from "../state/store";
import { fmtDuration, fmtInt, fmtSig, isNum, siPrefix } from "../utils/format";
import { splitUnit } from "./summary";
import { AXES, circuitAxisTitle, siTicks, sweepAxisTitle, stackClass, type AxisKind } from "./axes";
import { subs, withUnit } from "../plots/labels";
import { circuitPayload } from "../utils/payload";
import { isStale, logRange, nums, pos, useCurrentKey, useEntry, usePalette } from "../device/common";
import { Segmented } from "../schematic/ui";
import { BenchGlyph, BenchIcon } from "./BenchIcons";
import { Schematic } from "./Schematic";
import { SummaryStrip } from "./SummaryStrip";
import { useCircuitView, type CircuitView } from "./view";
import "./circuit.css";

export { splitUnit };

const axisOf = (s: Signal): AxisKind =>
  (s.axis as AxisKind) ?? (s.unit === "V" ? "voltage" : s.unit === "A" ? "current" : s.unit === "C" ? "charge" : s.unit === "1" ? "logic" : "state");

/** Summary cells of the quick benches, in this order when present (then the rest in server order). */
const BENCH_PRIORITY = ["V_LU", "V_LD", "window", "p_any_lu", "P1", "p_one", "P_sw", "P_sw1", "P_both", "vd_first_lu", "n_latch_up", "final_state", "delay", "delay1", "P_retained", "corr_LU", "corr_sw"];

/** Waveform signals shown by default in 간단히: supply / pulse voltage, drain (and source) voltage, drain current. */
const isDefaultSignal = (key: string) => /^(v_src|v_clk|v_d\d*|v_s\d*|i_d\d*)$/.test(key);

/** Default netlists shown before the first run (element names as server/compute/circuit/benches.py). */
function defaultSchematic(bench: BenchId, vg: number): CircuitResult["schematic"] {
  const cell = (k: number, d: string, g: string) => [
    { kind: "STL", name: `X${k}`, nodes: [d, g, "0"] },
    { kind: "V", name: `VG${k}`, nodes: [g, "0"], value: `${vg} V (DC)` },
  ];
  if (bench === "pbit")
    return {
      nodes: ["d", "g", "s", "0"],
      elements: [
        { kind: "V", name: "Vclk", nodes: ["d", "0"], value: "drain pulses" },
        { kind: "V", name: "VG1", nodes: ["g", "0"], value: `${vg} V (DC)` },
        { kind: "STL", name: "X1", nodes: ["d", "g", "s"] },
        { kind: "R", name: "RS", nodes: ["s", "0"], value: "100 kΩ" },
        { kind: "CMP", name: "CMP", nodes: ["s"], value: "bit = [V(R_S) > V_ref]" },
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

/** 모두 보기: the bench cards with a mini schematic and a two-line description. */
function BenchCards() {
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

/** 간단히: 44 px chips (glyph + name; the description is the tooltip and the line under the chips). */
function BenchChips() {
  const t = useT();
  const bench = useStore((s) => s.params.circuit.bench);
  const setParam = useStore((s) => s.setParam);
  const ids = BENCH_ORDER;
  const onKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const i = ids.indexOf(bench);
    const j = e.key === "ArrowRight" || e.key === "ArrowDown" ? (i + 1) % ids.length : e.key === "ArrowLeft" || e.key === "ArrowUp" ? (i - 1 + ids.length) % ids.length : -1;
    if (j < 0) return;
    e.preventDefault();
    setParam(["circuit", "bench"], ids[j]);
    (e.currentTarget.querySelectorAll("button")[j] as HTMLButtonElement | undefined)?.focus();
  };
  return (
    <div className="bench-chips" role="radiogroup" aria-label={t("c.bench")} data-testid="bench-picker" onKeyDown={onKey}>
      {ids.map((id) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={bench === id}
          tabIndex={bench === id ? 0 : -1}
          className="bench-chip"
          title={t(BENCHES[id].desc)}
          onClick={() => setParam(["circuit", "bench"], id)}
          data-testid={`bench-${id}`}
        >
          <BenchGlyph bench={id} />
          <span>{t(BENCHES[id].title)}</span>
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

/** "신호 ▾": a checkbox list of every returned signal, grouped by axis (the plot legend names what is drawn). */
function SignalMenu({ signals, shown, onToggle }: { signals: Signal[]; shown: Set<string>; onToggle: (key: string) => void }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    wrap.current?.querySelector<HTMLButtonElement>(".pmenu-item")?.focus();
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);
  const groups = AXES.map((ax) => ({ ax, items: signals.filter((s) => axisOf(s) === ax) })).filter((g) => g.items.length);
  const axisName: Record<AxisKind, StrKey> = { voltage: "c.axis.voltage", current: "c.axis.current", charge: "c.axis.charge", state: "c.axis.state", logic: "c.axis.logic" };
  return (
    <div
      className="pmenu-wrap"
      ref={wrap}
      onKeyDown={(e) => {
        if (e.key === "Escape" && open) {
          e.stopPropagation();
          setOpen(false);
          btn.current?.focus();
        }
      }}
    >
      <button ref={btn} type="button" className="btn sm tool-btn" aria-haspopup="true" aria-expanded={open} onClick={() => setOpen((o) => !o)} data-testid="signals-menu">
        {t("c.signals")} <span className="muted">{shown.size}/{signals.length}</span>
        <IconChevron size={12} />
      </button>
      {open && (
        <div className="pmenu signal-pop" role="group" aria-label={t("c.signals")}>
          {groups.map((g) => (
            <div key={g.ax} className="pmenu-sec">
              <div className="pmenu-title">{t(axisName[g.ax])}</div>
              {g.items.map((s) => {
                const on = shown.has(s.key);
                return (
                  <button key={s.key} type="button" className="pmenu-item" role="menuitemcheckbox" aria-checked={on} onClick={() => onToggle(s.key)} data-testid={`signal-${s.key}`}>
                    <span className={`pmenu-mark pm-box${on ? " is-on" : ""}`} aria-hidden>
                      {on && <IconCheck size={11} />}
                    </span>
                    {subs(t.l(s.label))}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SolverStats({ res }: { res: CircuitResult }) {
  const t = useT();
  const ss = res.solver_stats;
  return (
    <span className="solver-stats mono" data-testid="solver-stats">
      {t("c.steps")} {fmtInt(ss?.steps)} · {t("c.rejected")} {fmtInt(ss?.rejected)} · Newton {fmtInt(ss?.newton_iters)} · {fmtDuration(ss?.runtime_s ?? res.runtime_s)}
    </span>
  );
}

function WaveformPanel({ res, entry, currentKey }: { res: CircuitResult | undefined; entry: ResultEntry | undefined; currentKey: string }) {
  const t = useT();
  const c = usePalette();
  const all = useIsAll();
  const [run, setRun] = useState<string>("0");
  const [logI, setLogI] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  // 간단히: the signals the user added / removed per result shape (reset when the bench's signal set changes)
  const sigs = useMemo(() => res?.runs[0]?.signals ?? [], [res]);
  const sigKey = sigs.map((s) => s.key).join(",");
  const [picked, setPicked] = useState<{ for: string; keys: Set<string> } | null>(null);
  const shown = useMemo(() => {
    if (all) return new Set(sigs.map((s) => s.key));
    if (picked && picked.for === sigKey) return picked.keys;
    const d = sigs.filter((s) => isDefaultSignal(s.key)).map((s) => s.key);
    return new Set(d.length ? d : sigs.slice(0, 3).map((s) => s.key));
  }, [all, picked, sigKey, sigs]);
  const toggle = (key: string) => {
    const next = new Set(shown);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setPicked({ for: sigKey, keys: next });
  };
  const plot = useMemo(() => {
    if (!res || !res.runs.length) return undefined;
    const runs = run === "all" ? res.runs : res.runs.filter((r) => String(r.run) === run).slice(0, 1);
    const colorOf = new Map(sigs.map((s, i) => [s.key, c.categorical[i % c.categorical.length]] as const));
    const present = AXES.filter((a) => res.runs.some((r) => r.signals.some((s) => shown.has(s.key) && axisOf(s) === a)));
    const n = Math.max(1, present.length);
    const gap = 0.07;
    const h = (1 - gap * (n - 1)) / n;
    const [ts, tu] = timeScale(res.runs[0].t);
    const traces: Data[] = [];
    const layout: Partial<Layout> = {
      margin: { l: 70, r: 16, t: 40, b: 46 },
      xaxis: { title: { text: t("axis.time", { u: tu }) }, anchor: `y${n > 1 ? n : ""}` as never },
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 11.5 } },
    };
    present.forEach((ax, i) => {
      const yName = i === 0 ? "yaxis" : `yaxis${i + 1}`;
      const top = 1 - i * (h + gap);
      const unit = res.runs[0].signals.find((s) => axisOf(s) === ax)?.unit ?? "";
      const cur = ax === "current";
      (layout as Record<string, unknown>)[yName] = {
        domain: [Math.max(0, top - h), top],
        ...(cur ? { ...currentAxis(logI, ""), ...(logI ? { range: logRange(runs.flatMap((r) => r.signals.filter((x) => shown.has(x.key) && axisOf(x) === "current").map((x) => pos(x.values.map((v) => (v == null ? null : Math.abs(v))))))) } : {}) } : ax === "logic" ? {} : siTicks(unit)),
        title: { text: circuitAxisTitle(t, ax, unit, logI), font: { size: 11 } },
        ...(ax === "logic" ? { range: [-0.2, 1.2], dtick: 1 } : {}),
      };
      for (const r of runs) {
        for (const s of r.signals.filter((x) => shown.has(x.key) && axisOf(x) === ax)) {
          const color = colorOf.get(s.key) ?? c.categorical[0];
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
  }, [res, run, logI, showEvents, shown, sigs, c, t]);

  const runSel = res && res.runs.length > 1 && (
    <>
      <label className="tb-label" htmlFor="run-sel">{t("c.run")}</label>
      <select id="run-sel" data-testid="run-sel" className="select" style={{ width: 130, height: 26 }} value={run} onChange={(e) => setRun(e.target.value)}>
        {res.runs.map((r) => (
          <option key={r.run} value={String(r.run)}>#{r.run}</option>
        ))}
        <option value="all">{t("c.overlay")}</option>
      </select>
    </>
  );
  if (all) {
    return (
      <Panel id="waves" wide primary currentKey={currentKey} title={t("c.waves")} desc={t("c.waves.desc")} topic="circuit-element" entry={entry} hasData={!!res} csvName="waveforms" plot={plot}
        toolbar={
          <>
            {runSel}
            <Segmented label="I" value={logI ? "log" : "lin"} onChange={(v) => setLogI(v === "log")} options={[{ v: "log", label: `I ${t("log")}` }, { v: "lin", label: `I ${t("lin")}` }]} />
            <button type="button" className={`btn sm check-btn${showEvents ? " on" : ""}`} aria-pressed={showEvents} onClick={() => setShowEvents(!showEvents)}>
              <span aria-hidden className="check-mark" />
              {t("c.events")}
            </button>
          </>
        }
      />
    );
  }
  const menu: PanelMenuItem[] = [
    { kind: "radio", id: "i-log", group: t("c.axis.current"), label: t("log"), checked: logI, onSelect: () => setLogI(true), testId: "waves-log" },
    { kind: "radio", id: "i-lin", group: t("c.axis.current"), label: t("lin"), checked: !logI, onSelect: () => setLogI(false), testId: "waves-lin" },
    { kind: "check", id: "events", label: t("c.eventMarks"), checked: showEvents, onChange: setShowEvents, testId: "waves-events" },
  ];
  return (
    <Panel
      id="waves"
      wide
      primary
      currentKey={currentKey}
      title={t("c.waves")}
      desc={t("c.waves.desc")}
      topic="circuit-element"
      entry={entry}
      hasData={!!res}
      csvName="waveforms"
      plot={plot}
      menu={menu}
      toolbar={
        res && sigs.length ? (
          <div className="tb-row">
            {runSel}
            <SignalMenu signals={sigs} shown={shown} onToggle={toggle} />
          </div>
        ) : undefined
      }
      foot={res ? <SolverStats res={res} /> : undefined}
    />
  );
}

function TrajectoryPanel({ res, entry, currentKey }: { res: CircuitResult | undefined; entry: ResultEntry | undefined; currentKey: string }) {
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

function PickSelect({ value, onChange, options, label, width }: { value: number; onChange: (i: number) => void; options: string[]; label: string; width: number }) {
  return (
    <select className="select" style={{ width, maxWidth: "100%", height: 26 }} value={value} onChange={(e) => onChange(Number(e.target.value))} aria-label={label}>
      {options.map((x, i) => (
        <option key={i} value={i}>
          {x}
        </option>
      ))}
    </select>
  );
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
      toolbar={dists.length > 1 ? <PickSelect value={k} onChange={setK} options={dists.map((x) => t.l(x.label))} label={t("c.dist")} width={220} /> : undefined}
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
      toolbar={sw.length > 1 ? <PickSelect value={k} onChange={setK} options={sw.map((x) => t.l(x.label))} label={t("c.sweeps")} width={240} /> : undefined}
    />
  );
}

function EventsPanel({ res, stale }: { res: CircuitResult; stale: boolean }) {
  const t = useT();
  const all = useIsAll();
  const [ts, tu] = timeScale(res.runs[0]?.t ?? []);
  const ss = res.solver_stats;
  return (
    <Panel id="c-events" title={all ? `${t("c.events")} · ${t("c.solver")}` : t("c.eventsTable")} topic="numerics" hasData stale={stale}>
      <div className="panel-foot">
        {/* 간단히: the solver statistics are the waveform footnote */}
        {all && (
          <div className="row small mono" style={{ flexWrap: "wrap", gap: 12 }} data-testid="solver-stats">
            <span>{t("c.steps")}: <strong>{fmtInt(ss?.steps)}</strong></span>
            <span>{t("c.rejected")}: <strong>{fmtInt(ss?.rejected)}</strong></span>
            <span>{t("c.newton")}: <strong>{fmtInt(ss?.newton_iters)}</strong></span>
            <span>{t("kpi.runtime")}: <strong>{fmtDuration(ss?.runtime_s ?? res.runtime_s)}</strong></span>
          </div>
        )}
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
      icon: <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M3 6h5l2-3 3 6 2-3h6" /><path d="M6 6v6m0 4v5M3 12h6M3 16h6M18 6v15" /><path d="M14 21h8" /></svg>,
    },
    {
      v: "benches", title: "schematic.view.benches", sub: "schematic.view.benchesSub",
      icon: <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>,
    },
  ];
  return (
    <div className="view-switch" role="tablist" aria-label={t("schematic.view.aria")} data-testid="circuit-view-switch">
      {opts.map((o) => (
        <button key={o.v} type="button" role="tab" aria-selected={view === o.v} className="view-tab" title={t(o.sub)} onClick={() => setView(o.v)} data-testid={`circuit-view-${o.v}`}>
          {o.icon}
          <span className="view-tab-text">{t(o.title)}</span>
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
      <div className="circuit-top">
        <ViewSwitch />
        <LayoutToggle />
      </div>
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

function ResolvedParams({ params }: { params: Record<string, unknown> }) {
  return (
    <div className="chips" data-testid="resolved-params" aria-label="resolved bench parameters">
      {Object.entries(params)
        .filter(([, v]) => typeof v === "number" || typeof v === "string")
        .map(([k, v]) => (
          <span key={k} className="chip" title={k}>
            {k} = {typeof v === "number" ? (Number.isInteger(v) ? String(v) : fmtSig(v, 4)) : String(v)}
          </span>
        ))}
    </div>
  );
}

/** 간단히: the schematic as a ~180 px thumbnail (click → full size in a dialog) with the resolved bench parameters folded. */
function SchematicThumb({ sch, title, resolved, running, entry, currentKey, badge }: { sch: CircuitResult["schematic"]; title: string; resolved?: Record<string, unknown>; running: boolean; entry: ResultEntry | undefined; currentKey: string; badge: ReactNode }) {
  const t = useT();
  const [big, setBig] = useState(false);
  return (
    <Panel id="schematic" title={t("c.schematic")} desc={t("c.schematic.desc")} topic="circuit-element" hasData entry={running ? entry : undefined} currentKey={currentKey} badges={badge}>
      <div className="panel-foot schematic-thumb bench-thumb">
        <button type="button" className="thumb-btn" onClick={() => setBig(true)} aria-label={t("c.schematic.enlarge")} title={t("c.schematic.enlarge")} data-testid="schematic-enlarge">
          <Schematic nodes={sch.nodes} elements={sch.elements} title={title} />
        </button>
        {resolved && (
          <details className="resolved">
            <summary data-testid="resolved-params-toggle">{t("c.resolved")}</summary>
            <ResolvedParams params={resolved} />
          </details>
        )}
      </div>
      {big && (
        <Modal title={`${t("c.schematic")} · ${title}`} onClose={() => setBig(false)} width={980} testId="schematic-modal">
          <div className="schematic-big">
            <Schematic nodes={sch.nodes} elements={sch.elements} title={title} testId="schematic-full" />
          </div>
        </Modal>
      )}
    </Panel>
  );
}

function BenchesView() {
  const t = useT();
  const all = useIsAll();
  const params = useStore((s) => s.params);
  const mode = useStore((s) => s.mode);
  const { entry, data: res } = useEntry<CircuitResult>("circuit");
  const key = useCurrentKey("circuit", useMemo(() => circuitPayload(params, mode), [params, mode]));
  const bench = params.circuit.bench;
  const mine = res && res.bench === bench ? res : undefined;
  const sch = mine ? mine.schematic : defaultSchematic(bench, params.device.vg);
  const running = entry?.status === "running" || entry?.status === "queued";
  const stale = isStale(entry, key);
  const badge = mine ? <span className="badge">{mine.mode === "stochastic" ? t("mode.stochastic") : t("mode.deterministic")}</span> : <span className="badge">{t(BENCHES[bench].title)}</span>;

  if (all) {
    return (
      <>
        <BenchCards />
        {mine && mine.summary.length > 0 && <SummaryStrip items={mine.summary} priority={BENCH_PRIORITY} testId="circuit-summary" prefix="c-" moreTestId="summary-more" stale={stale} />}
        <div className="grid">
          <Panel id="schematic" wide title={t("c.schematic")} desc={t("c.schematic.desc")} topic="circuit-element" hasData entry={running ? entry : undefined} currentKey={key} badges={badge}>
            <div className="panel-foot schematic">
              <Schematic nodes={sch.nodes} elements={sch.elements} title={t(BENCHES[bench].title)} />
            </div>
            {mine?.bench_params && (
              <div className="panel-foot">
                <ResolvedParams params={mine.bench_params} />
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

  const st = entryStatus(entry, key);
  const tabs: MoreTab[] = [
    { id: "trajectory", label: t("c.traj"), panel: <TrajectoryPanel res={res} entry={entry} currentKey={key} />, status: st },
    { id: "c-dist", label: t("c.dist"), panel: res ? <DistributionsPanel res={res} stale={stale} /> : null, status: st, hidden: !res || !(res.distributions ?? []).some((d) => finite(d.values).length >= 3) },
    { id: "c-sweeps", label: t("c.sweeps"), panel: res ? <SweepsPanel res={res} stale={stale} /> : null, status: st, hidden: !res?.sweeps?.length },
    { id: "c-events", label: t("c.eventsTable"), panel: res ? <EventsPanel res={res} stale={stale} /> : null, status: st, hidden: !res },
  ];
  return (
    <>
      {/* chips, description and the result summary on the left; the schematic thumbnail spans them on the right */}
      <div className="bench-top">
        <BenchChips />
        <p className="bench-line">{t(BENCHES[bench].desc)}</p>
        {mine && mine.summary.length > 0 ? (
          <SummaryStrip items={mine.summary} priority={BENCH_PRIORITY} testId="circuit-summary" prefix="c-" moreTestId="summary-more" stale={stale} className="bench-sum" />
        ) : (
          <div className="sum-strip sum-empty bench-sum">
            <span>{running ? t("loading") : t("c.summary.empty")}</span>
          </div>
        )}
        <SchematicThumb sch={sch} title={t(BENCHES[bench].title)} resolved={mine?.bench_params} running={running} entry={entry} currentKey={key} badge={badge} />
      </div>
      <FocusLayout testId="circuit-panels" scope="circuit" hero={<WaveformPanel res={res} entry={entry} currentKey={key} />} tabs={tabs} defaultTab="trajectory" />
    </>
  );
}
