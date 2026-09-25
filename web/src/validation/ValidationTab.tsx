// Reference benchmark: authentic measured ID–VD versus the calibrated quasi-static voltage sweep.
// Engine reproducibility tests remain in the developer test suite, not an accuracy score in the UI.
import type { Data, Layout } from "plotly.js";
import { useEffect, useMemo, useState } from "react";
import type { BranchesResult } from "../api/types";
import { Panel } from "../components/Panel";
import { logRange, nums, pos, useEntry, usePalette } from "../device/common";
import { ROW_LEGEND, Seg } from "../device/DetPanels";
import { useT } from "../i18n";
import { REF } from "../i18n/strings.reference";
import { DEV } from "../i18n/strings.device";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { loadMeasured, runValidationIV } from "../state/runner";
import { useStore } from "../state/store";
import { currentFloor, curveError, switchVoltage } from "./benchmark";
import { fill } from "../i18n/strings.ux";
import { fmtSI } from "../utils/format";
import frozen from "./data/reference-idvd.json";
import "./reference.css";
export { checkText } from "./checkText";

export function ValidationTab() {
  const t = useT();
  const c = usePalette();
  const backend = useStore(s => s.backend);
  const measured = useStore(s => s.measured);
  const { entry, data } = useEntry<BranchesResult>("val_branches_paper");
  const [log, setLog] = useState(true);
  const genuine = backend === "online" || backend === "snapshot";
  const reference = (genuine ? measured.data?.paper_iv : null) ?? frozen.measured;
  const computed = genuine && !entry?.mock ? data : undefined;
  const model = computed ?? frozen.model;
  const L = (k: keyof typeof REF) => t.l(REF[k]);
  useEffect(() => {
    if (!genuine) return;
    if (!useStore.getState().results.val_branches_paper || useStore.getState().results.val_branches_paper?.mock) void runValidationIV();
    if (useStore.getState().measured.status === "idle") void loadMeasured();
  }, [genuine]);

  const plot = useMemo(() => {
    if (!model || !reference) return undefined;
    const Y = log ? pos : nums;
    const traces: Data[] = [];
    for (const [vd, id, color, name] of [
      [reference.vd_up, reference.median_up, c.hrs, L("measuredUp")],
      [reference.vd_down, reference.median_down, c.lrs, L("measuredDown")],
    ] as const) traces.push({ x: nums(vd), y: Y(id), type: "scatter", mode: "markers", name,
      marker: { color, symbol: "circle-open", size: 4.5, line: { width: 1 } }, opacity: 0.65,
      hovertemplate: `${HOVER_IV}<extra>${name}</extra>` });
    for (const [sweep, color, name] of [
      [model.double_sweep.up, c.hrs, L("modelUp")],
      [model.double_sweep.down, c.lrs, L("modelDown")],
    ] as const) traces.push({ x: nums(sweep.vd), y: Y(sweep.id), type: "scatter", mode: "lines", name,
      line: { color, width: 2.3 }, hovertemplate: `${HOVER_IV}<extra>${name}</extra>` });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vd") }, range: [0, 4.05] },
      yaxis: { ...currentAxis(log, log ? t("axis.idAbs") : t("axis.id")), ...(log ? { range: logRange(traces.map(tr => (tr as { y?: (number | null)[] }).y)) } : {}) },
      legend: ROW_LEGEND, margin: { l: 66, r: 20, t: 36, b: 48 },
    };
    return { data: traces, layout };
  }, [model, reference, log, c, t]);
  // headline (D5): the switching voltages and the log error above the instrument floor; the full-range error
  // (≈ 5.5 dec, set by the ~0.4 pA floor where the model current is orders of magnitude smaller) is a detail
  const metrics = useMemo(() => {
    if (!model || !reference) return null;
    const floor = currentFloor(reference.median_up, reference.median_down);
    const thr = floor === null ? -Infinity : 3 * floor;
    const sweeps = [
      { direction: "up" as const, sym: "LU", model: model.double_sweep.up, meas: { vd: reference.vd_up, id: reference.median_up } },
      { direction: "down" as const, sym: "LD", model: model.double_sweep.down, meas: { vd: reference.vd_down, id: reference.median_down } },
    ];
    const rows = sweeps.map((w) => {
      const vMeas = switchVoltage(w.meas, w.direction);
      const vModel = switchVoltage(w.model, w.direction);
      const between = vMeas !== null && vModel !== null ? ([vMeas, vModel] as const) : undefined;
      return {
        direction: w.direction, sym: w.sym, vMeas, vModel,
        delta: vMeas !== null && vModel !== null ? vModel - vMeas : null,
        floorExcluded: curveError(w.model, w.meas, thr),
        switchExcluded: between ? curveError(w.model, w.meas, thr, between) : null,
        all: curveError(w.model, w.meas),
      };
    });
    return { floor, thr, rows };
  }, [model, reference]);
  // "상향 스윕" / "up sweep" inside a sentence: no arrow, lower case in English
  const sweepName = (label: string) => { const s = label.replace(/\s*[↑↓]$/, ""); return t.lang === "en" ? s.charAt(0).toLowerCase() + s.slice(1) : s; };
  const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "—" : v.toFixed(d));
  const volts = (v: number | null) => (v === null ? "—" : `${v.toFixed(3)} V`);
  const mv = (v: number | null) => (v === null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 1e3).toFixed(1)} mV`);

  return <div className="reference-page" data-testid="validation">
    <header className="reference-header">
      <div><h2>{L("title")}</h2><p><SubText text={subs(L("geometry"))} /></p></div>
<div className="reference-header-tags"><span className="badge" data-testid="reference-fixed-label">{L("fixed")}</span><span className="reference-condition"><SubText text={subs(L("condition"))} /></span></div>
    </header>
    <>
      <Panel id="val-iv" title={L("plot")} primary entry={computed ? entry : undefined} hasData={!!plot} plot={plot} csvName="reference_idvd"
        toolbar={plot ? <Seg label={t.l(DEV["menu.y"])} value={log ? "log" : "lin"} onChange={v => setLog(v === "log")} options={[{ v: "log", label: t("log") }, { v: "lin", label: t("lin") }]} /> : undefined}
        error={measured.status === "error" ? L("missing") : undefined}
        empty={<button type="button" className="btn primary sm" onClick={() => { void loadMeasured(true); void runValidationIV(); }}>{L("retry")}</button>}
      />
      {metrics && <div className="reference-metrics" data-testid="reference-metrics">
        <table className="table" data-testid="reference-switching"><thead><tr><th>{L("switching")}</th><th>{L("measured")}</th><th>{L("model")}</th><th>{L("delta")}</th></tr></thead>
          <tbody>{metrics.rows.map((r) => <tr key={r.direction} data-row={r.direction}>
            <td><span className={`q ${r.sym === "LU" ? "q-lu" : "q-ld"}`}>V<sub>{r.sym}</sub></span> <span className="reference-dir">{L(r.direction)}</span></td>
            <td className="mono">{volts(r.vMeas)}</td><td className="mono">{volts(r.vModel)}</td><td className="mono" data-testid={`reference-delta-${r.direction}`}>{mv(r.delta)}</td>
          </tr>)}</tbody>
        </table>
        <table className="table" data-testid="reference-error"><thead><tr><th>{L("direction")}</th><th>{L("logErrorFloor")}</th><th>{L("points")}</th></tr></thead>
          <tbody>{metrics.rows.map((r) => <tr key={r.direction} data-row={r.direction}><td>{L(r.direction)}</td><td className="mono" data-testid={`reference-rmse-${r.direction}`}>{fmt(r.floorExcluded.logRmse)}</td><td className="mono">{r.floorExcluded.logN}</td></tr>)}</tbody>
        </table>
        {metrics.rows.filter((r) => r.switchExcluded?.logRmse != null && r.floorExcluded.logRmse != null && r.floorExcluded.logRmse > 2 * r.switchExcluded.logRmse! && r.vMeas !== null && r.vModel !== null).map((r) => (
          <p key={r.direction} className="reference-floor" data-testid={`reference-switch-note-${r.direction}`}>
            <SubText text={subs(fill(L("switchNote"), { dir: sweepName(L(r.direction)), a: Math.min(r.vMeas!, r.vModel!).toFixed(3), b: Math.max(r.vMeas!, r.vModel!).toFixed(3), v: fmt(r.switchExcluded!.logRmse) }))} />
          </p>
        ))}
        {metrics.floor !== null && <p className="reference-floor"><SubText text={subs(fill(L("floorNote"), { floor: fmtSI(metrics.floor, "A", 2), thr: fmtSI(metrics.thr, "A", 2) }))} /></p>}
        <details data-testid="reference-details"><summary>{L("method")}</summary><p>{L("dataNote")}</p><p><SubText text={subs(L("methodNote"))} /></p>
          <table className="table reference-full" data-testid="reference-full"><thead><tr><th>{L("direction")}</th><th>{L("logErrorAll")}</th><th>{L("logErrorSwitch")}</th><th>{L("linearError")}</th><th>{L("points")}</th></tr></thead>
            <tbody>{metrics.rows.map((r) => <tr key={r.direction}><td>{L(r.direction)}</td><td className="mono">{fmt(r.all.logRmse)}</td><td className="mono">{fmt(r.switchExcluded?.logRmse)}</td><td className="mono">{fmt(r.all.normalizedRmsePct)}</td><td className="mono">{r.all.n}</td></tr>)}</tbody>
          </table>
          <p className="mono small">{L("source")}</p></details>
      </div>}
    </>
  </div>;
}
