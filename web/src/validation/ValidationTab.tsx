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
import { curveError } from "./benchmark";
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
  const errors = useMemo(() => model && reference ? [
    { direction: "up" as const, ...curveError(model.double_sweep.up, { vd: reference.vd_up, id: reference.median_up }) },
    { direction: "down" as const, ...curveError(model.double_sweep.down, { vd: reference.vd_down, id: reference.median_down }) },
  ] : [], [model, reference]);
  const fmt = (v: number | null) => v === null ? "—" : v.toFixed(3);

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
      {errors.length > 0 && <div className="reference-metrics" data-testid="reference-metrics">
        <table className="table"><thead><tr><th>{L("direction")}</th><th>{L("logError")}</th><th>{L("linearError")}</th><th>{L("points")}</th></tr></thead>
          <tbody>{errors.map(e => <tr key={e.direction}><td>{L(e.direction)}</td><td className="mono">{fmt(e.logRmse)}</td><td className="mono">{fmt(e.normalizedRmsePct)}</td><td className="mono">{e.n}</td></tr>)}</tbody>
        </table>
        <details><summary>{L("method")}</summary><p>{L("dataNote")}</p><p>{L("methodNote")}</p><p className="mono small">{L("source")}</p></details>
      </div>}
    </>
  </div>;
}
