// Validation tab: `validation` kind (fast/full) check table + comparisons with the measured records
// (reference-calibration I–V, 8 illumination conditions, V_G dependence of V_LU and σ_LU).
// The answer comes first: "모델이 측정과 맞나요?" with a big "n / n 통과" status, a 4-column table (항목 · 계산값 ·
// 기대값 ± 허용오차 · 통과; failures first and expanded; "숫자 모두 보기" adds id, seconds and notes), then the
// figures in one tabbed card (간단히) or all three in a grid (모두 보기).
import type { Data, Layout } from "plotly.js";
import { useEffect, useMemo, useState } from "react";
import type { BranchesResult, SweepMCResult, ValidationCheck, ValidationResult, VgCurveStochasticResult } from "../api/types";
import { LayoutToggle } from "../components/LayoutToggle";
import { entryStatus, FocusLayout, mergeStatus, type MoreTab, type TabStatus } from "../components/MoreCard";
import { Panel, Progress } from "../components/Panel";
import { IconPlay, IconStop } from "../components/icons";
import { useT } from "../i18n";
import { signed } from "../plots/labels";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { cancelActive, cancelKey, loadMeasured, runValidation, runValidationIV, runValidationPhoto, runValidationVg } from "../state/runner";
import { useIsAll } from "../state/layout";
import { useStore } from "../state/store";
import { fmtDuration } from "../utils/format";
import { logRange, nums, pos, useEntry, usePalette } from "../device/common";
import { insideLegend, measuredIvTraces } from "../device/DetPanels";
import { PhotoConditionsStats } from "./PhotoStats";

const passMark = (c: ValidationCheck) => (
  <span className={`pass ${c.pass === true ? "yes" : c.pass === false ? "no" : "na"}`} aria-label={c.pass === true ? "pass" : c.pass === false ? "fail" : "not run"}>
    {c.pass === true ? "✓" : c.pass === false ? "✗" : "—"}
  </span>
);

/** "±1 mV" → "1 mV": the column header already says "± 허용오차". */
const tol = (s: string) => s.replace(/^\s*±\s*/, "");

/** Compact check table: failures first (with their id and note), 4 columns; `wide` adds id, seconds and notes. */
function ChecksTable({ res, wide }: { res: ValidationResult; wide: boolean }) {
  const t = useT();
  // failures first, then the passed checks, then the ones not run at this level (stable within each group)
  const rank = (c: ValidationCheck) => (c.pass === false ? 0 : c.pass === true ? 1 : 2);
  const rows = res.checks.map((c, i) => ({ c, i })).sort((a, b) => rank(a.c) - rank(b.c) || a.i - b.i);
  return (
    <div className="table-wrap val-table-wrap">
      <table className={`table val-table${wide ? " wide" : ""}`} data-testid="validation-table">
        <thead>
          <tr>
            <th>{t("v.check")}</th>
            {wide && <th>{t("v.id")}</th>}
            <th>{t("v.computed")}</th>
            <th>{t("v.expectedTol")}</th>
            <th className="center">{t("v.pass")}</th>
            {wide && <th className="num">{t("v.sec")}</th>}
            {wide && <th>{t("v.note")}</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ c }) => (
            <tr key={c.id} className={c.pass === false ? "fail" : c.pass === null ? "na" : undefined} data-check={c.id}>
              <td title={c.id}>
                <div className="val-label">{t.l(c.label)}</div>
                {!wide && c.pass === false && (
                  <div className="val-detail small mono">
                    {c.id}
                    {c.note ? ` · ${c.note}` : ""} · {c.seconds.toFixed(2)} s
                  </div>
                )}
              </td>
              {wide && <td className="mono small muted">{c.id}</td>}
              <td className="mono small" data-label={t("v.computed")}>{c.computed}</td>
              <td className="mono small" data-label={t("v.expectedTol")}>
                {c.expected}
                {c.tolerance && <span className="val-tol"> ± {tol(c.tolerance)}</span>}
              </td>
              <td className="center val-pass">{passMark(c)}</td>
              {wide && <td className="num small">{c.seconds.toFixed(2)}</td>}
              {wide && <td className="small muted">{c.note ?? ""}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PaperIvFigure() {
  const t = useT();
  const c = usePalette();
  const measured = useStore((s) => s.measured);
  const { entry, data } = useEntry<BranchesResult>("val_branches_paper");
  useEffect(() => {
    if (!useStore.getState().results.val_branches_paper) void runValidationIV();
  }, []);
  useEffect(() => {
    if (measured.status === "idle") void loadMeasured();
  }, [measured.status]);
  const plot = useMemo(() => {
    if (!data) return undefined;
    const traces: Data[] = [...measuredIvTraces(t, c, measured.data, "paper", 0)];
    traces.push(
      { x: nums(data.double_sweep.up.vd), y: pos(data.double_sweep.up.id), type: "scatter", mode: "lines", name: t("axis.leg.modelUp"), line: { color: c.up, width: 2 }, hovertemplate: `${HOVER_IV}<extra>${t("model")} ↑</extra>` },
      { x: nums(data.double_sweep.down.vd), y: pos(data.double_sweep.down.id), type: "scatter", mode: "lines", name: t("axis.leg.modelDown"), line: { color: c.down, width: 2 }, hovertemplate: `${HOVER_IV}<extra>${t("model")} ↓</extra>` },
      { x: nums(data.unstable.vd), y: pos(data.unstable.id), type: "scatter", mode: "lines", name: t("iv.unstable"), line: { color: c.unstable, width: 1.2, dash: "dash" }, hoverinfo: "skip" },
    );
    return { data: traces, layout: { xaxis: { title: { text: t("axis.vd") }, range: [0, 4.1] }, yaxis: { ...currentAxis(true, t("axis.idAbs")), range: logRange(traces.map((tr) => (tr as { y?: (number | null)[] }).y)) }, legend: insideLegend(c, "tl") } as Partial<Layout> };
  }, [data, measured.data, c, t]);
  return <Panel id="val-iv" title={t("v.fig.iv")} desc={t("v.fig.iv.desc")} topic="validation" entry={entry} hasData={!!data} csvName="validation_reference_iv" plot={plot} error={measured.status === "error" ? measured.error : null} />;
}

function PhotoFigure() {
  const t = useT();
  const c = usePalette();
  const measured = useStore((s) => s.measured);
  const conds = useStore((s) => s.meta.measured_photo_conditions ?? []);
  const results = useStore((s) => s.results);
  useEffect(() => {
    if (measured.status === "idle") void loadMeasured();
  }, [measured.status]);
  const entries = conds.map((_, k) => results[`val_photo_${k}`]);
  const done = entries.filter((e) => e?.status === "done").length;
  const running = entries.some((e) => e?.status === "running" || e?.status === "queued");
  const cur = entries.find((e) => e?.status === "running" || e?.status === "queued");
  const plot = useMemo(() => {
    const m = measured.data?.photo ?? [];
    if (!m.length && !done) return undefined;
    const traces: Data[] = [];
    const vgs = [...new Set(conds.map((x) => x.vg))];
    vgs.forEach((vg, gi) => {
      const col = gi === 0 ? c.sto : c.categorical[1];
      const mm = m.filter((x) => Math.abs(x.vg - vg) < 1e-6).sort((a, b) => a.power_mW - b.power_mW);
      traces.push(
        { x: mm.map((x) => x.power_mW), y: mm.map((x) => x.mean_V), type: "scatter", mode: "markers", name: t("axis.leg.measVg", { vg: signed(vg, 1) }), marker: { color: c.meas, size: 9, symbol: gi === 0 ? "square" : "diamond" }, hovertemplate: `P = %{x:.2f} mW<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra>${t("measured")}</extra>` },
        { x: mm.map((x) => x.power_mW), y: mm.map((x) => x.sd_mV), yaxis: "y2", type: "scatter", mode: "markers", showlegend: false, marker: { color: c.meas, size: 9, symbol: gi === 0 ? "square-open" : "diamond-open" }, hovertemplate: `P = %{x:.2f} mW<br>σ<sub>LU</sub> = %{y:.1f} mV<extra>${t("measured")}</extra>` },
      );
      const model = conds
        .map((cd, k) => ({ cd, r: results[`val_photo_${k}`]?.data as SweepMCResult | undefined }))
        .filter((x) => Math.abs(x.cd.vg - vg) < 1e-6 && x.r)
        .sort((a, b) => a.cd.power_mW - b.cd.power_mW);
      if (model.length) {
        traces.push(
          { x: model.map((x) => x.cd.power_mW), y: model.map((x) => x.r!.stats.LU.mean), type: "scatter", mode: "lines+markers", name: t("axis.leg.modelVg", { vg: signed(vg, 1) }), line: { color: col, width: 2 }, marker: { size: 7 }, hovertemplate: `P = %{x:.2f} mW<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra>${t("model")}</extra>` },
          { x: model.map((x) => x.cd.power_mW), y: model.map((x) => (x.r!.stats.LU.sd ?? NaN) * 1e3), yaxis: "y2", type: "scatter", mode: "lines+markers", showlegend: false, line: { color: col, width: 2, dash: "dash" }, marker: { size: 7 }, hovertemplate: `P = %{x:.2f} mW<br>σ<sub>LU</sub> = %{y:.1f} mV<extra>${t("model")}</extra>` },
        );
      }
    });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.power") }, anchor: "y2" },
      yaxis: { title: { text: t("axis.s.vluMean") }, domain: [0.5, 1] },
      yaxis2: { title: { text: t("axis.s.sigmaLu") }, domain: [0, 0.42], rangemode: "tozero" },
      margin: { l: 58, r: 16, t: 46, b: 46 },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [measured.data, results, conds, done, c, t]);
  return (
    <Panel
      id="val-photo"
      title={t("v.fig.photo")}
      desc={t("v.fig.photo.desc")}
      topic="photo"
      hasData={!!plot}
      csvName="validation_photo_conditions"
      plot={plot}
      entry={running ? { status: "running", progress: done / Math.max(1, conds.length), message: `${t("v.progress", { done, total: conds.length })}${cur?.message ? " · " + cur.message : ""}` } : undefined}
      toolbar={
        <>
          <button type="button" className="btn sm primary" onClick={() => void runValidationPhoto()} disabled={running} data-testid="val-photo-run">
            <IconPlay size={11} /> {t("v.photo.run")}
          </button>
          {running && (
            <button type="button" className="btn sm danger" onClick={cancelActive}>
              <IconStop size={10} /> {t("run.cancel")}
            </button>
          )}
          {done > 0 && <span className="small muted">{t("v.progress", { done, total: conds.length })}</span>}
        </>
      }
    >
      <PhotoConditionsStats />
    </Panel>
  );
}

function VgFigure() {
  const t = useT();
  const c = usePalette();
  const { entry, data } = useEntry<VgCurveStochasticResult>("val_vgs");
  const plot = useMemo(() => {
    if (!data) return undefined;
    const vg = nums(data.vg);
    const traces: Data[] = [
      { x: vg, y: nums(data.mean_VLU), type: "scatter", mode: "lines+markers", name: t("axis.leg.vluMean"), line: { color: c.sto, width: 2 }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra>${t("axis.leg.vluMean")}</extra>` },
      { x: vg, y: nums(data.fold_centre_V), type: "scatter", mode: "lines", name: t("vgs.fold"), line: { color: c.hrs, width: 1.3, dash: "dash" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>V<sub>LU</sub> = %{y:.3f} V<extra>${t("vgs.fold")}</extra>` },
      { x: vg, y: nums(data.sd_VLU_mV), yaxis: "y2", type: "scatter", mode: "lines+markers", name: t("axis.leg.sdTotal"), line: { color: c.sto, width: 2 }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ<sub>LU</sub> = %{y:.1f} mV<extra>${t("axis.leg.sdTotal")}</extra>` },
      { x: vg, y: nums(data.state_sd_mV), yaxis: "y2", type: "scatter", mode: "lines", name: t("vgs.state"), line: { color: c.categorical[1], width: 1.4, dash: "dash" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ = %{y:.1f} mV<extra>${t("vgs.state")}</extra>` },
      { x: vg, y: nums(data.noise_sd_mV), yaxis: "y2", type: "scatter", mode: "lines", name: t("vgs.noise"), line: { color: c.categorical[2], width: 1.4, dash: "dot" }, hovertemplate: `V<sub>G</sub> = %{x:.2f} V<br>σ = %{y:.1f} mV<extra>${t("vgs.noise")}</extra>` },
      { x: [-1.1], y: [4.354], type: "scatter", mode: "markers", name: t("brand.v.refMeanPeak"), marker: { color: c.meas, symbol: "star", size: 11 }, hovertemplate: `${t("brand.v.refMeanPeak")}<extra></extra>` },
      { x: [-1.25], y: [129.8], yaxis: "y2", type: "scatter", mode: "markers", name: t("brand.v.refSdPeak"), marker: { color: c.meas, symbol: "star-open", size: 11 }, hovertemplate: `${t("brand.v.refSdPeak")}<extra></extra>` },
    ];
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.vg") }, anchor: "y2" },
      yaxis: { title: { text: t("axis.s.vluMean") }, domain: [0.5, 1] },
      yaxis2: { title: { text: t("axis.s.sigmaLu") }, domain: [0, 0.42], rangemode: "tozero" },
      margin: { l: 58, r: 16, t: 58, b: 46 },
      legend: { font: { size: 10.5 }, traceorder: "normal" },
    };
    return { data: traces, layout, className: "plot tall" };
  }, [data, c, t]);
  const running = entry?.status === "running" || entry?.status === "queued";
  return (
    <Panel
      id="val-vg"
      title={t("v.fig.vg")}
      desc={t("v.fig.vg.desc")}
      topic="local-states"
      entry={entry}
      hasData={!!data}
      csvName="validation_vg"
      plot={plot}
      warnings={data?.warnings}
      empty={
        <>
          <span>{t("empty.compute")}</span>
          <button type="button" className="btn primary sm" onClick={() => void runValidationVg()}>{t("compute")}</button>
        </>
      }
      toolbar={
        <>
          <button type="button" className="btn sm" onClick={() => void runValidationVg()} disabled={running}>{data ? t("recompute") : t("compute")}</button>
          {running && <button type="button" className="btn sm danger" onClick={() => cancelKey("val_vgs")}>{t("run.cancel")}</button>}
        </>
      }
    />
  );
}

export function ValidationTab() {
  const t = useT();
  const all = useIsAll();
  const { entry, data } = useEntry<ValidationResult>("validation");
  const running = entry?.status === "running" || entry?.status === "queued";
  const pass = data?.checks.filter((c) => c.pass === true).length ?? 0;
  const total = data?.checks.filter((c) => c.pass !== null).length ?? 0;
  const failed = total - pass;
  const [wideCols, setWideCols] = useState<boolean | null>(null);
  const wide = wideCols ?? all;
  return (
    <>
      <section className="panel wide val-card" data-testid="validation" aria-labelledby="val-title">
        <header className="val-head">
          <h2 id="val-title" className="val-title">{t("v.question")}</h2>
          <LayoutToggle />
        </header>
        <div className="val-status-row">
          {data ? (
            <div className={`val-status ${failed ? "bad" : "good"}`} data-testid="val-status" role="status">
              {failed ? t("v.failed", { n: failed }) : t("v.passedAll", { pass, total })}
              {!failed && <span aria-hidden> ✓</span>}
            </div>
          ) : (
            <div className="val-status idle" data-testid="val-status">{running ? t("loading") : t("v.notYet")}</div>
          )}
          <div className="val-actions">
            <button type="button" className="btn primary" onClick={() => void runValidation("fast")} disabled={running} data-testid="val-fast">
              <IconPlay size={12} /> {t("v.fast")}
            </button>
            <button type="button" className="btn" onClick={() => void runValidation("full")} disabled={running} data-testid="val-full">
              <IconPlay size={12} /> {t("v.full")}
            </button>
            {running && (
              <button type="button" className="btn danger" onClick={() => cancelKey("validation")}>
                <IconStop size={11} /> {t("run.cancel")}
              </button>
            )}
            {data && !running && <span className="small muted mono">{fmtDuration(data.runtime_s)}</span>}
            {entry?.mock && data && all && <span className="badge demo">{t("demo")}</span>}
          </div>
        </div>
        {running && (
          <div className="val-progress">
            <Progress value={entry?.progress ?? 0} indeterminate={(entry?.progress ?? 0) < 0.01} />
            <span className="small muted mono">{entry?.message}</span>
          </div>
        )}
        {entry?.status === "error" && (
          <div className="panel-foot">
            <div className="err-box" role="alert">{entry.error}</div>
          </div>
        )}
        {data ? (
          <>
            <ChecksTable res={data} wide={wide} />
            <div className="val-foot">
              <span className="small muted">{t("v.desc")}</span>
              <button type="button" className="link-btn" aria-pressed={wide} onClick={() => setWideCols(!wide)} data-testid="val-cols-all">
                {wide ? t("v.colsFew") : t("v.colsAll")}
              </button>
            </div>
          </>
        ) : (
          !running && <p className="val-empty small muted">{t("v.emptyHint")}</p>
        )}
        {data?.warnings && data.warnings.length > 0 && (
          <div className="panel-foot">
            <details className="notices"><summary>{t("warnings")} · {data.warnings.length}</summary><ul>{data.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></details>
          </div>
        )}
      </section>
      {all ? (
        <>
          <div className="section-title">
            <h2>{t("v.figs")}</h2>
          </div>
          <div className="grid">
            <PaperIvFigure />
            <PhotoFigure />
            <VgFigure />
          </div>
        </>
      ) : (
        <ValidationFigures />
      )}
    </>
  );
}

/** 간단히: the three comparisons as tabs of one card (기준 I–V first). */
function ValidationFigures() {
  const t = useT();
  const iv = useStore((s) => s.results.val_branches_paper);
  const vgs = useStore((s) => s.results.val_vgs);
  // a primitive selector result (a new array would re-render forever)
  const photoStatus = useStore((s): TabStatus | null => mergeStatus(...Object.entries(s.results).filter(([k]) => k.startsWith("val_photo_")).map(([, e]) => entryStatus(e))));
  const tabs: MoreTab[] = [
    { id: "val-iv", label: t("v.tab.iv"), panel: <PaperIvFigure />, status: entryStatus(iv), title: t("v.fig.iv") },
    { id: "val-photo", label: t("v.tab.photo"), panel: <PhotoFigure />, status: photoStatus, title: t("v.fig.photo") },
    { id: "val-vg", label: t("v.tab.vg"), panel: <VgFigure />, status: entryStatus(vgs), title: t("v.fig.vg") },
  ];
  return <FocusLayout testId="val-figures" scope="validation" tabs={tabs} defaultTab="val-iv" />;
}
