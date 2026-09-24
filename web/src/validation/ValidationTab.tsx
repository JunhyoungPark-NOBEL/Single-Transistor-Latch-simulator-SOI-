// Validation tab: `validation` kind (fast/full) check table + figure reproductions (paper I–V, photo
// device 8 conditions, V_G dependence Fig. 3(b)/(c)).
import type { Data, Layout } from "plotly.js";
import { useEffect, useMemo } from "react";
import type { BranchesResult, SweepMCResult, ValidationResult, VgCurveStochasticResult } from "../api/types";
import { Panel, Progress } from "../components/Panel";
import { IconPlay, IconStop } from "../components/icons";
import { useT } from "../i18n";
import { currentAxis, HOVER_IV } from "../plots/theme";
import { cancelActive, cancelKey, loadMeasured, runValidation, runValidationIV, runValidationPhoto, runValidationVg } from "../state/runner";
import { useStore } from "../state/store";
import { fmtDuration } from "../utils/format";
import { logRange, nums, pos, useEntry, usePalette } from "../device/common";
import { insideLegend, measuredIvTraces } from "../device/DetPanels";

function ChecksTable({ res }: { res: ValidationResult }) {
  const t = useT();
  return (
    <div className="table-wrap" style={{ margin: "0 14px 14px" }}>
      <table className="table" data-testid="validation-table">
        <thead>
          <tr>
            <th>{t("v.check")}</th>
            <th>{t("v.expected")}</th>
            <th>{t("v.computed")}</th>
            <th>{t("v.tol")}</th>
            <th style={{ textAlign: "center" }}>{t("v.pass")}</th>
            <th className="num">{t("v.sec")}</th>
            <th>{t("v.note")}</th>
          </tr>
        </thead>
        <tbody>
          {res.checks.map((c) => (
            <tr key={c.id}>
              <td>
                <div style={{ fontWeight: 600 }}>{t.l(c.label)}</div>
                <div className="small muted mono">{c.id}</div>
              </td>
              <td className="mono small">{c.expected}</td>
              <td className="mono small">{c.computed}</td>
              <td className="mono small muted">{c.tolerance}</td>
              <td style={{ textAlign: "center" }}>
                <span className={`pass ${c.pass === true ? "yes" : c.pass === false ? "no" : "na"}`} aria-label={c.pass === true ? "pass" : c.pass === false ? "fail" : "not run"}>
                  {c.pass === true ? "✓" : c.pass === false ? "✗" : "—"}
                </span>
              </td>
              <td className="num small">{c.seconds.toFixed(2)}</td>
              <td className="small muted">{c.note ?? ""}</td>
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
      { x: nums(data.double_sweep.up.vd), y: pos(data.double_sweep.up.id), type: "scatter", mode: "lines", name: `${t("model")} ${t("iv.up")}`, line: { color: c.up, width: 2 }, hovertemplate: `${HOVER_IV}<extra>model ↑</extra>` },
      { x: nums(data.double_sweep.down.vd), y: pos(data.double_sweep.down.id), type: "scatter", mode: "lines", name: `${t("model")} ${t("iv.down")}`, line: { color: c.down, width: 2 }, hovertemplate: `${HOVER_IV}<extra>model ↓</extra>` },
      { x: nums(data.unstable.vd), y: pos(data.unstable.id), type: "scatter", mode: "lines", name: t("iv.unstable"), line: { color: c.unstable, width: 1.2, dash: "dash" }, hoverinfo: "skip" },
    );
    return { data: traces, layout: { xaxis: { title: { text: "V<sub>D</sub> (V)" }, range: [0, 4.1] }, yaxis: { ...currentAxis(true), range: logRange(traces.map((tr) => (tr as { y?: (number | null)[] }).y)) }, legend: insideLegend(c, "tl") } as Partial<Layout> };
  }, [data, measured.data, c, t]);
  return <Panel id="val-iv" title={t("v.fig.iv")} desc={t("v.fig.iv.desc")} topic="validation" entry={entry} hasData={!!data} csvName="validation_paper_iv" plot={plot} error={measured.status === "error" ? measured.error : null} />;
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
        { x: mm.map((x) => x.power_mW), y: mm.map((x) => x.mean_V), type: "scatter", mode: "markers", name: `${t("measured")} V<sub>G</sub>=${vg} V`, marker: { color: c.meas, size: 9, symbol: gi === 0 ? "square" : "diamond" }, hovertemplate: "P = %{x:.2f} mW<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra>measured</extra>" },
        { x: mm.map((x) => x.power_mW), y: mm.map((x) => x.sd_mV), yaxis: "y2", type: "scatter", mode: "markers", showlegend: false, marker: { color: c.meas, size: 9, symbol: gi === 0 ? "square-open" : "diamond-open" }, hovertemplate: "σ = %{y:.1f} mV<extra>measured</extra>" },
      );
      const model = conds
        .map((cd, k) => ({ cd, r: results[`val_photo_${k}`]?.data as SweepMCResult | undefined }))
        .filter((x) => Math.abs(x.cd.vg - vg) < 1e-6 && x.r)
        .sort((a, b) => a.cd.power_mW - b.cd.power_mW);
      if (model.length) {
        traces.push(
          { x: model.map((x) => x.cd.power_mW), y: model.map((x) => x.r!.stats.LU.mean), type: "scatter", mode: "lines+markers", name: `${t("model")} V<sub>G</sub>=${vg} V`, line: { color: col, width: 2 }, marker: { size: 7 }, hovertemplate: "P = %{x:.2f} mW<br>⟨V<sub>LU</sub>⟩ = %{y:.3f} V<extra>model</extra>" },
          { x: model.map((x) => x.cd.power_mW), y: model.map((x) => (x.r!.stats.LU.sd ?? NaN) * 1e3), yaxis: "y2", type: "scatter", mode: "lines+markers", showlegend: false, line: { color: col, width: 2, dash: "dash" }, marker: { size: 7 }, hovertemplate: "σ = %{y:.1f} mV<extra>model</extra>" },
        );
      }
    });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("v.power") }, anchor: "y2" },
      yaxis: { title: { text: "⟨V<sub>LU</sub>⟩ (V)" }, domain: [0.5, 1] },
      yaxis2: { title: { text: "σ<sub>LU</sub> (mV)" }, domain: [0, 0.42], rangemode: "tozero" },
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
    />
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
      { x: vg, y: nums(data.mean_VLU), type: "scatter", mode: "lines+markers", name: t("vgs.mean"), line: { color: c.sto, width: 2 }, hovertemplate: "V<sub>G</sub> = %{x:.2f} V<br>%{y:.3f} V<extra></extra>" },
      { x: vg, y: nums(data.fold_centre_V), type: "scatter", mode: "lines", name: t("vgs.fold"), line: { color: c.hrs, width: 1.3, dash: "dash" } },
      { x: vg, y: nums(data.sd_VLU_mV), yaxis: "y2", type: "scatter", mode: "lines+markers", name: t("vgs.sd"), line: { color: c.sto, width: 2 }, hovertemplate: "σ = %{y:.1f} mV<extra></extra>" },
      { x: vg, y: nums(data.state_sd_mV), yaxis: "y2", type: "scatter", mode: "lines", name: t("vgs.state"), line: { color: c.categorical[1], width: 1.4, dash: "dash" } },
      { x: vg, y: nums(data.noise_sd_mV), yaxis: "y2", type: "scatter", mode: "lines", name: t("vgs.noise"), line: { color: c.categorical[2], width: 1.4, dash: "dot" } },
      { x: [-1.1], y: [4.354], type: "scatter", mode: "markers", name: "paper: mean peak 4.354 V @ −1.10 V", marker: { color: c.meas, symbol: "star", size: 11 } },
      { x: [-1.25], y: [129.8], yaxis: "y2", type: "scatter", mode: "markers", name: "paper: σ peak 129.8 mV @ −1.25 V", marker: { color: c.meas, symbol: "star-open", size: 11 } },
    ];
    const layout: Partial<Layout> = {
      xaxis: { title: { text: "V<sub>G</sub> (V)" }, anchor: "y2" },
      yaxis: { title: { text: "⟨V<sub>LU</sub>⟩ (V)" }, domain: [0.5, 1] },
      yaxis2: { title: { text: "σ<sub>LU</sub> (mV)" }, domain: [0, 0.42], rangemode: "tozero" },
      margin: { l: 58, r: 16, t: 58, b: 46 },
      legend: { font: { size: 10.5 } },
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
  const { entry, data } = useEntry<ValidationResult>("validation");
  const running = entry?.status === "running" || entry?.status === "queued";
  const pass = data?.checks.filter((c) => c.pass === true).length ?? 0;
  const total = data?.checks.filter((c) => c.pass !== null).length ?? 0;
  return (
    <>
      <section className="panel wide" data-testid="validation">
        <header className="panel-head">
          <div style={{ flex: 1 }}>
            <h2 className="panel-title" style={{ margin: 0, fontSize: 16 }}>
              {t("v.title")}
              {data && <span className={`badge ${pass === total ? "ok" : "err"}`}>{t("v.summary", { pass, total })}</span>}
              {entry?.mock && data && <span className="badge demo">{t("demo")}</span>}
            </h2>
            <div className="panel-desc">{t("v.desc")}</div>
          </div>
        </header>
        <div className="panel-toolbar">
          <button type="button" className="btn primary" onClick={() => void runValidation("fast")} disabled={running} data-testid="val-fast">
            <IconPlay size={12} /> {t("v.fast")}
          </button>
          <button type="button" className="btn" onClick={() => void runValidation("full")} disabled={running} data-testid="val-full">
            <IconPlay size={12} /> {t("v.full")}
          </button>
          {running && (
            <>
              <button type="button" className="btn danger" onClick={() => cancelKey("validation")}>
                <IconStop size={11} /> {t("run.cancel")}
              </button>
              <div style={{ width: 200 }}>
                <Progress value={entry?.progress ?? 0} indeterminate={(entry?.progress ?? 0) < 0.01} />
              </div>
              <span className="small muted mono">{entry?.message}</span>
            </>
          )}
          {data && !running && <span className="small muted mono">{fmtDuration(data.runtime_s)}</span>}
        </div>
        {entry?.status === "error" && (
          <div className="panel-foot">
            <div className="err-box" role="alert">{entry.error}</div>
          </div>
        )}
        {data ? <ChecksTable res={data} /> : !running && <div className="panel-foot"><div className="empty" style={{ height: 120 }}>{t("v.fast")} ▶</div></div>}
        {data?.warnings && data.warnings.length > 0 && (
          <div className="panel-foot">
            <details className="notices"><summary>{t("warnings")} · {data.warnings.length}</summary><ul>{data.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></details>
          </div>
        )}
      </section>
      <div className="section-title">
        <h2>{t("v.figs")}</h2>
      </div>
      <div className="grid">
        <PaperIvFigure />
        <PhotoFigure />
        <VgFigure />
      </div>
    </>
  );
}
