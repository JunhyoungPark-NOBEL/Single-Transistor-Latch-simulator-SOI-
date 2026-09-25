// Comparator firing statistics of a schematic run (§6.3 `comparators`): for every comparator a run × pulse
// raster (fired = the output went high within that pulse period) above the firing probability per pulse
// (binomial ± 1 SE over the runs), with P(fire), the lag-1 autocorrelation of the bit stream and the number
// of observed pulses in the panel header. Deterministic runs show one raster row.
import type { Data, Layout } from "plotly.js";
import { useMemo } from "react";
import type { ComparatorStats, CustomCircuitResult } from "../api/circuitCustom";
import { Panel } from "../components/Panel";
import { usePalette } from "../device/common";
import { useT } from "../i18n";
import { fmtInt, isNum } from "../utils/format";

export function CmpPlot({ cmp, stale }: { cmp: ComparatorStats; stale: boolean }) {
  const t = useT();
  const c = usePalette();
  const nW = cmp.bits[0]?.length ?? 0;
  const plot = useMemo(() => {
    if (!nW) return undefined;
    const xs = Array.from({ length: nW }, (_, k) => k + 1);
    const runs = cmp.bits.map((_, r) => r);
    const z = cmp.bits.map((row) => row.map((v) => (v == null ? null : v)));
    const p = cmp.p_fire_window.slice(0, nW).map((v) => (isNum(v) ? v : null));
    const e = cmp.p_fire_window_err.slice(0, nW).map((v) => (isNum(v) ? v : 0));
    const data: Data[] = [
      {
        type: "heatmap",
        x: xs,
        y: runs,
        z,
        zmin: 0,
        zmax: 1,
        colorscale: [
          [0, c.grid],
          [0.5, c.grid],
          [0.5, c.lrs],
          [1, c.lrs],
        ],
        showscale: false,
        xgap: nW <= 60 ? 1 : 0,
        ygap: runs.length <= 40 ? 1 : 0,
        hovertemplate: `${t("schematic.res.pulse")} %{x} · ${t("schematic.res.run")} %{y}: %{z}<extra>${cmp.name}</extra>`,
        xaxis: "x",
        yaxis: "y",
      } as Data,
      {
        type: "bar",
        x: xs,
        y: p,
        error_y: { type: "data", array: e, visible: runs.length > 1, color: c.muted, thickness: 1 },
        marker: { color: c.sto },
        opacity: 0.85,
        name: t("schematic.res.cmpP"),
        hovertemplate: `${t("schematic.res.pulse")} %{x}: P = %{y:.2f}<extra>${cmp.name}</extra>`,
        xaxis: "x",
        yaxis: "y2",
      } as Data,
    ];
    if (isNum(cmp.p_fire)) data.push({ type: "scatter", mode: "lines", x: [0.5, nW + 0.5], y: [cmp.p_fire, cmp.p_fire], line: { color: c.text2, width: 1, dash: "dot" }, hoverinfo: "skip", xaxis: "x", yaxis: "y2" } as Data);
    const layout: Partial<Layout> = {
      margin: { l: 60, r: 16, t: 14, b: 44 },
      showlegend: false,
      xaxis: { title: { text: t("schematic.res.pulse") }, range: [0.5, nW + 0.5], anchor: "y2" as never },
      yaxis: { domain: [0.45, 1], title: { text: t("schematic.res.run"), font: { size: 11 } }, autorange: "reversed", tickformat: "d", dtick: runs.length > 12 ? undefined : 1 },
      yaxis2: { domain: [0, 0.36], range: [0, 1.05], title: { text: t("schematic.res.cmpP"), font: { size: 11 } } },
    };
    return { data, layout, className: "plot" };
  }, [cmp, nW, c, t]);
  const head = [
    `${t("schematic.res.cmpP")} = ${isNum(cmp.p_fire) ? cmp.p_fire.toFixed(3) : "—"}`,
    `${t("schematic.res.cmpLag")} = ${isNum(cmp.lag1) ? cmp.lag1.toFixed(3) : "—"}`,
    `${t("schematic.res.cmpBits")} ${fmtInt(cmp.n_bits)}`,
    cmp.window_source ? t("schematic.res.cmpWin", { src: cmp.window_source }) : "",
  ].filter(Boolean);
  return (
    <Panel
      id={`sch-cmp-${cmp.name}`}
      wide
      title={`${t("schematic.res.cmp")} · ${cmp.name}`}
      desc={t("schematic.res.cmpDesc")}
      topic="stochastic-events"
      hasData={!!plot}
      stale={stale}
      csvName={`schematic_${cmp.name}_firing`}
      plot={plot}
      empty={<span>{t("schematic.res.cmpNoWin")}</span>}
      toolbar={
        <span className="small muted mono" data-testid={`sch-cmp-stats-${cmp.name}`}>
          {head.join(" · ")}
        </span>
      }
    />
  );
}

/** One panel per comparator of the circuit (nothing when the result has none). */
export function ComparatorPanels({ res, stale }: { res: CustomCircuitResult; stale: boolean }) {
  const cmps = res.comparators ?? [];
  if (!cmps.length) return null;
  return (
    <>
      {cmps.map((cmp) => (
        <CmpPlot key={cmp.name} cmp={cmp} stale={stale} />
      ))}
    </>
  );
}
