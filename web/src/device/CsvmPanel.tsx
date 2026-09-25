import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import type { CustomCircuitResult } from "../api/circuitCustom";
import { Panel } from "../components/Panel";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { fmtSI, isNum } from "../utils/format";
import { useCurrentKey, useEntry, usePalette } from "./common";
import { csvmPayload, useForcing } from "./forcing";
import { analyzeCsvm, csvmNotes } from "./csvmMetrics";
import { KpiCell } from "./KpiStrip";

export function CsvmPanel() {
  const t = useT();
  const ko = t.lang === "ko";
  const c = usePalette();
  const params = useStore((s) => s.params);
  const mode = useStore((s) => s.mode);
  const settings = useForcing((s) => s.settings);
  const { entry, data } = useEntry<CustomCircuitResult>("device_csvm");
  const payload = useMemo(() => csvmPayload(params, mode, settings), [params, mode, settings]);
  const key = useCurrentKey("circuit", payload);
  const metrics = useMemo(() => data ? analyzeCsvm(data) : null, [data]);
  const stale = !!data && entry?.dataKey !== key;
  const loading = !data && (entry?.status === "running" || entry?.status === "queued");
  const v = (n: number | null | undefined) => isNum(n) ? n.toFixed(3) : "—";
  const value = (n: number | null | undefined, unit: string) => isNum(n) ? unit : undefined;
  const freq = metrics?.frequency_Hz;
  const freqScale = isNum(freq) && freq >= 1e3 ? 1e-3 : 1;
  const tooltip = ko ? "시작 구간과 첫 주기를 뺀, 계산한 드레인 전압 파형의 주기별 최댓값·최솟값 평균" : "Per-cycle maxima and minima of the computed drain-voltage waveform, averaged after the start-up and the first cycle";
  const notes = useMemo(() => csvmNotes(data?.warnings, t.lang), [data, t.lang]);
  const plot = useMemo(() => {
    const run = data?.runs[0];
    const drain = run?.signals.find((s) => s.key === "V(drain)");
    if (!run || !drain) return undefined;
    const traces: Data[] = [{
      x: run.t.map((v) => v == null ? null : v * 1e3), y: drain.values,
      type: "scatter", mode: "lines", name: "V<sub>D</sub>",
      line: { color: c.hrs, width: 2 },
      hovertemplate: "t = %{x:.5g} ms<br>V<sub>D</sub> = %{y:.5g} V<extra></extra>",
    }];
    const layout: Partial<Layout> = {
      xaxis: { title: { text: t("axis.time", { u: "ms" }) }, autorange: true, zeroline: false },
      yaxis: { title: { text: t("axis.vd") }, autorange: true, zeroline: false },
      showlegend: false, margin: { l: 58, r: 18, t: 22, b: 48 },
    };
    return { data: traces, layout };
  }, [data, c, t]);
  const foot = data && metrics ? <span data-testid="csvm-status">
    {t.l(metrics.reason)}
    {mode === "stochastic" && ` · ${ko ? "확률 궤적 1개" : "one stochastic trace"}`}
  </span> : undefined;
  return <>
    <div className="answer-row">
      <div className={`answer-bar${stale ? " stale" : ""}`} data-testid="csvm-kpis" role="group" aria-label={ko ? "CSVM 결과" : "CSVM results"}>
        <KpiCell id="vtop" label={ko ? "상단 전압" : "Peak"} sym={<>V<sub>top</sub></>} value={v(metrics?.vTop_V)} unit={value(metrics?.vTop_V, "V")} color="var(--hrs)" loading={loading} title={tooltip} />
        <KpiCell id="vbottom" label={ko ? "하단 전압" : "Trough"} sym={<>V<sub>bottom</sub></>} value={v(metrics?.vBottom_V)} unit={value(metrics?.vBottom_V, "V")} color="var(--lrs)" loading={loading} title={tooltip} />
        <KpiCell id="frequency" label={ko ? "주파수" : "Frequency"} sym={<>f</>} value={isNum(freq) ? Number((freq * freqScale).toPrecision(4)).toString() : "—"} unit={value(freq, freqScale === 1 ? "Hz" : "kHz")} color="var(--text)" loading={loading} sub={isNum(metrics?.period_s) ? `T = ${fmtSI(metrics!.period_s!, "s", 4)}` : undefined} title={ko ? "시작 구간 뒤의 완전한 주기 2개 이상에서 구한 1 / 평균 주기" : "1 / mean period over at least two complete cycles after the start-up"} />
        {stale && <span className="badge stale ans-badge">{ko ? "변경됨" : "Changed"}</span>}
      </div>
    </div>
    <div className="device-sweep" data-testid="panels-csvm">
      <Panel id="csvm" primary title={ko ? "드레인 전압 V_D(t)" : "Drain voltage V_D(t)"} desc={ko ? "전류원으로 드레인을 충전합니다 (C_drain과 병렬)" : "A current source charges the drain (in parallel with C_drain)"} entry={entry} hasData={!!plot} currentKey={key} plot={plot} csvName="csvm_transient" foot={foot} warnings={notes} />
    </div>
  </>;
}
