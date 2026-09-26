import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import type { CustomCircuitResult } from "../api/circuitCustom";
import { Panel } from "../components/Panel";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { fmtSI, isNum } from "../utils/format";
import { useCurrentKey, useEntry, usePalette } from "./common";
import { csvmPayload, useForcing } from "./forcing";
import { analyzeCsvm } from "./csvmMetrics";
import { SubText } from "../plots/SubText";
import { KpiCell } from "./KpiStrip";

export function CsvmPanel() {
  const t = useT();
  const selectedSimple = useStore((s) => s.params.device.model === "simple");
  const ko = t.lang === "ko";
  const c = usePalette();
  const params = useStore((s) => s.params);
  const mode = useStore((s) => s.mode);
  const settings = useForcing((s) => s.settings);
  const { entry, data } = useEntry<CustomCircuitResult>("device_csvm");
  const plottedDevice = data?.elements?.find((e) => e.name === "X1")?.device as { model?: string } | undefined;
  const simple = data ? plottedDevice?.model === "simple" : selectedSimple;
  const payload = useMemo(() => csvmPayload(params, mode, settings), [params, mode, settings]);
  const key = useCurrentKey("circuit", payload);
  const metrics = useMemo(() => data ? analyzeCsvm(data) : null, [data]);
  const stale = !!data && entry?.dataKey !== key;
  const loading = !data && (entry?.status === "running" || entry?.status === "queued");
  const v = (n: number | null | undefined) => isNum(n) ? n.toFixed(3) : "—";
  const value = (n: number | null | undefined, unit: string) => isNum(n) ? unit : undefined;
  const freq = metrics?.frequency_Hz;
  const freqScale = isNum(freq) && freq >= 1e3 ? 1e-3 : 1;
  const tooltip = ko ? "시작 구간과 첫 주기를 제외한 실제 VD(t)의 주기별 극값 평균" : "Cycle extrema averaged from actual VD(t), excluding startup and the first cycle";
  const plot = useMemo(() => {
    const run = data?.runs[0];
    const drain = run?.signals.find((s) => s.key === "V(drain)");
    const body = run?.signals.find((s) => s.key === "X1.vb");
    if (!run || !drain) return undefined;
    const time = run.t.map((v) => v == null ? null : v * 1e3);
    const traces: Data[] = [{
      x: time, y: drain.values,
      type: "scatter", mode: "lines", name: "V<sub>D</sub>",
      line: { color: c.hrs, width: 2 },
      hovertemplate: "t = %{x:.5g} ms<br>V<sub>D</sub> = %{y:.5g} V<extra></extra>",
    }];
    if (body) traces.push({
      x: time, y: body.values, yaxis: "y2",
      type: "scatter", mode: "lines", name: "V<sub>B</sub>",
      line: { color: c.lrs, width: 2 },
      hovertemplate: "t = %{x:.5g} ms<br>V<sub>B</sub> = %{y:.5g} V<extra></extra>",
    });
    const layout: Partial<Layout> = {
      xaxis: { title: { text: ko ? "시간 (ms)" : "Time (ms)" }, anchor: body ? "y2" : "y", autorange: true, zeroline: false },
      yaxis: { title: { text: "V<sub>D</sub> (V)" }, domain: body ? [0.43, 1] : [0, 1], autorange: true, zeroline: false },
      ...(body ? { yaxis2: { title: { text: "V<sub>B</sub> (V)" }, domain: [0, 0.27] as [number, number], autorange: true, zeroline: false } } : {}),
      showlegend: !!body, legend: { orientation: "h", x: 0, y: 1.04 },
      margin: { l: 62, r: 18, t: body ? 42 : 22, b: 48 },
    };
    return { data: traces, layout };
  }, [data, c, ko]);
  const foot = data && metrics ? <span data-testid="csvm-status">
    {t.l(metrics.reason)}
    {mode === "stochastic" && ` · ${ko ? "단일 확률 궤적" : "Single stochastic trace"}`}
  </span> : undefined;
  return <>
    <div className="answer-row">
      <div className={`answer-bar${stale ? " stale" : ""}`} data-testid="csvm-kpis" role="group" aria-label="CSVM results">
        <KpiCell id="vtop" label={ko ? "상단" : "Top"} sym={<>V<sub>top</sub></>} value={v(metrics?.vTop_V)} unit={value(metrics?.vTop_V, "V")} color="var(--hrs)" loading={loading} title={tooltip} />
        <KpiCell id="vbottom" label={ko ? "하단" : "Bottom"} sym={<>V<sub>bottom</sub></>} value={v(metrics?.vBottom_V)} unit={value(metrics?.vBottom_V, "V")} color="var(--lrs)" loading={loading} title={tooltip} />
        <KpiCell id="frequency" label={ko ? "주파수" : "Frequency"} sym={<>f</>} value={isNum(freq) ? Number((freq * freqScale).toPrecision(4)).toString() : "—"} unit={value(freq, freqScale === 1 ? "Hz" : "kHz")} color="var(--text)" loading={loading} sub={isNum(metrics?.period_s) ? <SubText text={`T = ${fmtSI(metrics!.period_s!, "s", 4)}`} /> : undefined} title={ko ? "시작 구간 이후 완전한 주기 2개 이상에서 1 / 평균 주기" : "1 / mean period from at least two complete post-startup cycles"} />
        {stale && <span className="badge stale ans-badge">{ko ? "변경됨" : "Changed"}</span>}
      </div>
    </div>
    <div className="device-sweep" data-testid="panels-csvm">
      <Panel id="csvm" primary title="VD(t) · VB(t)" desc={simple ? (ko ? "전류 구동 · VB는 저장 전하의 등가 전위 w = u + ID·RLRS입니다. B 접점 전압 u와 구분합니다." : "Current forcing · VB is the reservoir potential w = u + ID·RLRS, distinct from B-contact voltage u.") : ko ? "전류 구동 · VB는 소스 기준 내부 정전기 전위이며, 회로의 B 접점 전압과 구분합니다." : "Current forcing · VB is the internal electrostatic potential relative to source, distinct from the B contact voltage."} entry={entry} hasData={!!plot} currentKey={key} plot={plot} csvName="csvm_transient" foot={foot} warnings={data?.warnings} />
    </div>
  </>;
}
