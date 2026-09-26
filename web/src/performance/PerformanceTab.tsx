import { useEffect, useState } from "react";
import { useConnection } from "../api/connection";
import { useStore } from "../state/store";
import { Progress } from "../components/Panel";
import { benchmarkCatalog, comparisonRows, pairedSpeedup, type BenchmarkCase } from "./catalog";
import { computeLocation, secondsLabel } from "./format";
import { calibratePerformance, refreshPerformance, usePerformance } from "./store";
import "./performance.css";

function Timing({ sample, ko }: { sample?: BenchmarkCase; ko: boolean }) {
  if (!sample) return <span className="perf-muted">—</span>;
  if (!sample.supported) return <span className="perf-unavailable" title={sample.unsupported_reason}>{ko ? "미지원" : "Unsupported"}</span>;
  if (!sample.timings) return <span className="perf-muted">{ko ? "미측정" : "Not measured"}</span>;
  const time = sample.timings;
  return <span className={`perf-time ${sample.model}`} title={`${ko ? "최소–최대" : "Min–max"}: ${secondsLabel(time.min_s)} – ${secondsLabel(time.max_s)} · n = ${time.repeats}`}>
    <strong>{secondsLabel(time.median_s)}</strong>
    <small>{secondsLabel(time.min_s)} – {secondsLabel(time.max_s)}</small>
  </span>;
}

export function PerformanceTab() {
  const ko = useStore((s) => s.lang === "ko");
  const online = useStore((s) => s.backend === "online");
  const endpoint = useConnection((s) => s.endpoint);
  const { data, status, error, calibrating, calibrationProgress, calibrationMessage } = usePerformance();
  const [all, setAll] = useState(false);
  const [condition, setCondition] = useState<string | null>(null);
  const catalog = benchmarkCatalog(data);
  const rows = comparisonRows(catalog?.cases ?? []).filter((r) => all || r.group === "standard");
  const host = data?.host;
  const calibration = data?.calibration;
  const measured = !!calibration && (calibration.status === "calibrated" || calibration.status === "ready" || calibration.calibrated === true || !!calibration.measured_at || !!calibration.completed_at);
  const samples = (Array.isArray(calibration?.samples) ? calibration.samples : []) as {case_id:string; model:string; family:string; warm_s:number; first_s:number; warm_samples_s:number[]}[];
  const env = catalog?.environment;
  const L = (a: string, b: string) => ko ? a : b;
  useEffect(() => { void refreshPerformance(); }, []);
  return <div className="performance-page" data-testid="performance-page">
    <header className="perf-heading">
      <div><div className="workspace-eyebrow">COMPUTE PERFORMANCE</div><h1>{L("성능", "Performance")}</h1><p>{L("모델별 실행 시간과 이 환경의 예상 시간", "Model runtimes and estimates for your compute host")}</p></div>
      <a className="perf-guide-link" href={`${import.meta.env.BASE_URL}docs/performance.html`} target="_blank" rel="noreferrer">{L("측정 방법", "Methodology")} <span aria-hidden>↗</span></a>
    </header>
    <section className="perf-host-card" aria-label={L("계산 환경", "Compute host")}>
      <div className="perf-host-icon" aria-hidden><svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4"><rect x="4" y="3" width="16" height="13" rx="2"/><path d="M8 21h8M12 16v5M8 8h3m-3 3h8"/></svg></div>
      <div className="perf-host-copy"><span className="perf-kicker">{L("시간 예측 기준", "Estimate basis")}</span><h2>{host ? computeLocation(endpoint, ko) : L("기준 환경", "Reference host")}
        <span className={`perf-state${measured ? " measured" : ""}`}>{measured ? L("측정 완료", "Measured") : L("보정 전", "Not calibrated")}</span></h2>
        <p>{host ? `${host.os} · ${host.architecture} · Python ${host.python} · ${host.workers} ${L("워커", "workers")}` : L("연결 후 짧은 실제 계산으로 실행 시간을 보정합니다.", "Connect and run a short solver check to calibrate runtime estimates.")}</p>
      </div>
      <button type="button" className="btn perf-calibrate" disabled={!online || calibrating} onClick={() => void calibratePerformance()} data-testid="performance-calibrate">
        {calibrating ? L("측정 중…", "Measuring…") : measured ? L("다시 측정", "Measure again") : L("이 환경 측정", "Measure this host")}
      </button>
      {calibrating && <div className="perf-calibration-progress" role="status"><Progress value={calibrationProgress} indeterminate={calibrationProgress < .01}/><span><span title={calibrationMessage}>{L("실제 계산으로 측정 중 · 최초 준비 시간은 별도", "Running solver checks · first-run setup is separate")}</span></span></div>}
      {!calibrating && <div className="perf-host-note">{L("로컬 실행은 이 PC, 서버 연결은 해당 서버에서 측정합니다. 완료된 계산으로 예측을 갱신합니다.", "Measured on this PC for local runs, or on the connected server. Completed calculations refine estimates.")}</div>}
      {samples.length > 0 && <details className="perf-host-samples"><summary>{L("이 환경 측정 기록", "Host measurement records")} · {samples.length}</summary>
        <p>{L("측정한 모델·해석 종류에만 보정을 적용합니다. 다른 해석은 기준 환경 추정치로 표시합니다.", "Calibration applies to measured model and analysis families. Other analyses retain reference-host estimates.")}</p>
        <table><thead><tr><th>{L("모델 · 해석", "Model · analysis")}</th><th>{L("반복 계산", "Repeated compute")}</th><th>{L("초기 호출", "First call")}</th></tr></thead><tbody>{samples.map((sample) => <tr key={sample.case_id}><td>{sample.model === "simple" ? "Simple" : "Detailed"} · {sample.family === "idvd" ? "ID–VD" : sample.family === "circuit_csvm" ? "CSVM" : sample.family}</td><td>{secondsLabel(sample.warm_s)}</td><td>{secondsLabel(sample.first_s)}</td></tr>)}</tbody></table>
      </details>}
    </section>
    {error && <div className="perf-error" role="alert">{L("성능 정보를 가져오지 못했습니다.", "Performance data could not be loaded.")} <button type="button" onClick={() => void refreshPerformance(true)}>{L("다시 시도", "Retry")}</button><details><summary>{L("상세", "Details")}</summary>{error}</details></div>}
    <section className="perf-benchmark" aria-labelledby="perf-table-title">
      <div className="perf-table-heading"><div><h2 id="perf-table-title">{L("모델 벤치마크", "Model benchmarks")} <span className="perf-reference-tag">{L("기준 환경 측정", "Reference host")}</span></h2><p>{L("동일 입력 · 반복 실행 중앙값 · 결과 캐시 제외", "Same inputs · median of repeated runs · result cache excluded")}</p></div>
        <div className="perf-filter" role="group" aria-label={L("벤치마크 범위", "Benchmark scope")}><button type="button" aria-pressed={!all} onClick={() => setAll(false)}>{L("주요 해석", "Main")}</button><button type="button" aria-pressed={all} onClick={() => setAll(true)}>{L("전체", "All")}</button></div>
      </div>
      {catalog ? <div className="perf-table-scroll"><table className="perf-table"><thead><tr><th scope="col">{L("시뮬레이션", "Simulation")}</th><th scope="col"><span className="perf-model-dot detailed"/>Detailed</th><th scope="col"><span className="perf-model-dot simple"/>Simple</th><th scope="col">{L("속도 향상", "Speedup")}</th></tr></thead>
        {rows.map((row) => {
          const ratio = pairedSpeedup(row);
          const sample = row.detailed ?? row.simple ?? row.common ?? row.mixed!;
          const expanded = condition === row.id;
          return <tbody key={row.id} className="perf-row-group"><tr>
            <th scope="row"><button type="button" className="perf-case-label" aria-expanded={expanded} onClick={() => setCondition(expanded ? null : row.id)}>{row.label[ko ? "ko" : "en"]}<span aria-hidden className="perf-info">i</span></button></th>
            {row.common || row.mixed ? <td colSpan={2}><span className="perf-common-label">{row.mixed ? "Detailed + Simple" : L("공통", "Shared")}</span><Timing sample={row.common ?? row.mixed} ko={ko}/></td> : <><td><Timing sample={row.detailed} ko={ko}/></td><td><Timing sample={row.simple} ko={ko}/></td></>}
            <td>{ratio !== null ? <span className={`perf-speed${ratio >= 1 ? " faster" : ""}`}>{ratio.toFixed(1)}×</span> : <span className="perf-muted">—</span>}</td>
          </tr>{expanded && <tr className="perf-condition"><td colSpan={4}><div><span>{L("계산 종류", "Compute kind")}: {sample.kind} · {sample.timings ? `n = ${sample.timings.repeats}` : L("미지원", "Unsupported")}</span>
            <p>{L("반복 실행 범위", "Repeated-run range")}: {[row.detailed, row.simple, row.common, row.mixed].filter((value): value is BenchmarkCase => !!value?.timings).map(value => `${value.model === "simple" ? "Simple" : value.model === "detailed" ? "Detailed" : value.model === "mixed" ? "Mixed" : L("공통", "Shared")} ${secondsLabel(value.timings!.min_s)} – ${secondsLabel(value.timings!.max_s)}`).join(" · ") || "—"}</p>
            {sample.kind === "hazard" || sample.kind === "vg_curve_stochastic" || sample.kind === "sweep_mc" ? <p>{L("반복 시간은 이미 만든 확률 모델 표를 재사용합니다. 새로운 조건의 표 준비 시간은 추가됩니다.", "Repeated timings reuse prepared stochastic model tables. New conditions require additional table preparation.")}</p> : null}
            {!!sample.notes?.length && <p>{sample.notes.join(" ")}</p>}
            <details><summary>{L("입력 조건", "Input settings")}</summary><pre>{JSON.stringify(sample.payload, null, 2)}</pre></details>
            <p>{sample.group === "holdout" ? L("예측 확인용 추가 조건 · ", "Additional prediction-check condition · ") : ""}{L("초기 호출", "First call")}: {row.common || row.mixed ? secondsLabel(sample.timings?.first_call_s) : <>Detailed {secondsLabel(row.detailed?.timings?.first_call_s)} · Simple {secondsLabel(row.simple?.timings?.first_call_s)}</>}. {L("기존 컴파일 캐시를 사용하는 초기 호출이며, 새 설치의 최초 준비 시간과 다릅니다.", "Uses an existing compiler cache; this is not fresh-install startup time.")}</p>
          </div></td></tr>}</tbody>;
        })}</table></div> : <div className="perf-empty">{status === "loading" ? L("측정 기록을 불러오는 중…", "Loading benchmark records…") : L("측정 기록이 아직 없습니다.", "No benchmark records available.")}</div>}
      <div className="perf-table-foot"><span>{L("반복 실행 범위 · 항목을 눌러 상세 보기", "Repeat ranges · select a row for details")}</span><span>{L("속도 비교는 모델의 정확도 비교가 아닙니다.", "Runtime comparisons do not compare model accuracy.")}</span></div>
    </section>
    <details className="perf-method"><summary>{L("기준 환경과 예측 범위", "Reference environment and prediction limits")}</summary>
      <div className="perf-method-content"><p>{env ? `${env.cpu_model ?? "CPU"} · ${env.platform ?? ""} · Python ${env.python ?? ""} · Numba ${env.numba ?? ""}` : "—"}</p>
        <p>{L("반복 계산 시간에는 모델 풀이가 포함됩니다. 결과 재사용, 첫 컴파일, 작업 대기는 별도로 취급합니다. 파라미터·회로·적응 스텝에 따라 실제 시간은 달라질 수 있습니다.", "Repeated-run timings include model evaluation. Result reuse, first compilation, and queueing are treated separately. Parameters, circuit topology, and adaptive steps can change actual runtime.")}</p>
        <p>{L("보정 전 실행 버튼의 숫자는 기준 환경 추정치입니다. 이 환경 측정 후에는 지원되는 해석 종류를 보정하고, 실제 완료된 계산으로 갱신합니다.", "Before calibration, the Run button shows reference-host estimates. Measuring this host calibrates supported analysis families; completed runs then refine predictions.")}</p>
        <span>{catalog?.generated_at ? `${L("측정 기록", "Benchmark record")}: ${catalog.generated_at.slice(0, 10)}` : ""}</span>
      </div>
    </details>
  </div>;
}
