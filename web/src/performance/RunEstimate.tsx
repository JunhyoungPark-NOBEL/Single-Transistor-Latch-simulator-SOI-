import { useEffect, useMemo, useState } from "react";
import { request } from "../api/client";
import { useConnection } from "../api/connection";
import { useCircuitView } from "../circuit/view";
import { useForcing } from "../device/forcing";
import { useLayout } from "../state/layout";
import { useStore } from "../state/store";
import { useSch } from "../schematic/store";
import { hasErrors, runErc } from "../schematic/erc";
import { buildRequest } from "../schematic/netlist";
import { extractNets } from "../schematic/nets";
import { canonical } from "../utils/object";
import { computeLocation, rangeLabel } from "./format";
import { currentRunJobs } from "./runGroup";
import { performanceConfig, usePerformance } from "./store";
import type { EstimateJob, PerformanceEstimate } from "./types";
import "./performance.css";

export function RunEstimate() {
  const s = useStore();
  const ctx = useForcing();
  const layout = useLayout((v) => v.layout);
  const circuitView = useCircuitView((v) => v.view);
  const doc = useSch((v) => v.doc);
  const traces = useSch((v) => v.traces);
  const endpoint = useConnection((v) => v.endpoint);
  const connectionStatus = useConnection((v) => v.status);
  const revision = usePerformance((v) => v.revision);
  const [result, setResult] = useState<{ key: string; data?: PerformanceEstimate } | null>(null);
  const ko = s.lang === "ko";
  const jobs = useMemo<EstimateJob[]>(() => {
    try {
      let customRequest;
      if (s.tab === "circuit" && circuitView === "schematic") {
        const nets = extractNets(doc);
        if (hasErrors(runErc(doc, nets))) return [];
        customRequest = buildRequest(doc, nets, s.mode, traces);
      }
      return currentRunJobs(s, { layout, forcing: ctx.forcing, csvm: ctx.settings, circuitView, customRequest });
    } catch { return []; }
  }, [s.tab, s.params, s.mode, s.results, s.vgRange, s.cbVd, layout, ctx.forcing, ctx.settings, circuitView, doc, traces]);
  const jobKey = canonical(jobs);
  const key = `${endpoint}|${connectionStatus}|${s.health?.version}|${revision}|${jobKey}`;
  useEffect(() => {
    let active = true;
    if (!jobs.length || s.backend !== "online") return;
    const config = performanceConfig();
    const timer = setTimeout(() => {
      void request<PerformanceEstimate>("POST", "/api/performance/estimate", { jobs: JSON.parse(jobKey) }, 15000, config)
        .then((data) => { if (active) setResult({ key, data }); })
        .catch(() => { if (active) setResult({ key }); });
    }, 350);
    return () => { active = false; clearTimeout(timer); };
  }, [key, jobKey, jobs.length, s.backend]);
  const data = result?.key === key ? result.data : undefined;
  const unsupported = data?.items.some((item) => !item.supported);
  const usable = data && !unsupported && data.total?.seconds !== null;
  const source = data?.source === "reference" ? (ko ? "기준 환경" : "Reference host") : computeLocation(endpoint, ko);
  const prefix = data?.cached ? (ko ? "저장 결과" : "Cached result") : source;
  const label = usable ? data.cached ? (ko ? "저장 결과 사용" : "Saved result available") : `${prefix} · ${ko ? "예상" : "est."} ${rangeLabel(data.total)}`
    : !jobs.length ? (ko ? "예상 — · 설정 확인" : "Estimate — · check settings")
      : s.backend !== "online" ? (ko ? "예상 — · 서버 연결 필요" : "Estimate — · connect server")
        : result?.key === key ? (ko ? "예상 — · 측정 범위 밖" : "Estimate — · unavailable")
          : (ko ? "예상 시간 확인 중…" : "Estimating…");
  const detail = [
    ko ? "현재 실행에 포함되는 계산의 예상 시간입니다. 전송·화면 표시 시간은 제외합니다." : "Estimate for the current Run action, excluding transfer and rendering.",
    data?.source === "reference" ? (ko ? "성능 탭에서 이 환경을 측정하면 계산 컴퓨터에 맞게 보정됩니다." : "Measure this compute host in Performance to calibrate the estimate.") : "",
    data?.queue?.unknown ? (ko ? "작업 대기 시간은 추가될 수 있습니다." : "Queue wait may add to this estimate.") : "",
    data?.setup?.unknown ? (ko ? "첫 실행 준비 시간은 별도입니다." : "First-run setup is additional.") : "",
    ...(data?.warnings ?? []),
  ].filter(Boolean).join(" ");
  return <div className="run-estimate" data-testid="run-estimate">
    <button type="button" className="run-estimate-link" title={detail} onClick={() => s.setTab("performance")} aria-label={`${label}. ${ko ? "성능 보기" : "View performance"}`}>
      <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.25" aria-hidden><circle cx="8" cy="8" r="5.7"/><path d="M8 4.5V8l2.5 1.5"/></svg>
      <span>{label}</span>
    </button>
    {usable && data.queue?.unknown === true && <small title={detail}>{ko ? "+ 준비·대기" : "+ setup / queue"}</small>}
    {usable && data.setup?.unknown && <small title={detail}>{ko ? "+ 첫 실행 준비" : "+ first-run setup"}</small>}
  </div>;
}
