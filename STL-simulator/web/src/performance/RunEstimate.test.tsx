// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { request } from "../api/client";
import { useConnection } from "../api/connection";
import { useStore } from "../state/store";
import { useForcing } from "../device/forcing";
import { useLayout } from "../state/layout";
import { RunEstimate } from "./RunEstimate";
import { usePerformance } from "./store";
import type { PerformanceEstimate } from "./types";

vi.mock("../api/client", () => ({ request: vi.fn(), runJob: vi.fn() }));
vi.mock("../state/runner", () => ({ getBackend: () => ({ isMock: false }), setCircuitRunOverride: vi.fn() }));
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement;
let root: Root;
function estimate(seconds: number): PerformanceEstimate {
  return { host: {id:"host",instance_id:"instance",engine_version:"v1",label:"host",os:"Linux",architecture:"x64",python:"3",workers:1,cpu_count:2}, items:[{key:"branches",kind:"branches",model:"detailed",family:"idvd",supported:true,cached:false,estimate:{seconds,low_s:seconds,high_s:seconds},source:"calibrated",confidence:"medium"}],total:{seconds,low_s:seconds,high_s:seconds},compute:{seconds,low_s:seconds,high_s:seconds},setup:{seconds:null,low_s:0,high_s:null,unknown:true},queue:{unknown:false},cached:false,confidence:"medium",source:"calibrated",warnings:[] };
}
beforeEach(() => {
  vi.useFakeTimers(); vi.clearAllMocks();
  useStore.setState({tab:"device",lang:"ko",backend:"online",health:{ok:true,version:"v1"},results:{}});
  useConnection.setState({endpoint:"http://127.0.0.1:8000",status:"online"});
  useForcing.setState({forcing:"vscm"}); useLayout.setState({layout:"simple"});
  usePerformance.setState({revision:0});
  host=document.createElement("div"); document.body.appendChild(host); root=createRoot(host);
});
afterEach(() => { act(()=>root.unmount()); host.remove(); vi.useRealTimers(); });
const render = () => act(()=>root.render(<RunEstimate/>));
const tick = async () => { await act(async ()=>vi.advanceTimersByTimeAsync(360)); };

describe("compute-host estimate isolation", () => {
  it("discards the previous server response even if it finishes last", async () => {
    let oldDone!: (x: PerformanceEstimate)=>void;
    vi.mocked(request).mockImplementationOnce(()=>new Promise(r=>{oldDone=r;})).mockResolvedValueOnce(estimate(2));
    render(); await tick();
    act(()=>useConnection.setState({endpoint:"https://lab.kaist.ac.kr"}));
    expect(host.textContent).not.toContain("2.0 s"); await tick();
    expect(host.textContent).toContain("계산 서버 · 예상 2.0 s");
    await act(async ()=>oldDone(estimate(9)));
    expect(host.textContent).not.toContain("9.0 s");
    expect(vi.mocked(request).mock.calls[0][4]).toMatchObject({endpoint:"http://127.0.0.1:8000"});
    expect(vi.mocked(request).mock.calls[1][4]).toMatchObject({endpoint:"https://lab.kaist.ac.kr"});
  });
  it("invalidates previous-version estimates and names unknown queue time", async () => {
    vi.mocked(request).mockResolvedValueOnce(estimate(2));
    render(); await tick(); expect(host.textContent).toContain("2.0 s");
    const next=estimate(4); next.queue={unknown:true}; vi.mocked(request).mockResolvedValueOnce(next);
    act(()=>useStore.setState({health:{ok:true,version:"v2"}}));
    expect(host.textContent).not.toContain("2.0 s"); await tick();
    expect(host.textContent).toContain("+ 준비·대기"); expect(host.textContent).toContain("+ 첫 실행 준비");
  });
  it("does not present a result cache as zero-second compute performance", async () => {
    const cached=estimate(0); cached.cached=true; cached.setup.unknown=false;
    vi.mocked(request).mockResolvedValueOnce(cached); render(); await tick();
    expect(host.textContent).toContain("저장 결과 사용"); expect(host.textContent).not.toContain("0 s");
  });
  it("hides unsupported estimates rather than extrapolating a number", async () => {
    const value=estimate(2); value.items[0].supported=false;
    vi.mocked(request).mockResolvedValueOnce(value); render(); await tick();
    expect(host.textContent).toContain("측정 범위 밖"); expect(host.textContent).not.toContain("2.0 s");
  });
});
