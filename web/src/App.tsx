// App shell: header + context strip (hint, backend status pill, credits) + sidebar/main layout + floating
// Details window. There is no full-width banner: demo / offline / snapshot is the pill in the strip.
import { useEffect } from "react";
import { PerformanceTab } from "./performance/PerformanceTab";
import { startPerformanceTracking } from "./performance/store";
import { CircuitTab } from "./circuit/CircuitTab";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ContextStrip, Header } from "./components/Header";
import { DeviceShelf } from "./devices/DeviceShelf";
import { DeviceTab } from "./device/DeviceTab";
import { translate } from "./i18n";
import { PhysicsTab } from "./physics/PhysicsTab";
import { PhysicsWindow } from "./physics/PhysicsWindow";
import { Sidebar } from "./sidebar/Sidebar";
import { runCurrent, startAutoRun, startHealthPolling } from "./state/runner";
import { useStore } from "./state/store";
import { ValidationTab } from "./validation/ValidationTab";
import { usesGeometryModel } from "./params/geometry";

/** A short workspace label, with the model's scope available before a run. */
function WorkspaceHeading() {
  const tab = useStore((s) => s.tab);
  const lang = useStore((s) => s.lang);
  const device = useStore((s) => s.params.device);
  const ko = lang === "ko";
  const extended = usesGeometryModel(device);
  if (tab !== "device" && tab !== "circuit") return null;
  return <div className="workspace-heading">
    <div className="workspace-heading-copy">
      <div className="workspace-eyebrow">{tab === "device" ? "DEVICE WORKSPACE" : "CIRCUIT WORKSPACE"}</div>
      <h1>{tab === "device" ? (ko ? "소자 특성" : "Device characterization") : (ko ? "회로 설계" : "Circuit design")}</h1>
      <p>{tab === "device" ? (ko ? "FDSOI · 전압·전류 구동 해석" : "FDSOI · voltage and current forcing") : (ko ? "소자 배치부터 과도 응답까지" : "Place, connect, and simulate")}</p>
    </div>
    {tab === "device" && <a className={`model-scope-link${extended ? " extended" : ""}`} href={`${import.meta.env.BASE_URL}docs/geometry-model.html`} target="_blank" rel="noreferrer" data-testid="workspace-model-scope" title={ko ? "기준 치수와 확장 모델의 가정·적용 범위" : "Reference dimensions, assumptions, and model limits"}>
      <span className="scope-marker" aria-hidden />
      {extended ? (ko ? "확장 · 미보정" : "Extended · uncalibrated") : (ko ? "기준 치수" : "Reference geometry")}
      <span aria-hidden className="scope-info">i</span>
    </a>}
  </div>;
}

export default function App() {
  const tab = useStore((s) => s.tab);
  const theme = useStore((s) => s.theme);
  const mode = useStore((s) => s.mode);
  const lang = useStore((s) => s.lang);

  useEffect(() => {
    const el = document.documentElement;
    el.dataset.theme = theme;
    el.dataset.mode = mode;
    el.lang = lang;
    // the header shows only the name; the subtitle lives in the tab title and the About card
    document.title = `STL simulator — ${translate(lang, "app.subtitle")}`;
  }, [theme, mode, lang]);

  useEffect(() => {
    const stopHealth = startHealthPolling();
    const stopPerformance = startPerformanceTracking();
    const stopAuto = startAutoRun();
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void runCurrent();
      }
    };
    window.addEventListener("keydown", onKey);
    // crossing the drawer breakpoint: the sidebar is inline (open) when wide, a closed drawer when narrow
    // (otherwise shrinking the window leaves the drawer + scrim covering the results)
    const mq = window.matchMedia?.("(max-width: 1100px)");
    const onMq = (e: MediaQueryListEvent) => useStore.getState().setSidebar(!e.matches);
    mq?.addEventListener?.("change", onMq);
    return () => {
      stopHealth();
      stopPerformance();
      stopAuto();
      window.removeEventListener("keydown", onKey);
      mq?.removeEventListener?.("change", onMq);
    };
  }, []);

  const hasSidebar = tab === "device" || tab === "circuit";
  return (
    <>
      <a className="studio-skip-link" href="#main">{lang === "ko" ? "작업 영역으로 이동" : "Skip to workspace"}</a>
      <Header />
      <ContextStrip />
      <div className={`layout${hasSidebar ? "" : " no-sidebar"}`}>
        {hasSidebar && <Sidebar />}
        <main className="main" id="main" role="tabpanel" aria-labelledby={`tab-${tab}`} data-testid={`main-${tab}`}>
          <WorkspaceHeading />
          <ErrorBoundary label={tab} resetKey={tab}>
            {tab === "device" && <div className="device-workspace"><DeviceTab /><DeviceShelf /></div>}
            {tab === "circuit" && <CircuitTab />}
            {tab === "validation" && <ValidationTab />}
            {tab === "physics" && <PhysicsTab />}
            {tab === "performance" && <PerformanceTab />}
          </ErrorBoundary>
        </main>
      </div>
      <ErrorBoundary label="Details">
        <PhysicsWindow />
      </ErrorBoundary>
    </>
  );
}
