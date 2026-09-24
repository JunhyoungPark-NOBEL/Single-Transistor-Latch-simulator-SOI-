// App shell: header + mode bar + (offline banner) + sidebar/main layout + floating Details window.
import { useEffect } from "react";
import { CircuitTab } from "./circuit/CircuitTab";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Header, ModeBar } from "./components/Header";
import { DeviceTab } from "./device/DeviceTab";
import { useT } from "./i18n";
import { PhysicsTab } from "./physics/PhysicsTab";
import { PhysicsWindow } from "./physics/PhysicsWindow";
import { Sidebar } from "./sidebar/Sidebar";
import { initBackend, runCurrent, startAutoRun, startHealthPolling } from "./state/runner";
import { useStore } from "./state/store";
import { ValidationTab } from "./validation/ValidationTab";

function Banner() {
  const t = useT();
  const backend = useStore((s) => s.backend);
  if (backend !== "offline" && backend !== "mock") return null;
  return (
    <div className="banner" role="status" data-testid="offline-banner">
      <span className="dot mock" aria-hidden />
      <span>
        <strong>{backend === "mock" ? "Demo" : "Offline"}</strong> — {backend === "mock" ? t("banner.mockForced") : t("banner.offline")}
      </span>
      {backend === "offline" && (
        <button type="button" className="btn sm" onClick={() => void initBackend()}>
          {t("banner.retry")}
        </button>
      )}
    </div>
  );
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
  }, [theme, mode, lang]);

  useEffect(() => {
    const stopHealth = startHealthPolling();
    const stopAuto = startAutoRun();
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        void runCurrent();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      stopHealth();
      stopAuto();
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  const hasSidebar = tab === "device" || tab === "circuit";
  return (
    <>
      <Header />
      <ModeBar />
      <Banner />
      <div className={`layout${hasSidebar ? "" : " no-sidebar"}`}>
        {hasSidebar && <Sidebar />}
        <main className="main" id="main" role="tabpanel" aria-labelledby={`tab-${tab}`} data-testid={`main-${tab}`}>
          <ErrorBoundary label={tab} resetKey={tab}>
            {tab === "device" && <DeviceTab />}
            {tab === "circuit" && <CircuitTab />}
            {tab === "validation" && <ValidationTab />}
            {tab === "physics" && <PhysicsTab />}
          </ErrorBoundary>
        </main>
      </div>
      <ErrorBoundary label="Details">
        <PhysicsWindow />
      </ErrorBoundary>
    </>
  );
}
