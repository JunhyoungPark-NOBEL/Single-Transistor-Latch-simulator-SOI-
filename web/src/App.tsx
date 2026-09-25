// App shell: header + context strip (hint, backend status pill, credits) + sidebar/main layout + floating
// Details window. There is no full-width banner: demo / offline / snapshot is the pill in the strip.
import { useEffect } from "react";
import { CircuitTab } from "./circuit/CircuitTab";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ContextStrip, Header } from "./components/Header";
import { DeviceTab } from "./device/DeviceTab";
import { translate } from "./i18n";
import { PhysicsTab } from "./physics/PhysicsTab";
import { PhysicsWindow } from "./physics/PhysicsWindow";
import { Sidebar } from "./sidebar/Sidebar";
import { runCurrent, startAutoRun, startHealthPolling } from "./state/runner";
import { useStore } from "./state/store";
import { ValidationTab } from "./validation/ValidationTab";

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
    document.title = `${translate(lang, "app.title")} — ${translate(lang, "app.subtitle")}`;
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
    // crossing the drawer breakpoint: the sidebar is inline (open) when wide, a closed drawer when narrow
    // (otherwise shrinking the window leaves the drawer + scrim covering the results)
    const mq = window.matchMedia?.("(max-width: 1100px)");
    const onMq = (e: MediaQueryListEvent) => useStore.getState().setSidebar(!e.matches);
    mq?.addEventListener?.("change", onMq);
    return () => {
      stopHealth();
      stopAuto();
      window.removeEventListener("keydown", onKey);
      mq?.removeEventListener?.("change", onMq);
    };
  }, []);

  const hasSidebar = tab === "device" || tab === "circuit";
  return (
    <>
      <Header />
      <ContextStrip />
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
