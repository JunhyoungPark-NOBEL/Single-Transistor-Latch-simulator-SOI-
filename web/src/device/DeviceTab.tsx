// Device tab: (first visit) GettingStarted hint → answer bar (3 numbers + [간단히 | 모두 보기]) → (stochastic)
// one-line statistics → FocusLayout: the I–V hero plus one tabbed analysis card (간단히), or today's full
// grid of panels (모두 보기). Every tab result is still computed on Run, so switching tabs is instant; a tab
// whose result is running, failed or stale carries a dot.
import { entryStatus, FocusLayout, mergeStatus, type MoreTab } from "../components/MoreCard";
import { useT } from "../i18n";
import { DEV } from "../i18n/strings.device";
import { useStore } from "../state/store";
import { useDeviceKeys } from "./common";
import { ChargeBalancePanel, ComponentsPanel, IvPanel, VgPanel } from "./DetPanels";
import "./device.css";
import { GettingStarted } from "./GettingStarted";
import { KpiStrip } from "./KpiStrip";
import { CyclePanel, DesignMapPanel, DistPanel, HazardPanel, McIvPanel, StatsPanel, VgStochPanel } from "./StoPanels";

function DetLayout() {
  const t = useT();
  const results = useStore((s) => s.results);
  const keys = useDeviceKeys();
  const tabs: MoreTab[] = [
    { id: "vg", label: t.l(DEV["tab.vg"]), plainLabel: t.l(DEV["tab.vg.plain"]), title: t("p.vg.desc"), panel: <VgPanel />, status: entryStatus(results.vg_curve, keys.vg_curve) },
    { id: "components", label: t.l(DEV["tab.components"]), title: t("p.comp.desc"), panel: <ComponentsPanel />, status: entryStatus(results.branches, keys.branches) },
    { id: "charge-balance", label: t.l(DEV["tab.cb"]), title: t("p.cb.desc"), panel: <ChargeBalancePanel />, status: entryStatus(results.charge_balance, keys.charge_balance) },
  ];
  return <FocusLayout testId="panels-deterministic" scope="device-det" hero={<IvPanel />} tabs={tabs} defaultTab="vg" side allOrder={["hero", "components", "charge-balance", "vg"]} />;
}

function StoLayout() {
  const t = useT();
  const results = useStore((s) => s.results);
  const dm = useStore((s) => s.designMap);
  const keys = useDeviceKeys();
  const mc = entryStatus(results.sweep_mc, keys.sweep_mc);
  const tabs: MoreTab[] = [
    { id: "dist", label: t.l(DEV["tab.dist"]), title: t("p.dist.desc"), panel: <DistPanel />, status: mc },
    { id: "hazard", label: t.l(DEV["tab.hazard"]), title: t("p.hazard.desc"), panel: <HazardPanel />, status: entryStatus(results.hazard, keys.hazard) },
    { id: "cycles", label: t.l(DEV["tab.cycles"]), title: t("p.cycles.desc"), panel: <CyclePanel />, status: mc },
    { id: "vg-sto", label: t.l(DEV["tab.vgs"]), plainLabel: t.l(DEV["tab.vgs.plain"]), title: t("p.vgs.desc"), panel: <VgStochPanel />, status: entryStatus(results.vg_curve_stochastic, keys.vg_curve_stochastic) },
    { id: "design-map", label: t.l(DEV["tab.dmap"]), title: t("p.dmap.desc"), panel: <DesignMapPanel />, status: mergeStatus(dm.status === "loading" ? "running" : dm.status === "error" ? "error" : null) },
  ];
  return <FocusLayout testId="panels-stochastic" scope="device-sto" hero={<McIvPanel />} tabs={tabs} defaultTab="dist" side allOrder={["hero", "dist", "hazard", "vg-sto", "cycles", "design-map"]} />;
}

export function DeviceTab() {
  const mode = useStore((s) => s.mode);
  return (
    <div className="device-tab">
      <GettingStarted />
      <KpiStrip />
      {mode === "stochastic" && <StatsPanel />}
      {mode === "deterministic" ? <DetLayout /> : <StoLayout />}
    </div>
  );
}
