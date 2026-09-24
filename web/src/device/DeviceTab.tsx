// Device tab: KPI strip + responsive panel grid (deterministic or stochastic set).
import { useStore } from "../state/store";
import { ChargeBalancePanel, ComponentsPanel, IvPanel, VgPanel } from "./DetPanels";
import { KpiStrip } from "./KpiStrip";
import { CyclePanel, DesignMapPanel, DistPanel, HazardPanel, McIvPanel, VgStochPanel } from "./StoPanels";

export function DeviceTab() {
  const mode = useStore((s) => s.mode);
  return (
    <>
      <KpiStrip />
      {mode === "deterministic" ? (
        <div className="grid" data-testid="panels-deterministic">
          <IvPanel />
          <ComponentsPanel />
          <ChargeBalancePanel />
          <VgPanel />
        </div>
      ) : (
        <div className="grid" data-testid="panels-stochastic">
          <McIvPanel />
          <DistPanel />
          <HazardPanel />
          <VgStochPanel />
          <CyclePanel />
          <DesignMapPanel />
        </div>
      )}
    </>
  );
}
