// Circuit tab sub-view (schematic editor | quick benches), persisted per browser. While the schematic
// view is active the circuit-tab Run action (Run button, Ctrl+Enter, empty-panel buttons) runs the
// user-drawn circuit; the editor code is loaded on demand.
import { create } from "zustand";
import { setCircuitRunOverride } from "../state/runner";

export type CircuitView = "schematic" | "benches";
const KEY = "stl-websim:circuit-view";

function load(): CircuitView {
  try {
    localStorage.setItem(KEY, "schematic");
    return "schematic";
  } catch {
    return "schematic";
  }
}

export const useCircuitView = create<{ view: CircuitView; setView: (v: CircuitView) => void }>((set) => ({
  view: typeof window !== "undefined" ? load() : "schematic",
  setView: (view) => {
    set({ view });
    try {
      localStorage.setItem(KEY, view);
    } catch {
      /* ignore */
    }
  },
}));

export const loadSchematicRun = () => import("../schematic/run");

function syncOverride(view: CircuitView) {
  setCircuitRunOverride(view === "schematic" ? () => loadSchematicRun().then((m) => m.runSchematic()) : null);
}
if (typeof window !== "undefined") {
  syncOverride(useCircuitView.getState().view);
  useCircuitView.subscribe((s, prev) => {
    if (s.view !== prev.view) syncOverride(s.view);
  });
}
