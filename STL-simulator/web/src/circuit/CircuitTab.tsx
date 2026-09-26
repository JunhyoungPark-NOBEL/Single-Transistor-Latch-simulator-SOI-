// Circuit workspace: examples and an editable schematic share the same simulation flow.
import { lazy, Suspense } from "react";
import { useT } from "../i18n";
import "./circuit.css";
const SchematicView = lazy(() => import("../schematic/SchematicView"));
export function CircuitTab() {
  const t = useT();
  return <Suspense fallback={<div className="skeleton-plot" aria-label={t("schematic.loading")} style={{ height: 500 }} />}><SchematicView /></Suspense>;
}
