// Themed, lazily loaded Plotly chart. Plotly (~4 MB) is split into its own chunk so the app shell
// renders immediately; a shimmer placeholder is shown until it arrives.
import type { Config, Data, Layout } from "plotly.js";
import { lazy, Suspense, useMemo } from "react";
import { themedLayout } from "../plots/theme";
import { mathPlotLabels } from "../plots/labels";
import { useStore } from "../state/store";

export interface PlotImplProps {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  onGraph?: (gd: HTMLElement) => void;
  className?: string;
}

const PlotImpl = lazy(() => import("./PlotImpl"));

export function Plot({ data, layout, config, onGraph, className = "plot" }: PlotImplProps) {
  const theme = useStore((s) => s.theme);
  const themed = useMemo(() => mathPlotLabels(themedLayout(theme, layout)), [theme, layout]);
  const formattedData = useMemo(() => data.map(mathPlotLabels), [data]);
  return (
    <div className={className} data-testid="plot">
      <Suspense fallback={<div className="skeleton-plot" style={{ height: "100%" }} />}>
        <PlotImpl data={formattedData} layout={themed} config={config} onGraph={onGraph} />
      </Suspense>
    </div>
  );
}

export async function exportPlotPng(gd: HTMLElement, filename: string, bg: string) {
  const m = await import("./PlotImpl");
  return m.exportPng(gd, filename, bg);
}
