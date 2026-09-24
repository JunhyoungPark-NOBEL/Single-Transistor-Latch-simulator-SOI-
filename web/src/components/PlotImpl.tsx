// Plotly component (react-plotly.js factory + the cartesian Plotly bundle: scatter, bar, histogram, heatmap,
// contour … — every trace type the app uses; 1.4 MB instead of 4.7 MB). Loaded lazily by ./Plot.tsx.
import Plotly from "plotly.js-cartesian-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { PlotImplProps } from "./Plot";

// react-plotly.js/factory is CommonJS: depending on the bundler interop the default export may be wrapped.
type Factory = typeof createPlotlyComponent;
const factory: Factory = ((createPlotlyComponent as unknown as { default?: Factory }).default ?? createPlotlyComponent) as Factory;
const PlotlyComponent = factory(Plotly);

export default function PlotImpl({ data, layout, config, onGraph, className }: PlotImplProps) {
  return (
    <PlotlyComponent
      data={data}
      layout={layout}
      config={{
        displaylogo: false,
        responsive: true,
        displayModeBar: "hover",
        modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage", "autoScale2d"],
        ...config,
      }}
      useResizeHandler
      className={className}
      style={{ width: "100%", height: "100%" }}
      onInitialized={(_, gd) => onGraph?.(gd as unknown as HTMLElement)}
      onUpdate={(_, gd) => onGraph?.(gd as unknown as HTMLElement)}
    />
  );
}

export async function exportPng(gd: HTMLElement, filename: string, bg: string) {
  const g = gd as unknown as { data: Plotly.Data[]; layout: Partial<Plotly.Layout> };
  const w = Math.max(gd.clientWidth, 600);
  const h = Math.max(gd.clientHeight, 360);
  const url = await Plotly.toImage(
    { data: g.data, layout: { ...g.layout, paper_bgcolor: bg, plot_bgcolor: bg, width: w, height: h } } as never,
    { format: "png", width: w, height: h, scale: 2 } as never,
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".png") ? filename : `${filename}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
