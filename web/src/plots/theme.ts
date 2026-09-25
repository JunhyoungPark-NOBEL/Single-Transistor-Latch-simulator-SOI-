// Plotly layout theme matching the app tokens + the semantic plot colours shared by all panels.
import type { Layout } from "plotly.js";
import type { Theme } from "../state/store";

export interface PlotPalette {
  text: string; text2: string; muted: string; grid: string; axis: string; zero: string; surface: string; border: string;
  hrs: string; lrs: string; unstable: string; meas: string; measBand: string; det: string; sto: string; stoSoft: string;
  detSoft: string; up: string; down: string; warn: string; categorical: string[]; sequential: string[];
}

const LIGHT: PlotPalette = {
  text: "#0f172a", text2: "#3b4658", muted: "#6b7587", grid: "#eceef2", axis: "#c9ced6", zero: "#d5d9e0", surface: "#ffffff", border: "#e2e5eb",
  hrs: "#2563eb", lrs: "#dc2626", unstable: "#94a3b8", meas: "#111827", measBand: "rgba(17,24,39,0.10)",
  det: "#0d9488", sto: "#7c3aed", stoSoft: "rgba(124,58,237,0.16)", detSoft: "rgba(13,148,136,0.14)", up: "#0d9488", down: "#b45309", warn: "#b45309",
  categorical: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  sequential: ["#f0f5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
};
const DARK: PlotPalette = {
  text: "#e7eaf0", text2: "#b7bfcc", muted: "#8a93a3", grid: "#232833", axis: "#3a4250", zero: "#2f3642", surface: "#161920", border: "#2a303b",
  hrs: "#60a5fa", lrs: "#f87171", unstable: "#7c8799", meas: "#e5e7eb", measBand: "rgba(229,231,235,0.12)",
  det: "#2dd4bf", sto: "#a78bfa", stoSoft: "rgba(167,139,250,0.18)", detSoft: "rgba(45,212,191,0.16)", up: "#2dd4bf", down: "#fbbf24", warn: "#fbbf24",
  categorical: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2f9e2f", "#9085e9", "#e66767"],
  sequential: ["#161f33", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
};

export const palette = (theme: Theme): PlotPalette => (theme === "dark" ? DARK : LIGHT);

export const PLOT_FONT = '"Pretendard", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

function axisDefaults(c: PlotPalette) {
  return {
    gridcolor: c.grid,
    zerolinecolor: c.zero,
    linecolor: c.axis,
    tickcolor: c.axis,
    ticks: "outside" as const,
    ticklen: 4,
    showline: true,
    automargin: true,
    exponentformat: "power" as const,
    title: { font: { size: 12, color: c.text2 }, standoff: 6 },
    tickfont: { size: 11, color: c.muted },
  };
}

function isObj(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}
function deepMerge<T>(a: T, b: unknown): T {
  if (!isObj(b)) return (b === undefined ? a : b) as T;
  const out: Record<string, unknown> = { ...(isObj(a) ? a : {}) };
  for (const [k, v] of Object.entries(b)) out[k] = isObj(v) && isObj(out[k]) ? deepMerge(out[k], v) : v;
  return out as T;
}

/** Base layout for the theme merged with a panel's layout; every x/y axis gets the axis defaults. */
export function themedLayout(theme: Theme, layout: Partial<Layout> = {}): Partial<Layout> {
  const c = palette(theme);
  const base: Partial<Layout> = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: PLOT_FONT, size: 12, color: c.text2 },
    margin: { l: 62, r: 16, t: 36, b: 46, pad: 0 },
    legend: { orientation: "h", x: 0, y: 1.01, xanchor: "left", yanchor: "bottom", font: { size: 11.5, color: c.text2 }, bgcolor: "rgba(0,0,0,0)", itemclick: "toggle", itemdoubleclick: "toggleothers" },
    hoverlabel: { bgcolor: c.surface, bordercolor: c.border, font: { color: c.text, family: PLOT_FONT, size: 12 }, align: "left" },
    hovermode: "closest",
    colorway: c.categorical,
    modebar: { bgcolor: "rgba(0,0,0,0)", color: c.muted, activecolor: c.text } as Layout["modebar"],
    uirevision: "keep",
    // size from the container on every update (panels change height with their stacked subplots)
    autosize: true,
  };
  let out = deepMerge(base, layout);
  const axes = new Set(["xaxis", "yaxis", ...Object.keys(layout).filter((k) => /^[xy]axis\d*$/.test(k))]);
  for (const k of axes) {
    (out as Record<string, unknown>)[k] = deepMerge(axisDefaults(c), (layout as Record<string, unknown>)[k] ?? {});
  }
  out = deepMerge(out, {});
  return out;
}

/**
 * Axis settings for a current axis in amperes with SI tick prefixes and an "A" suffix. Log: one prefix per
 * decade (1pA, 10pA … 1µA); linear: Plotly's shared engineering exponent, so every tick carries the same
 * prefix (0.5µA, 1µA, 1.5µA). The title comes from the axis.* dictionary (e.g. t("axis.idAbs")).
 */
export function currentAxis(log: boolean, title: string) {
  return log
    ? { type: "log" as const, title: { text: title }, tickformat: "~s", exponentformat: "SI" as const, ticksuffix: "A" }
    : { type: "linear" as const, title: { text: title }, exponentformat: "SI" as const, ticksuffix: "A" };
}

export const HOVER_IV = "V<sub>D</sub> = %{x:.3f} V<br>I<sub>D</sub> = %{y:.3~s}A";
