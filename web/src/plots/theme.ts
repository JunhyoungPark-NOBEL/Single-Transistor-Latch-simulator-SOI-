// Plotly layout theme matching the app tokens + the semantic plot colours shared by all panels.
import type { Layout } from "plotly.js";
import type { Theme } from "../state/store";

export interface PlotPalette {
  text: string; text2: string; muted: string; grid: string; axis: string; zero: string; surface: string; border: string;
  hrs: string; lrs: string; unstable: string; meas: string; measBand: string; det: string; sto: string; stoSoft: string;
  detSoft: string; up: string; down: string; warn: string; categorical: string[]; sequential: string[];
  /** Translucent fills of the V_LU (HRS) / V_LD (LRS) colours and a neutral one (window shading, ± σ bands). */
  hrsSoft: string; lrsSoft: string; neutralSoft: string;
  /** Grey of the "이전" (previous run) ghost curve. */
  ghost: string;
}

const LIGHT: PlotPalette = {
  text: "#172c46", text2: "#435a73", muted: "#607286", grid: "#edf1f7", axis: "#bbcbdc", zero: "#d5deea", surface: "#ffffff", border: "#dce4ef",
  hrs: "#245eac", lrs: "#087f83", unstable: "#8b9aad", meas: "#293849", measBand: "rgba(41,56,73,0.10)",
  det: "#245eac", sto: "#986215", stoSoft: "rgba(152,98,21,0.13)", detSoft: "rgba(36,94,172,0.11)", up: "#245eac", down: "#087f83", warn: "#986215",
  categorical: ["#245eac", "#087f83", "#b87719", "#9d4972", "#5368a4", "#318667", "#b25745", "#697b8f"],
  sequential: ["#f0f5fc", "#d9e8fb", "#b4d1f5", "#8ab9eb", "#5697d5", "#3379bd", "#245eac", "#194c91"],
  hrsSoft: "rgba(36,94,172,0.10)", lrsSoft: "rgba(8,127,131,0.10)", neutralSoft: "rgba(102,121,141,0.09)", ghost: "#8b9aad",
};
const DARK: PlotPalette = {
  text: "#e8eff7", text2: "#b7c8db", muted: "#91a6bd", grid: "#263a50", axis: "#4b637e", zero: "#354b63", surface: "#182536", border: "#2c4056",
  hrs: "#8ab9f5", lrs: "#65c9c3", unstable: "#6b819a", meas: "#e8eff7", measBand: "rgba(232,239,247,0.12)",
  det: "#8ab9f5", sto: "#edbf70", stoSoft: "rgba(237,191,112,0.14)", detSoft: "rgba(138,185,245,0.14)", up: "#8ab9f5", down: "#65c9c3", warn: "#edbf70",
  categorical: ["#8ab9f5", "#65c9c3", "#edbf70", "#dfa0bd", "#a8b3e9", "#91caa5", "#e7a491", "#b7c8db"],
  sequential: ["#172b45", "#213b5c", "#275384", "#3379bd", "#5697d5", "#8ab9eb", "#b4d1f5", "#d9e8fb"],
  hrsSoft: "rgba(138,185,245,0.13)", lrsSoft: "rgba(101,201,195,0.13)", neutralSoft: "rgba(145,166,189,0.12)", ghost: "#6b819a",
};

export const palette = (theme: Theme): PlotPalette => (theme === "dark" ? DARK : LIGHT);

export const PLOT_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR Variable", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif';

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
    title: { font: { size: 14, color: c.text2 }, standoff: 6 },
    tickfont: { size: 13, color: c.muted },
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
    font: { family: PLOT_FONT, size: 13, color: c.text2 },
    margin: { l: 68, r: 18, t: 40, b: 52, pad: 0 },
    legend: { orientation: "h", x: 0, y: 1.01, xanchor: "left", yanchor: "bottom", font: { size: 13, color: c.text2 }, bgcolor: "rgba(0,0,0,0)", itemclick: "toggle", itemdoubleclick: "toggleothers" },
    hoverlabel: { bgcolor: c.surface, bordercolor: c.border, font: { color: c.text, family: PLOT_FONT, size: 13 }, align: "left" },
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
    const input = ((layout as Record<string, unknown>)[k] ?? {}) as Record<string, unknown>;
    (out as Record<string, unknown>)[k] = deepMerge(axisDefaults(c), {
      // Plotly preserves a zoomed range while uirevision stays the same. A log range contains
      // exponents, so reusing it as amperes blanks the linear plot. Scope persistence to the scale.
      uirevision: `${String(layout.uirevision ?? "keep")}:${String(input.type ?? "linear")}`,
      autorange: !Array.isArray(input.range),
      ...input,
    });
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
    : { type: "linear" as const, range: undefined, autorange: true, title: { text: title }, tickformat: "~s", exponentformat: "SI" as const, ticksuffix: "A" };
}

export const HOVER_IV = "V<sub>D</sub> = %{x:.3f} V<br>I<sub>D</sub> = %{y:.3~s}A";
