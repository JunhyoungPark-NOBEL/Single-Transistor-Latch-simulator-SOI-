// Axis titles and tick settings for circuit waveforms, shared by the quick benches and the schematic results.
// Titles come from the axis.* dictionary block; linear axes use Plotly's shared SI exponent so every tick of an
// axis carries the same prefix and the unit (20mV, 0.5µA, 2fC — "units follow the data").
import type { T } from "../i18n";
import { subs, withUnit } from "../plots/labels";

export const AXES = ["voltage", "current", "charge", "state", "logic"] as const;
export type AxisKind = (typeof AXES)[number];

/** Axis title of a waveform subplot (circuit signals have no single symbol: the name is the compact form). */
export function circuitAxisTitle(t: T, ax: AxisKind, unit: string, logI: boolean): string {
  const u = unit && unit !== "1" ? unit : "";
  switch (ax) {
    case "voltage":
      return t("axis.c.voltage", { u: u || "V" });
    case "current":
      return logI ? t("axis.c.currentAbs", { u: u || "A" }) : t("axis.c.current", { u: u || "A" });
    case "charge":
      return t("axis.c.charge", { u: u || "C" });
    case "state":
      return u ? t("axis.c.state", { u }) : t("axis.c.stateNoUnit");
    default:
      return t("axis.c.logic");
  }
}

/** Plot class for n stacked waveform subplots: 340 px up to two, 420 px for three, then ≈ 80 px per axis. */
export function stackClass(n: number): string {
  return n <= 2 ? "plot" : n === 3 ? "plot tall" : n === 4 ? "plot tall x4" : "plot tall x5";
}

/** Tick settings of a linear waveform axis with a physical unit (shared SI prefix + unit suffix). */
export function siTicks(unit: string) {
  return unit && unit !== "1" ? { exponentformat: "SI" as const, minexponent: 1, ticksuffix: unit } : {};
}

/** Title of a circuit sweep axis from the server's symbol (V_amp, P_sw, V_G, I_PH, P(1)) and unit. */
export function sweepAxisTitle(t: T, label: string, unit: string): string {
  const u = unit && unit !== "1" ? unit : "";
  switch (label) {
    case "V_amp":
    case "amplitude":
      return t("axis.sw.vamp", { u: u || "V" });
    case "P_sw":
      return t("axis.sw.psw");
    case "V_G":
      return t("axis.vg");
    case "I_PH":
      return t("axis.sw.iph", { u: u || "pA" });
    case "P(1)":
      return t("axis.sw.p1");
    default:
      return withUnit(subs(label), u);
  }
}
