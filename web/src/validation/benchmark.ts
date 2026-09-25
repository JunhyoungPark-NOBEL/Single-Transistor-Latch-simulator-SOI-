import type { Arr } from "../api/types";

export interface CurveError {
  /** Valid measured samples overlapping the simulated voltage range. */
  n: number;
  logN: number;
  logRmse: number | null;
  normalizedRmsePct: number | null;
}

/** Compare a simulated sweep with measured I_D samples at the same V_D.
 * Linear interpolation in amperes; no extrapolation, no replacement of missing values.
 * Log RMSE uses positive pairs only. Linear RMSE is normalized to the measured current span.
 * `minCurrent`: samples whose measured current is not above it are left out (the instrument floor);
 * `exclude`: samples with V_D inside this closed interval are left out (e.g. between two switching voltages).
 */
export function curveError(model: { vd: Arr; id: Arr }, measured: { vd: Arr; id: Arr }, minCurrent = -Infinity, exclude?: readonly [number, number]): CurveError {
  const points = model.vd.map((x, i) => [x, model.id[i]] as const)
    .filter((p): p is readonly [number, number] => typeof p[0] === "number" && Number.isFinite(p[0]) && typeof p[1] === "number" && Number.isFinite(p[1]))
    .sort((a, b) => a[0] - b[0]);
  let n = 0, logN = 0, squared = 0, logSquared = 0, loI = Infinity, hiI = -Infinity;
  if (points.length < 2) return { n, logN, logRmse: null, normalizedRmsePct: null };
  for (let i = 0; i < measured.vd.length; i++) {
    const x = measured.vd[i], ref = measured.id[i];
    if (typeof x !== "number" || !Number.isFinite(x) || typeof ref !== "number" || !Number.isFinite(ref)) continue;
    if (x < points[0][0] || x > points[points.length - 1][0]) continue;
    if (!(ref > minCurrent)) continue;
    if (exclude && x >= Math.min(...exclude) && x <= Math.max(...exclude)) continue;
    let lo = 0, hi = points.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (points[mid][0] > x) hi = mid; else lo = mid;
    }
    const [x0, y0] = points[lo], [x1, y1] = points[hi];
    const sim = x1 === x0 ? y1 : y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    n++;
    squared += (sim - ref) ** 2;
    loI = Math.min(loI, ref); hiI = Math.max(hiI, ref);
    if (sim > 0 && ref > 0) { logN++; logSquared += (Math.log10(sim) - Math.log10(ref)) ** 2; }
  }
  return {
    n, logN,
    logRmse: logN ? Math.sqrt(logSquared / logN) : null,
    normalizedRmsePct: n && hiI > loI ? 100 * Math.sqrt(squared / n) / (hiI - loI) : null,
  };
}

const num = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Low-current floor of a measurement: the median of its lowest 20 % positive currents (the instrument
 *  floor of the reference record is about 0.4 pA; the model keeps falling below it). Null without data. */
export function currentFloor(...currents: Arr[]): number | null {
  const v = currents.flat().filter((x): x is number => num(x) && x > 0).sort((a, b) => a - b);
  if (!v.length) return null;
  const low = v.slice(0, Math.max(1, Math.floor(v.length / 5)));
  return low[Math.floor((low.length - 1) / 2)];
}

/** Voltage of the switching jump of one sweep direction: the largest rise of log|I| between neighbouring
 *  samples for "up" (latch-up), the largest fall for "down" (latch-down); the midpoint of that step. A model
 *  sweep jumps at the fold itself (both samples at V_LU / V_LD). Null when there is no clear jump (< 1 decade). */
export function switchVoltage(sweep: { vd: Arr; id: Arr }, direction: "up" | "down"): number | null {
  let best = 0;
  let at: number | null = null;
  for (let i = 1; i < sweep.vd.length; i++) {
    const x0 = sweep.vd[i - 1], x1 = sweep.vd[i], y0 = sweep.id[i - 1], y1 = sweep.id[i];
    if (!num(x0) || !num(x1) || !num(y0) || !num(y1) || y0 <= 0 || y1 <= 0) continue;
    const step = Math.log10(y1 / y0) * (direction === "up" ? 1 : -1);
    if (step > best) {
      best = step;
      at = (x0 + x1) / 2;
    }
  }
  return best >= 1 ? at : null;
}
