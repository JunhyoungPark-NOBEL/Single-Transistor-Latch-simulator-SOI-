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
 */
export function curveError(model: { vd: Arr; id: Arr }, measured: { vd: Arr; id: Arr }): CurveError {
  const points = model.vd.map((x, i) => [x, model.id[i]] as const)
    .filter((p): p is readonly [number, number] => typeof p[0] === "number" && Number.isFinite(p[0]) && typeof p[1] === "number" && Number.isFinite(p[1]))
    .sort((a, b) => a[0] - b[0]);
  let n = 0, logN = 0, squared = 0, logSquared = 0, loI = Infinity, hiI = -Infinity;
  if (points.length < 2) return { n, logN, logRmse: null, normalizedRmsePct: null };
  for (let i = 0; i < measured.vd.length; i++) {
    const x = measured.vd[i], ref = measured.id[i];
    if (typeof x !== "number" || !Number.isFinite(x) || typeof ref !== "number" || !Number.isFinite(ref)) continue;
    if (x < points[0][0] || x > points[points.length - 1][0]) continue;
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
