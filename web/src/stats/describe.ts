// Descriptive statistics for Monte Carlo samples (device sweeps, circuit runs) — docs/WEB_CONTRACT.md §9.
//
// Everything is null-safe: `null`, `undefined`, NaN and ±Infinity entries are *censored* (e.g. a cycle that
// did not latch within the sweep). They count in `n_total` but not in the moments. Definitions follow
// numpy/scipy so the numbers match the server (server/compute/stoch_core.py `stats`, `lag1`, `ecdf`,
// `histogram`):
//   quantiles  numpy default (linear interpolation between order statistics, "type 7")
//   sd         ddof = 1                                   (np.std(x, ddof=1))
//   skewness   sample-adjusted G1                         (scipy.stats.skew(x, bias=False))
//   kurtosis   sample-adjusted excess G2                  (scipy.stats.kurtosis(x, bias=False))
//   lag1       Pearson r of consecutive pairs of the *raw* series with both entries finite
//              (pairs across a censored cycle are dropped, not bridged); null for < 3 pairs or a constant
//              series — stoch_core.lag1
//   ks2        two-sample Kolmogorov–Smirnov D (scipy.stats.ks_2samp statistic) with the asymptotic
//              Kolmogorov p-value Q_KS(√(n·m/(n+m))·D)  (scipy.stats.kstwobign.sf)
//   histogram  numpy.histogram semantics (last bin closed), Freedman–Diaconis bin width by default
//   ecdf       P(V ≤ x) normalised to all cycles (censored cycles keep the curve below 1)

export type Num = number | null | undefined;

export interface Describe {
  /** finite (observed) values */
  n: number;
  /** all entries, observed + censored (or opts.n_total) */
  n_total: number;
  /** n_total − n */
  censored: number;
  mean: number | null;
  /** sample standard deviation (ddof = 1); null for n < 2 */
  sd: number | null;
  /** standard error of the mean sd/√n */
  se: number | null;
  /** half-width of the 95 % t-based confidence interval of the mean, t(0.975, n−1)·se */
  ci95_half: number | null;
  /** coefficient of variation sd/|mean| (fraction, not %) */
  cv: number | null;
  median: number | null;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  p05: number | null;
  p95: number | null;
  min: number | null;
  max: number | null;
  /** sample-adjusted skewness G1 (n ≥ 3) */
  skewness: number | null;
  /** sample-adjusted excess kurtosis G2 (n ≥ 4); 0 for a normal distribution */
  kurtosis_excess: number | null;
  /** lag-1 autocorrelation of the series (consecutive finite pairs) */
  lag1: number | null;
}

export interface DescribeOptions {
  /** Total number of trials when `values` holds only the observed ones (censored = n_total − n). */
  n_total?: number;
}

export const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Finite entries, in order. */
export function finiteValues(values: readonly Num[] | null | undefined): number[] {
  const out: number[] = [];
  if (!values) return out;
  for (const v of values) if (isFiniteNum(v)) out.push(v);
  return out;
}

/** numpy.quantile (linear, "type 7") of an ascending-sorted array. */
export function quantileSorted(sorted: readonly number[], p: number): number | null {
  const n = sorted.length;
  if (!n || !(p >= 0 && p <= 1)) return null;
  const h = n * p + (1 - p) - 1; // numpy's virtual index (alpha = beta = 1), same rounding
  const lo = Math.max(0, Math.min(n - 1, Math.floor(h)));
  const hi = Math.min(n - 1, lo + 1);
  const f = h - lo;
  // numpy's lerp form (a + (b − a)·f, switched to b − (b − a)(1 − f) for f ≥ 0.5) keeps exact endpoints
  const a = sorted[lo];
  const b = sorted[hi];
  return f >= 0.5 ? b - (b - a) * (1 - f) : a + (b - a) * f;
}

/** numpy.quantile of arbitrary (unsorted, null-containing) values. */
export function quantile(values: readonly Num[], p: number): number | null {
  return quantileSorted(finiteValues(values).sort((a, b) => a - b), p);
}

/** Two-pass mean (stable for values with a large common offset, e.g. voltages near 3.6 V). */
function meanOf(x: readonly number[]): number {
  let s = 0;
  for (const v of x) s += v;
  const m = s / x.length;
  let c = 0;
  for (const v of x) c += v - m;
  return m + c / x.length;
}

/**
 * Lag-1 autocorrelation of a series (stoch_core.lag1): Pearson r of (x[i], x[i+1]) over the pairs where both
 * are finite. Null for fewer than 3 pairs or a (numerically) constant series.
 */
export function lag1(values: readonly Num[]): number | null {
  const a: number[] = [];
  const b: number[] = [];
  for (let i = 0; i + 1 < values.length; i++) {
    const u = values[i];
    const v = values[i + 1];
    if (isFiniteNum(u) && isFiniteNum(v)) {
      a.push(u);
      b.push(v);
    }
  }
  if (a.length < 3) return null;
  // "constant series" guard: the server uses an absolute 1e-9 (volts); relative below |x| = 1 so that
  // quantities in seconds or amperes are not mistaken for constants.
  const scale = Math.max(...a.map(Math.abs), ...b.map(Math.abs));
  const tol = 1e-9 * Math.min(1, scale);
  const ptp = (x: number[]) => Math.max(...x) - Math.min(...x);
  if (ptp(a) <= tol || ptp(b) <= tol) return null;
  const ma = meanOf(a);
  const mb = meanOf(b);
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  if (!(saa > 0 && sbb > 0)) return null;
  const r = sab / Math.sqrt(saa * sbb);
  return Math.max(-1, Math.min(1, r));
}

/** Descriptive statistics of a sample with censored (null / non-finite) entries. */
export function describe(values: readonly Num[] | null | undefined, opts: DescribeOptions = {}): Describe {
  const raw = values ?? [];
  const x = finiteValues(raw);
  const n = x.length;
  const n_total = Math.max(n, isFiniteNum(opts.n_total) ? Math.round(opts.n_total) : raw.length);
  const out: Describe = {
    n, n_total, censored: n_total - n,
    mean: null, sd: null, se: null, ci95_half: null, cv: null,
    median: null, q1: null, q3: null, iqr: null, p05: null, p95: null, min: null, max: null,
    skewness: null, kurtosis_excess: null, lag1: null,
  };
  if (!n) return out;
  const s = [...x].sort((a, b) => a - b);
  const mean = meanOf(x);
  out.mean = mean;
  out.min = s[0];
  out.max = s[n - 1];
  out.median = quantileSorted(s, 0.5);
  out.q1 = quantileSorted(s, 0.25);
  out.q3 = quantileSorted(s, 0.75);
  out.iqr = out.q3! - out.q1!;
  out.p05 = quantileSorted(s, 0.05);
  out.p95 = quantileSorted(s, 0.95);
  out.lag1 = lag1(raw);
  if (n < 2) return out;
  const constant = s[0] === s[n - 1];
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  if (!constant) {
    for (const v of x) {
      const d = v - mean;
      const d2 = d * d;
      m2 += d2;
      m3 += d2 * d;
      m4 += d2 * d2;
    }
  }
  const sd = constant ? 0 : Math.sqrt(m2 / (n - 1));
  out.sd = sd;
  out.se = sd / Math.sqrt(n);
  out.ci95_half = tQuantile(0.975, n - 1) * out.se;
  out.cv = mean !== 0 ? sd / Math.abs(mean) : null;
  if (constant) return out;
  m2 /= n;
  m3 /= n;
  m4 /= n;
  if (n >= 3) {
    const g1 = m3 / Math.pow(m2, 1.5);
    out.skewness = (g1 * Math.sqrt(n * (n - 1))) / (n - 2);
  }
  if (n >= 4) {
    const g2 = m4 / (m2 * m2) - 3;
    out.kurtosis_excess = (((n + 1) * g2 + 6) * (n - 1)) / ((n - 2) * (n - 3));
  }
  return out;
}

/** Per-index difference a[i] − b[i] (null where either is missing), e.g. the window V_LU − V_LD per cycle. */
export function diffSeries(a: readonly Num[], b: readonly Num[]): (number | null)[] {
  const n = Math.min(a.length, b.length);
  const out: (number | null)[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = a[i];
    const v = b[i];
    out[i] = isFiniteNum(u) && isFiniteNum(v) ? u - v : null;
  }
  return out;
}

// ---------------------------------------------------------------- confidence interval (Student t)
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
/** ln Γ(x), x > 0 (Lanczos, g = 7; relative error ~1e-15). */
export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  const z = x - 1;
  let a = LANCZOS[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(a: number, b: number, x: number): number {
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function ibeta(x: number, a: number, b: number): number {
  if (!(x > 0)) return 0;
  if (!(x < 1)) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log1p(-x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Student t CDF with df degrees of freedom. */
export function tCdf(t: number, df: number): number {
  if (Number.isNaN(t) || !(df > 0)) return NaN;
  if (!Number.isFinite(t)) return t > 0 ? 1 : 0;
  const tail = 0.5 * ibeta(df / (df + t * t), df / 2, 0.5);
  return t >= 0 ? 1 - tail : tail;
}

/** Student t quantile (inverse CDF) by bracketing + bisection (|error| < 1e-12 relative). */
export function tQuantile(p: number, df: number): number {
  if (!(p > 0 && p < 1) || !(df > 0)) return NaN;
  if (p === 0.5) return 0;
  if (p < 0.5) return -tQuantile(1 - p, df);
  let lo = 0;
  let hi = 2;
  while (tCdf(hi, df) < p && hi < 1e12) {
    lo = hi;
    hi *= 2;
  }
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (tCdf(mid, df) < p) lo = mid;
    else hi = mid;
    if (hi - lo <= 1e-13 * Math.max(1, hi)) break;
  }
  return 0.5 * (lo + hi);
}

/** Two-sided t-based confidence interval of the mean (level 0.95 → 95 %). */
export function meanCI(d: Pick<Describe, "n" | "mean" | "se">, level = 0.95): { lo: number; hi: number; half: number } | null {
  if (d.n < 2 || !isFiniteNum(d.mean) || !isFiniteNum(d.se)) return null;
  const half = tQuantile(1 - (1 - level) / 2, d.n - 1) * d.se;
  return { lo: d.mean - half, hi: d.mean + half, half };
}

// ---------------------------------------------------------------- two-sample Kolmogorov–Smirnov
/** Kolmogorov survival function Q_KS(z) = P(K > z) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²z²} (= scipy kstwobign.sf). */
export function kolmogorovQ(z: number): number {
  if (!(z > 0)) return 1;
  if (!Number.isFinite(z)) return 0;
  if (z < 1.18) {
    // Jacobi form of the CDF, converges fast for small z: P = √(2π)/z Σ exp(−(2k−1)²π²/(8z²))
    const y = Math.exp(-(Math.PI * Math.PI) / (8 * z * z));
    let s = 0;
    for (let k = 1; k <= 8; k++) s += Math.pow(y, (2 * k - 1) * (2 * k - 1));
    return Math.min(1, Math.max(0, 1 - (Math.sqrt(2 * Math.PI) / z) * s));
  }
  const x = Math.exp(-2 * z * z);
  let s = 0;
  for (let k = 1; k <= 12; k++) s += (k % 2 ? 1 : -1) * Math.pow(x, k * k);
  return Math.min(1, Math.max(0, 2 * s));
}

export interface KS2 {
  /** sup_x |F_a(x) − F_b(x)| over the finite values; null when a sample is empty */
  D: number | null;
  /** asymptotic p-value Q_KS(√(n_a n_b/(n_a+n_b))·D) */
  p: number | null;
  n_a: number;
  n_b: number;
}

/** Two-sample Kolmogorov–Smirnov test on the finite values of a and b (censored entries ignored). */
export function ks2(a: readonly Num[] | null | undefined, b: readonly Num[] | null | undefined): KS2 {
  const x = finiteValues(a).sort((u, v) => u - v);
  const y = finiteValues(b).sort((u, v) => u - v);
  const n1 = x.length;
  const n2 = y.length;
  if (!n1 || !n2) return { D: null, p: null, n_a: n1, n_b: n2 };
  let i = 0;
  let j = 0;
  let D = 0;
  while (i < n1 && j < n2) {
    const v = Math.min(x[i], y[j]);
    while (i < n1 && x[i] <= v) i++;
    while (j < n2 && y[j] <= v) j++;
    D = Math.max(D, Math.abs(i / n1 - j / n2));
  }
  const en = (n1 * n2) / (n1 + n2);
  return { D, p: kolmogorovQ(Math.sqrt(en) * D), n_a: n1, n_b: n2 };
}

// ---------------------------------------------------------------- histogram / ECDF
export interface Histogram {
  edges: number[];
  counts: number[];
  /** bin width (equal-width bins) */
  width: number;
  /** finite values inside [edges[0], edges[last]] */
  n: number;
}

export interface HistogramOptions {
  /** [lo, hi] of the bins (default: min … max of the finite values) */
  range?: [number, number];
  /** bounds on the automatic (Freedman–Diaconis) bin count; default 1 … 100 */
  minBins?: number;
  maxBins?: number;
}

/** Freedman–Diaconis bin width 2·IQR·n^(−1/3) (numpy "fd"); 0 when the IQR is 0. */
export function fdWidth(values: readonly Num[]): number {
  const s = finiteValues(values).sort((a, b) => a - b);
  if (s.length < 2) return 0;
  return 2 * ((quantileSorted(s, 0.75) ?? 0) - (quantileSorted(s, 0.25) ?? 0)) * Math.pow(s.length, -1 / 3);
}

/**
 * Equal-width histogram with numpy.histogram semantics ([a, b) bins, last bin [a, b]). `bins` = a count, or
 * "fd" (default, Freedman–Diaconis like numpy.histogram_bin_edges(x, "fd"), clamped to minBins…maxBins).
 * A sample without spread gets one bin of ±0.1 % around the value (±0.5 at 0, as numpy).
 */
export function histogram(values: readonly Num[] | null | undefined, bins: number | "fd" = "fd", opts: HistogramOptions = {}): Histogram {
  const x = finiteValues(values);
  if (!x.length) return { edges: [], counts: [], width: 0, n: 0 };
  let lo = opts.range ? opts.range[0] : Math.min(...x);
  let hi = opts.range ? opts.range[1] : Math.max(...x);
  if (!(hi > lo)) {
    const half = lo !== 0 ? Math.abs(lo) * 1e-3 : 0.5;
    lo -= half;
    hi += half;
  }
  let nb: number;
  if (bins === "fd") {
    const w = fdWidth(x);
    nb = w > 0 ? Math.ceil((hi - lo) / w) : 1;
    nb = Math.min(opts.maxBins ?? 100, Math.max(opts.minBins ?? 1, nb));
  } else {
    nb = Math.max(1, Math.round(bins));
  }
  const step = (hi - lo) / nb; // numpy.linspace: start + k·step, last edge exactly hi
  const edges = Array.from({ length: nb + 1 }, (_, k) => (k === nb ? hi : lo + k * step));
  const counts: number[] = new Array(nb).fill(0);
  let n = 0;
  for (const v of x) {
    if (v < lo || v > hi) continue;
    let k = Math.floor(((v - lo) / (hi - lo)) * nb);
    if (k >= nb) k = nb - 1;
    // numpy's correction for rounding at the edges
    if (k > 0 && v < edges[k]) k--;
    if (k < nb - 1 && v >= edges[k + 1]) k++;
    counts[k]++;
    n++;
  }
  return { edges, counts, width: (hi - lo) / nb, n };
}

export interface Ecdf {
  /** sorted finite values */
  v: number[];
  /** P(V ≤ v[i]) = (i + 1)/n_total */
  p: number[];
  n: number;
  n_total: number;
}

/**
 * Empirical CDF normalised to all cycles (stoch_core.ecdf): censored cycles keep the curve below 1 (it plateaus
 * at the observed fraction n/n_total). n_total defaults to values.length.
 */
export function ecdf(values: readonly Num[] | null | undefined, n_total?: number): Ecdf {
  const v = finiteValues(values).sort((a, b) => a - b);
  const n = v.length;
  const total = Math.max(n, isFiniteNum(n_total) ? Math.round(n_total) : (values?.length ?? 0));
  return { v, p: v.map((_, i) => (i + 1) / total), n, n_total: total };
}
