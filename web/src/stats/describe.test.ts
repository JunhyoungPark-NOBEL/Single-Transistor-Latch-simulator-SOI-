// Reference values: numpy 2.4 / scipy 1.17 (np.quantile, np.std(ddof=1), scipy.stats.skew/kurtosis(bias=False),
// scipy.stats.t.ppf, scipy.stats.ks_2samp statistic, scipy.stats.kstwobign.sf, np.histogram) and the server's
// own stoch_core.stats / lag1 on real sweep_mc results (fixtures/sweep_mc_presets.json).
import { describe as suite, expect, it } from "vitest";
import {
  describe, diffSeries, ecdf, fdWidth, histogram, ks2, kolmogorovQ, lag1, meanCI, quantile, tCdf, tQuantile, type Num,
} from "./describe";
import fx from "./fixtures/sweep_mc_presets.json";

type ServerStats = { n: number; mean: number | null; sd: number | null; median: number | null; p05: number | null; p95: number | null; min: number | null; max: number | null; censored: number; lag1: number | null };
const F = fx as unknown as {
  photo: { V_LU: Num[]; V_LD: Num[]; stats: { LU: ServerStats; LD: ServerStats }; measured_V_LU: number[]; measured_stats_LU: ServerStats; cdf_LU_n: number; cdf_LU_n_total: number };
  paper: { V_LU: Num[]; V_LD: Num[]; stats: { LU: ServerStats; LD: ServerStats }; measured_V_LU: number[]; measured_V_LD: number[]; measured_stats: { LU: ServerStats; LD: ServerStats }; cdf_LU: { v: number[]; p: number[]; n: number; n_total: number } };
};

/** |a − b| ≤ tol·max(1, |b|) */
function close(a: number | null | undefined, b: number | null | undefined, tol = 1e-12) {
  if (b === null || b === undefined) {
    expect(a).toBeNull();
    return;
  }
  expect(a).not.toBeNull();
  expect(Math.abs((a as number) - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));
}

function sameAsServer(values: Num[], st: ServerStats, tol = 1e-12) {
  const d = describe(values);
  expect(d.n).toBe(st.n);
  expect(d.censored).toBe(st.censored);
  for (const k of ["mean", "sd", "median", "p05", "p95", "min", "max", "lag1"] as const) close(d[k], st[k], tol);
  return d;
}

suite("describe — hand-checked examples", () => {
  it("[1, 2, 3, 4, 10]", () => {
    const d = describe([1, 2, 3, 4, 10]);
    expect(d.n).toBe(5);
    expect(d.n_total).toBe(5);
    expect(d.censored).toBe(0);
    close(d.mean, 4);
    close(d.sd, Math.sqrt(12.5)); // Σ(x−4)² = 50, /(n−1)
    close(d.se, Math.sqrt(12.5) / Math.sqrt(5));
    close(d.cv, Math.sqrt(12.5) / 4);
    close(d.median, 3);
    close(d.q1, 2);
    close(d.q3, 4);
    close(d.iqr, 2);
    close(d.p05, 1.2);
    close(d.p95, 8.799999999999999);
    close(d.min, 1);
    close(d.max, 10);
    // m2 = 10, m3 = 36, m4 = 278.8: G1 = 36/10^1.5·√20/3, G2 = ((6)(0.2788·10 − 3)·… ) — scipy values:
    close(d.skewness, 1.6970562748477143);
    close(d.kurtosis_excess, 3.152000000000001);
    close(d.lag1, 0.8980265101338745);
    close(d.ci95_half, 4.389945165425414, 1e-10);
  });

  it("censored entries (null, NaN, ±Infinity) are excluded and counted; lag1 drops pairs across them", () => {
    const v: Num[] = [3.1, null, 3.4, 2.9, NaN, 3.6, 3.3, undefined, 3.0, 3.2];
    const d = describe(v);
    expect(d.n).toBe(7);
    expect(d.n_total).toBe(10);
    expect(d.censored).toBe(3);
    close(d.mean, 3.2142857142857144);
    close(d.sd, 0.24102953780654787);
    close(d.median, 3.2);
    close(d.q1, 3.05);
    close(d.q3, 3.3499999999999996);
    close(d.p05, 2.93);
    close(d.p95, 3.54);
    close(d.skewness, 0.3672769361362857, 1e-11);
    close(d.kurtosis_excess, -0.5034130610051037, 1e-11);
    // pairs (3.4, 2.9), (3.6, 3.3), (3.0, 3.2) only
    close(d.lag1, 0.052414241836095284, 1e-11);
    close(d.ci95_half, 0.22291514325944914, 1e-10);
    expect(describe([1, Infinity, 2, -Infinity]).censored).toBe(2);
  });

  it("n_total option counts censored cycles that are not in the array", () => {
    const d = describe([1, 2, 3], { n_total: 5 });
    expect(d.n).toBe(3);
    expect(d.n_total).toBe(5);
    expect(d.censored).toBe(2);
    expect(describe([1, 2, 3], { n_total: 1 }).n_total).toBe(3); // never below n
  });

  it("degenerate samples", () => {
    const e = describe([]);
    expect(e).toMatchObject({ n: 0, n_total: 0, censored: 0, mean: null, sd: null, median: null, lag1: null, skewness: null });
    const nulls = describe([null, null, undefined]);
    expect(nulls).toMatchObject({ n: 0, n_total: 3, censored: 3, mean: null });
    expect(describe(null).n).toBe(0);
    const one = describe([2.5]);
    expect(one).toMatchObject({ n: 1, mean: 2.5, median: 2.5, min: 2.5, max: 2.5, sd: null, se: null, skewness: null, kurtosis_excess: null, lag1: null });
    const c = describe([2.5, 2.5, 2.5, 2.5]);
    expect(c.sd).toBe(0);
    expect(c.ci95_half).toBe(0);
    expect(c.skewness).toBeNull(); // scipy: nan (catastrophic cancellation)
    expect(c.kurtosis_excess).toBeNull();
    expect(c.lag1).toBeNull(); // constant series (stoch_core.lag1)
    const three = describe([1, 2, 4]);
    expect(three.skewness).not.toBeNull();
    expect(three.kurtosis_excess).toBeNull(); // needs n ≥ 4
    expect(describe([0, 0, 1]).cv).not.toBeNull();
    expect(describe([-1, 1]).cv).toBeNull(); // mean 0
  });

  it("lag1 needs ≥ 3 finite pairs and a non-constant series; tolerance is relative for small magnitudes", () => {
    expect(lag1([1, 2, 3])).toBeNull(); // 2 pairs
    expect(lag1([1, 2, null, 3, 4])).toBeNull(); // 2 pairs
    close(lag1([1, 2, 3, 5]), 0.9819805060619656, 1e-12);
    // quantities in seconds (µs scale) are not treated as constant
    const us = [1.0e-6, 1.2e-6, 0.9e-6, 1.1e-6, 1.3e-6];
    close(lag1(us), lag1([1.0, 1.2, 0.9, 1.1, 1.3]), 1e-9);
    expect(lag1([3.6, 3.6, 3.6, 3.6])).toBeNull();
  });

  it("quantile matches numpy's linear method", () => {
    expect(quantile([4, 1, null, 3, 2], 0.5)).toBe(2.5);
    expect(quantile([1, 2, 3, 4, 10], 0.95)).toBeCloseTo(8.8, 12);
    expect(quantile([], 0.5)).toBeNull();
    expect(quantile([1, 2], 1.5)).toBeNull();
    expect(quantile([1, 2, 3], 1)).toBe(3);
    expect(quantile([1, 2, 3], 0)).toBe(1);
  });

  it("diffSeries: per-index window, null where either side is censored", () => {
    expect(diffSeries([3.6, null, 3.7, 3.8], [2.7, 2.6, undefined, 2.75])).toEqual([3.6 - 2.7, null, null, 3.8 - 2.75]);
    expect(diffSeries([1, 2, 3], [1])).toEqual([0]);
  });
});

suite("describe — identical to the server (stoch_core.stats) on real sweep_mc results", () => {
  it("photo preset (general engine, 400 cycles): V_LU, V_LD and the measured record", () => {
    const lu = sameAsServer(F.photo.V_LU, F.photo.stats.LU);
    sameAsServer(F.photo.V_LD, F.photo.stats.LD);
    sameAsServer(F.photo.measured_V_LU, F.photo.measured_stats_LU);
    // scipy on the same arrays
    close(lu.skewness, 0.3874958796, 1e-9);
    close(lu.kurtosis_excess, 0.4585395373, 1e-9);
    close(lu.q1, 3.6934068277, 1e-10);
    close(lu.q3, 3.9178983233, 1e-10);
    close(lu.se, 0.0086900534, 1e-9);
    close(lu.ci95_half, 0.0170840133, 1e-9);
    const m = describe(F.photo.measured_V_LU);
    close(m.skewness, -0.0864170879, 1e-9);
    close(m.kurtosis_excess, -0.3000716986, 1e-9);
    close(m.lag1, 0.1987026259, 1e-9);
  });

  it("reference preset (calibrated lookup, 100 cycles, one censored): lag1 drops the pairs around the censored cycle", () => {
    const lu = sameAsServer(F.paper.V_LU, F.paper.stats.LU);
    expect(lu.n_total).toBe(100);
    expect(lu.censored).toBe(1);
    close(lu.lag1, 0.6827202072, 1e-9);
    close(lu.skewness, -0.3882437584, 1e-9);
    close(lu.kurtosis_excess, -0.3331193006, 1e-9);
    const ld = sameAsServer(F.paper.V_LD, F.paper.stats.LD);
    close(ld.skewness, 0.639573324, 1e-8);
    close(ld.kurtosis_excess, 1.6660507018, 1e-9);
    sameAsServer(F.paper.measured_V_LU, F.paper.measured_stats.LU);
    sameAsServer(F.paper.measured_V_LD, F.paper.measured_stats.LD);
    // per-cycle window V_LU − V_LD (numpy on the paired array)
    const w = describe(diffSeries(F.paper.V_LU, F.paper.V_LD));
    expect(w.n).toBe(99);
    close(w.mean, 0.9344444444, 1e-9);
    close(w.sd, 0.1197124294, 1e-9);
    close(w.median, 0.95, 1e-12);
    close(w.lag1, 0.6531695212, 1e-9);
    close(w.skewness, -0.4116155095, 1e-9);
  });
});

suite("confidence interval (Student t)", () => {
  const ref: [number, number, number, number, number][] = [
    // df, t(0.975), t(0.995), t(0.9), cdf(1.5)
    [1, 12.706204736174694, 63.656741162871526, 3.0776835371752544, 0.8128329581890011],
    [2, 4.302652729749462, 9.924843200918287, 1.8856180831641272, 0.8638034375544994],
    [5, 2.5705818356363146, 4.032142983555228, 1.4758840488244815, 0.9030481598787634],
    [10, 2.228138851986274, 3.16927267261695, 1.372183641110336, 0.9177463367772799],
    [30, 2.0422724563012378, 2.7499956535672254, 1.3104150253913955, 0.927967035435677],
    [99, 1.9842169515864174, 2.626405457280827, 1.2901614420344854, 0.9316015911420134],
    [399, 1.965927295920882, 2.588207164030976, 1.2836769219240642, 0.9327973834497422],
  ];
  it.each(ref)("df = %i", (df, q975, q995, q90, c15) => {
    close(tQuantile(0.975, df), q975, 1e-10);
    close(tQuantile(0.995, df), q995, 1e-10);
    close(tQuantile(0.9, df), q90, 1e-10);
    close(tQuantile(0.025, df), -q975, 1e-10);
    close(tCdf(1.5, df), c15, 1e-12);
  });
  it("edge cases and meanCI", () => {
    expect(tQuantile(0.5, 7)).toBe(0);
    expect(Number.isNaN(tQuantile(1, 7))).toBe(true);
    expect(Number.isNaN(tQuantile(0.9, 0))).toBe(true);
    const ci = meanCI(describe([1, 2, 3, 4, 10]))!;
    close(ci.half, 4.389945165425414, 1e-10);
    close(ci.lo, 4 - 4.389945165425414, 1e-10);
    close(meanCI(describe([1, 2, 3, 4, 10]), 0.99)!.half, 4.604094871349992 * Math.sqrt(12.5) / Math.sqrt(5), 1e-9);
    expect(meanCI(describe([1]))).toBeNull();
  });
});

suite("two-sample Kolmogorov–Smirnov", () => {
  it("statistic matches scipy.stats.ks_2samp (with ties)", () => {
    expect(ks2([1, 2, 3], [1.5, 2.5, 3.5, 4.5]).D).toBeCloseTo(0.5, 12);
    expect(ks2([1, 1, 2, 2], [1, 2, 2, 3]).D).toBeCloseTo(0.25, 12);
    expect(ks2([1, 2, 3], [1, 2, 3]).D).toBe(0);
    expect(ks2([1, 2], [5, 6]).D).toBe(1);
    close(ks2([1, 2, 3], [1.5, 2.5, 3.5, 4.5]).p, 0.784769805922802, 1e-9);
  });
  it("asymptotic Kolmogorov survival Q_KS = scipy.stats.kstwobign.sf", () => {
    const ref: [number, number][] = [
      [0.1, 1.0], [0.3, 0.9999906941986655], [0.5, 0.9639452436648751], [0.8, 0.5441424115741981],
      [1.0, 0.26999967167735456], [1.17, 0.12939004218561884], [1.18, 0.1234538094297657], [1.2, 0.11224966667072497],
      [1.5, 0.022217962616525127], [2.0, 0.0006709252557796953],
    ];
    for (const [z, q] of ref) close(kolmogorovQ(z), q, 1e-12);
    expect(Math.abs(kolmogorovQ(3.0) - 3.045995948942526e-8)).toBeLessThan(1e-18);
    expect(kolmogorovQ(0)).toBe(1);
    expect(kolmogorovQ(Infinity)).toBe(0);
  });
  it("model vs measured on real data (scipy D; kstwobign p)", () => {
    const a = ks2(F.photo.V_LU, F.photo.measured_V_LU);
    close(a.D, 0.04999999999999993, 1e-12);
    close(a.p, 0.6993741991310172, 1e-9);
    expect([a.n_a, a.n_b]).toEqual([400, 400]);
    const b = ks2(F.paper.V_LU, F.paper.measured_V_LU); // 99 finite vs 100
    close(b.D, 0.09858585858585855, 1e-12);
    close(b.p, 0.7189557636596338, 1e-9);
    const c = ks2(F.paper.V_LD, F.paper.measured_V_LD);
    close(c.D, 0.2, 1e-12);
    close(c.p, 0.03663105270711935, 1e-9);
  });
  it("empty samples give nulls", () => {
    expect(ks2([], [1, 2])).toEqual({ D: null, p: null, n_a: 0, n_b: 2 });
    expect(ks2([null, NaN], null)).toEqual({ D: null, p: null, n_a: 0, n_b: 0 });
  });
});

suite("histogram and ECDF", () => {
  const x = [0, 0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.9, 1.0];
  it("fixed bin count = numpy.histogram (last bin closed)", () => {
    const h = histogram(x, 4);
    expect(h.edges).toEqual([0, 0.25, 0.5, 0.75, 1.0]);
    expect(h.counts).toEqual([3, 2, 2, 2]);
    expect(h.n).toBe(9);
    expect(h.width).toBeCloseTo(0.25, 15);
  });
  it("Freedman–Diaconis default = numpy 'fd'", () => {
    const h = histogram(x);
    expect(h.edges.length - 1).toBe(3);
    expect(h.counts).toEqual([5, 2, 2]);
    // real data: numpy.histogram_bin_edges(x, 'fd')
    const p = histogram(F.photo.V_LU);
    expect(p.counts.length).toBe(18);
    expect(p.counts.slice(0, 8)).toEqual([4, 8, 11, 32, 33, 54, 55, 56]);
    expect(p.counts.reduce((s, c) => s + c, 0)).toBe(400);
    const q = histogram(F.paper.V_LU);
    expect(q.counts.slice(0, 8)).toEqual([5, 5, 14, 17, 22, 24, 10, 2]);
    expect(q.n).toBe(99);
    expect(fdWidth([1, 1, 1])).toBe(0);
  });
  it("range, clamps and degenerate samples", () => {
    const h = histogram([0.5, 1.5, 2.5, 9], 3, { range: [0, 3] });
    expect(h.counts).toEqual([1, 1, 1]);
    expect(h.n).toBe(3); // 9 is outside the range
    expect(histogram([3.6, 3.6, 3.6]).counts).toEqual([3]);
    expect(histogram([0, 0]).edges).toEqual([-0.5, 0.5]);
    expect(histogram([]).counts).toEqual([]);
    expect(histogram(x, "fd", { minBins: 10 }).counts.length).toBe(10);
    expect(histogram(x, "fd", { maxBins: 2 }).counts.length).toBe(2);
  });
  it("ecdf is normalised to all cycles (censored ones keep it below 1) — same as the server's cdf", () => {
    const e = ecdf([3, null, 1, 2]);
    expect(e).toEqual({ v: [1, 2, 3], p: [0.25, 0.5, 0.75], n: 3, n_total: 4 });
    expect(ecdf([1, 2], 5).p).toEqual([0.2, 0.4]);
    expect(ecdf([1, 2, 3], 1).n_total).toBe(3);
    const s = ecdf(F.paper.V_LU);
    expect(s.n).toBe(F.paper.cdf_LU.n);
    expect(s.n_total).toBe(F.paper.cdf_LU.n_total);
    expect(s.v).toEqual(F.paper.cdf_LU.v);
    s.p.forEach((p, i) => close(p, F.paper.cdf_LU.p[i], 1e-15));
    expect(s.p[s.p.length - 1]).toBeCloseTo(0.99, 12);
  });
});
