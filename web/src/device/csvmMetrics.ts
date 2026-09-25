import type { CustomCircuitResult, CustomEvent } from "../api/circuitCustom";
import type { L10n } from "../content/physics/types";

export interface CsvmMetricsOptions {
  run?: number;
  drainKey?: string;
  cell?: string;
  /** Full cycles to exclude in addition to the initial settling time. Default: 1. */
  discardCycles?: number;
  /** Fraction of the saved time range excluded as settling. Default: 0.2. */
  settleFraction?: number;
  /** Complete retained cycles required for a frequency. Default: 2; minimum: 2. */
  minCycles?: number;
}

export interface CsvmMetrics {
  vTop_V: number | null;
  vBottom_V: number | null;
  frequency_Hz: number | null;
  period_s: number | null;
  /** Complete cycles retained after startup/settling exclusion. */
  cycles: number;
  /** Complete cycles found before startup/settling exclusion. */
  observedCycles: number;
  status: "oscillating" | "insufficient-cycles" | "no-oscillation" | "no-data";
  reason: L10n;
  windowStart_s: number | null;
  windowEnd_s: number | null;
  source: "events" | "waveform";
}

type Sample = { t: number; v: number; segment: number };
type Cycle = { start: number; end: number; top: number; bottom: number };
const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const mean = (a: number[]) => a.reduce((sum, value) => sum + value, 0) / a.length;
const L = (ko: string, en: string): L10n => ({ ko, en });

function lowerBound(samples: Sample[], t: number): number {
  let a = 0, b = samples.length;
  while (a < b) {
    const m = (a + b) >>> 1;
    if (samples[m].t < t) a = m + 1;
    else b = m;
  }
  return a;
}

/** Real waveform extrema over a complete interval; never bridge a missing-data gap. */
function cycleRange(samples: Sample[], start: number, end: number): Cycle | null {
  if (!(end > start) || start < samples[0].t || end > samples.at(-1)!.t) return null;
  const i = lowerBound(samples, start), j = lowerBound(samples, end);
  const actualEnd = j < samples.length && samples[j].t === end ? j + 1 : j;
  // Interpolated boundaries cannot stand in for saved samples of a cycle.
  if (actualEnd - i < 3) return null;
  const a = samples[i].t === start ? i : i - 1;
  if (a < 0 || j >= samples.length || samples[a].segment !== samples[j].segment) return null;
  const at = (t: number, right: number) => {
    const r = samples[right];
    if (r.t === t) return r.v;
    const l = samples[right - 1];
    return l.v + (r.v - l.v) * ((t - l.t) / (r.t - l.t));
  };
  let measuredLow = Infinity, measuredHigh = -Infinity;
  for (let k = i; k < actualEnd; k++) {
    measuredLow = Math.min(measuredLow, samples[k].v);
    measuredHigh = Math.max(measuredHigh, samples[k].v);
  }
  const prominence = Math.max(1e-4, Math.max(Math.abs(measuredLow), Math.abs(measuredHigh)) * 1e-4);
  if (measuredHigh - measuredLow < prominence) return null;
  const v0 = at(start, i), v1 = at(end, j);
  let top = Math.max(v0, v1), bottom = Math.min(v0, v1);
  for (let k = i; k < j; k++) {
    top = Math.max(top, samples[k].v);
    bottom = Math.min(bottom, samples[k].v);
  }
  return { start, end, top, bottom };
}

function eventCycles(samples: Sample[], events: CustomEvent[]): { cycles: Cycle[]; unresolved: number } {
  const cycles: Cycle[] = [];
  let unresolved = 0;
  let up: CustomEvent | null = null;
  let down = false;
  for (const event of events) {
    if (event.kind === "latch_down") {
      if (up && event.t > up.t) down = true;
    } else {
      if (up && down) {
        const cycle = cycleRange(samples, up.t, event.t);
        if (cycle) cycles.push(cycle);
        else unresolved++;
      }
      up = event;
      down = false;
    }
  }
  return { cycles, unresolved };
}

/** Prominent peak-to-peak cycles, used only when the solver supplies no STL events. */
function waveformCycles(samples: Sample[], settlingTime: number): { cycles: Cycle[]; peaks: number } {
  const tail = samples.filter((sample) => sample.t >= settlingTime);
  let low = Infinity, high = -Infinity, magnitude = 0;
  for (const sample of tail) {
    low = Math.min(low, sample.v);
    high = Math.max(high, sample.v);
    magnitude = Math.max(magnitude, Math.abs(sample.v));
  }
  // 0.1 mV absolute / 0.01% relative floor rejects flat-trace numerical ripple.
  const prominence = Math.max(1e-4, magnitude * 1e-4, (high - low) * 0.05);
  const peaks: number[] = [];
  let rising = true, extreme = 0;
  for (let i = 1; i < samples.length; i++) {
    if (samples[i].segment !== samples[i - 1].segment) {
      rising = true;
      extreme = i;
      continue;
    }
    const delta = samples[i].v - samples[extreme].v;
    if (rising) {
      if (delta >= 0) extreme = i;
      else if (-delta >= prominence) {
        // A trace beginning at a maximum does not establish a preceding rise.
        if (extreme > 0 && samples[extreme - 1].segment === samples[extreme].segment) peaks.push(extreme);
        rising = false;
        extreme = i;
      }
    } else if (delta <= 0) extreme = i;
    else if (delta >= prominence) {
      rising = true;
      extreme = i;
    }
  }
  const cycles: Cycle[] = [];
  for (let i = 1; i < peaks.length; i++) {
    const cycle = cycleRange(samples, samples[peaks[i - 1]].t, samples[peaks[i]].t);
    if (cycle && cycle.top - cycle.bottom >= prominence) cycles.push(cycle);
  }
  return { cycles, peaks: peaks.length };
}

/**
 * Measure CSVM from a single saved V_D(t) trace. Vtop/Vbottom are the mean of
 * actual maxima/minima of retained full cycles, not static VLU/VLD or event
 * threshold voltages. Frequency is 1 / mean observed period; it is never taken
 * from a predicted period or inferred from Iin/Cdrain. Partial cycles, the first
 * full cycle, and the first 20% of the saved time range are excluded by default.
 */
export function analyzeCsvm(result: CustomCircuitResult, options: CsvmMetricsOptions = {}): CsvmMetrics {
  const empty: CsvmMetrics = {
    vTop_V: null, vBottom_V: null, frequency_Hz: null, period_s: null,
    cycles: 0, observedCycles: 0, status: "no-data", reason: L("전압 데이터가 없습니다", "No voltage data"),
    windowStart_s: null, windowEnd_s: null, source: "waveform",
  };
  const run = options.run === undefined ? result.runs[0] : result.runs.find((r) => r.run === options.run);
  const voltage = run?.signals.find((signal) => signal.key === (options.drainKey ?? "V(drain)"));
  if (!run || !voltage) return empty;
  const samples: Sample[] = [];
  let segment = 0;
  for (let i = 0; i < run.t.length; i++) {
    const t = run.t[i], v = voltage.values[i];
    if (!finite(t) || !finite(v)) { segment++; continue; }
    if (samples.length && t <= samples.at(-1)!.t) {
      // Duplicate solver samples are harmless; unsorted times are not usable.
      if (t < samples.at(-1)!.t) return empty;
      continue;
    }
    samples.push({ t, v, segment });
  }
  if (samples.length < 3 || samples.at(-1)!.t <= samples[0].t) return empty;
  const fraction = finite(options.settleFraction) ? Math.min(0.95, Math.max(0, options.settleFraction)) : 0.2;
  const settlingTime = samples[0].t + fraction * (samples.at(-1)!.t - samples[0].t);
  const discard = finite(options.discardCycles) ? Math.max(0, Math.floor(options.discardCycles)) : 1;
  const required = finite(options.minCycles) ? Math.max(2, Math.floor(options.minCycles)) : 2;
  const events = (result.events ?? []).filter((event) =>
    event.run === run.run && (event.cell === undefined || event.cell === (options.cell ?? "X1")) && finite(event.t)
    && (event.kind === "latch_up" || event.kind === "latch_down")
    && event.t >= samples[0].t && event.t <= samples.at(-1)!.t,
  ).sort((a, b) => a.t - b.t);
  const wave = events.length ? null : waveformCycles(samples, settlingTime);
  const detected = events.length ? eventCycles(samples, events) : null;
  const candidates = detected ? detected.cycles : wave!.cycles;
  const retained = candidates.slice(discard).filter((cycle) => cycle.start >= settlingTime);
  const observed = candidates.length;
  const activity = observed > 0 || (wave?.peaks ?? 0) > 0
    || (events.some((e) => e.kind === "latch_up") && events.some((e) => e.kind === "latch_down"));
  const status = retained.length >= required ? "oscillating" : activity ? "insufficient-cycles" : "no-oscillation";
  const period = retained.length >= required ? mean(retained.map((cycle) => cycle.end - cycle.start)) : null;
  return {
    ...empty,
    vTop_V: retained.length ? mean(retained.map((cycle) => cycle.top)) : null,
    vBottom_V: retained.length ? mean(retained.map((cycle) => cycle.bottom)) : null,
    frequency_Hz: period !== null && period > 0 ? 1 / period : null,
    period_s: period,
    cycles: retained.length,
    observedCycles: observed,
    status,
    reason: status === "oscillating"
      ? L(`${retained.length}주기 · 초기 구간 제외`, `${retained.length} cycles · startup excluded`)
      : status === "insufficient-cycles"
        ? detected?.unresolved
          ? L("파형 해상도 부족", "Waveform resolution too low")
          : L("관측 주기 부족 · 시간을 늘려주세요", "Too few cycles · increase duration")
        : L("발진이 관측되지 않았습니다", "No oscillation observed"),
    windowStart_s: retained[0]?.start ?? null,
    windowEnd_s: retained.at(-1)?.end ?? null,
    source: events.length ? "events" : "waveform",
  };
}
