import type { TimeRange } from "./types";

export function secondsLabel(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return "—";
  if (value === 0) return "0 s";
  if (value < 0.001) return "< 1 ms";
  if (value < 1) return `${Math.max(1, Math.round(value * 1000))} ms`;
  if (value < 60) return `${value < 10 ? value.toFixed(1) : Math.round(value)} s`;
  return `${(value / 60).toFixed(1)} min`;
}
export function rangeLabel(range: TimeRange | undefined): string {
  if (!range || range.seconds === null || !Number.isFinite(range.seconds)) return "—";
  const lo = range.low_s, hi = range.high_s;
  if (lo === null || hi === null || !Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return secondsLabel(range.seconds);
  if (hi < 1) return `${Math.max(1, Math.round(lo * 1000))}–${Math.max(1, Math.round(hi * 1000))} ms`;
  if (hi < 60) return `${lo < 10 ? lo.toFixed(1) : Math.round(lo)}–${hi < 10 ? hi.toFixed(1) : Math.round(hi)} s`;
  return `${(lo / 60).toFixed(1)}–${(hi / 60).toFixed(1)} min`;
}
/** Same-origin deployment can be a laboratory server; do not call every same-origin host this PC. */
export function computeLocation(endpoint: string, ko: boolean): string {
  let name = "";
  try { name = new URL(endpoint || window.location.origin).hostname; } catch { /* SSR / unavailable */ }
  return /^(localhost|127(?:\.\d{1,3}){3}|\[?::1\]?)$/.test(name)
    ? (ko ? "이 PC" : "This PC") : (ko ? "계산 서버" : "Compute server");
}
