import type { PerformanceData } from "./types";
export interface BenchmarkCase {
  id: string; comparison_id: string; model: "detailed" | "simple" | "common" | "mixed";
  kind: string; family: string; label: { ko: string; en: string };
  payload: Record<string, unknown>; features?: Record<string, unknown>;
  supported: boolean; group: string; calibration?: boolean;
  timings?: { median_s: number; min_s: number; max_s: number; p25_s: number; p75_s: number; repeats: number; samples_s: number[]; first_call_s: number };
  notes?: string[]; unsupported_reason?: string;
}
export interface BenchmarkCatalog {
  schema_version: number; generated_at: string; environment: Record<string, unknown>;
  engine_fingerprint: string; methodology: string[]; cases: BenchmarkCase[];
}
export function benchmarkCatalog(data: PerformanceData | null): BenchmarkCatalog | null {
  const catalog = data?.catalog;
  return catalog && Array.isArray(catalog.cases) ? catalog as unknown as BenchmarkCatalog : null;
}
export interface ComparisonRow { id: string; label: { ko: string; en: string }; group: string; detailed?: BenchmarkCase; simple?: BenchmarkCase; common?: BenchmarkCase; mixed?: BenchmarkCase }
export function comparisonRows(cases: BenchmarkCase[]): ComparisonRow[] {
  const rows = new Map<string, ComparisonRow>();
  for (const sample of cases) {
    const id = sample.comparison_id || sample.id;
    const row = rows.get(id) ?? { id, label: sample.label, group: sample.group };
    row[sample.model] = sample;
    rows.set(id, row);
  }
  return [...rows.values()];
}
export function pairedSpeedup(row: ComparisonRow): number | null {
  const a = row.detailed, b = row.simple;
  if (!a?.supported || !b?.supported || !a.timings || !b.timings || b.timings.median_s <= 0) return null;
  return a.timings.median_s / b.timings.median_s;
}
