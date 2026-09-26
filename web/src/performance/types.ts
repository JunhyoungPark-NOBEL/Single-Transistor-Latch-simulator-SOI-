import type { Kind } from "../api/types";

export interface EstimateJob { kind: Kind; payload: unknown; key: string; depends_on?: string[] }
export interface TimeRange { low_s: number | null; seconds: number | null; high_s: number | null }
export interface ComputeHost { id: string; instance_id: string; engine_version: string; label: string; os: string; architecture: string; python: string; workers: number; cpu_count: number }
export interface PerformanceEstimate {
  host: ComputeHost;
  items: { key: string; kind: string; model: string; family: string; supported: boolean; cached: boolean; joined?: boolean; estimate: TimeRange; source: string; confidence: string }[];
  total: TimeRange; compute: TimeRange;
  setup: TimeRange & { unknown?: boolean; note?: string };
  queue: Record<string, unknown>;
  cached: boolean; confidence: "low" | "medium" | "high";
  source: "reference" | "calibrated" | "observed"; warnings: string[];
}
export interface PerformanceData {
  catalog: Record<string, unknown>; host?: ComputeHost;
  calibration?: Record<string, unknown>; observations?: unknown[]; queue?: Record<string, unknown>;
}
