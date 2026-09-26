// Statistics module (docs/WEB_CONTRACT.md §9): pure statistics + the shared table component.
export * from "./describe";
export { fmtCoef, fmtLevel, fmtP, fmtPct, fmtShare, fmtSpread, levelDecimals, planUnits, type UnitPlan } from "./format";
export { computeRow, computeRows, statsCsv, type ComputedRow, type StatsColumn, type StatsGroup, type StatsRow } from "./table";
export { StatsTable, type StatsTableProps } from "./StatsTable";
