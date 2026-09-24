// KPI strip: folds (V_LU, V_LD, window, I at fold) in deterministic mode; mean ± σ of V_LU/V_LD,
// cycles, censored, runtime in stochastic mode, followed by the full-width statistics summary panel.
import { useMemo, type ReactNode } from "react";
import type { BranchesResult, SweepMCResult } from "../api/types";
import { Tex } from "../components/Tex";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { fmtDuration, fmtSI, isNum } from "../utils/format";
import { branchesPayload, sweepMcPayload } from "../utils/payload";
import { fmtShare } from "../stats/format";
import { isStale, useCurrentKey, useEntry } from "./common";
import { StatsPanel } from "./StoPanels";

export function Kpi({ id, label, sym, value, unit, sub, color, loading }: { id: string; label: string; sym?: string; value: ReactNode; unit?: string; sub?: ReactNode; color?: string; loading?: boolean }) {
  return (
    <div className={`kpi${loading ? " skeleton" : ""}`} style={color ? ({ "--kpi-color": color } as React.CSSProperties) : undefined} data-testid={`kpi-${id}`}>
      <div className="kpi-label">
        {sym && <Tex tex={sym} />}
        <span>{label}</span>
      </div>
      <div className="kpi-value" data-testid={`kpi-${id}-value`}>
        {value}
        {unit && <span className="u">{unit}</span>}
      </div>
      {sub !== undefined && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

const v3 = (v: number | null | undefined) => (isNum(v) ? v.toFixed(3) : "—");
const mV = (v: number | null | undefined) => (isNum(v) ? (v * 1e3).toFixed(1) : "—");

export function KpiStrip() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const params = useStore((s) => s.params);
  const { entry: be, data: br } = useEntry<BranchesResult>("branches");
  const { entry: me, data: mc } = useEntry<SweepMCResult>("sweep_mc");
  const bKey = useCurrentKey("branches", useMemo(() => branchesPayload(params), [params]));
  const mKey = useCurrentKey("sweep_mc", useMemo(() => sweepMcPayload(params), [params]));
  // values from an older parameter set are dimmed (same rule as the panels' "parameters changed" badge)
  const stale = mode === "deterministic" ? isStale(be, bKey) : isStale(me, mKey);
  const cls = `kpis${stale ? " stale" : ""}`;
  const staleSub = stale ? <span className="badge stale">{t("stale").split("—")[0].trim()}</span> : undefined;
  const bLoading = !br && (be?.status === "running" || be?.status === "queued");
  const f = br?.folds;
  const noLatch = br && !br.latch;
  const win = f?.window_V ?? (isNum(f?.V_LU) && isNum(f?.V_LD) ? f!.V_LU! - f!.V_LD! : null);

  if (mode === "deterministic") {
    return (
      <div className={cls} data-testid="kpis" title={stale ? t("stale") : undefined}>
        <Kpi id="vlu" label={t("kpi.vlu")} sym="V_{\mathrm{LU}}" value={noLatch ? t("kpi.nolatch") : v3(f?.V_LU)} unit={noLatch || !br ? undefined : "V"} color="var(--hrs)" loading={bLoading} sub={f?.u_LU != null ? `u = ${v3(f.u_LU)} V` : undefined} />
        <Kpi id="vld" label={t("kpi.vld")} sym="V_{\mathrm{LD}}" value={noLatch ? t("kpi.nolatch") : v3(f?.V_LD)} unit={noLatch || !br ? undefined : "V"} color="var(--lrs)" loading={bLoading} sub={f?.u_LD != null ? `u = ${v3(f.u_LD)} V` : undefined} />
        <Kpi id="window" label={t("kpi.window")} sym="\Delta V" value={mV(win)} unit={isNum(win) ? "mV" : undefined} color="var(--det)" loading={bLoading} sub="V_LU − V_LD" />
        <Kpi id="ifold" label={t("kpi.ifold")} sym="I_{\mathrm{LU}}" value={fmtSI(f?.I_LU, "A", 3)} color="var(--unstable)" loading={bLoading} sub={isNum(f?.I_LD) ? `I_LD = ${fmtSI(f?.I_LD, "A", 3)}` : undefined} />
        <Kpi id="runtime" label={t("kpi.runtime")} value={fmtDuration(br?.runtime_s)} color="var(--border-strong)" loading={bLoading} sub={staleSub ?? (be?.cached ? t("cached") : be?.mock ? t("demo") : isNum(br?.iph_A) ? `I_PH = ${fmtSI(br!.iph_A, "A", 3)}` : undefined)} />
      </div>
    );
  }
  const mLoading = !mc && (me?.status === "running" || me?.status === "queued");
  const lu = mc?.stats.LU;
  const ld = mc?.stats.LD;
  const stoWin = isNum(lu?.mean) && isNum(ld?.mean) ? lu!.mean! - ld!.mean! : null;
  const nCyc = mc ? mc.V_LU.length : 0;
  return (
    <>
    <div className={cls} data-testid="kpis" title={stale ? t("stale") : undefined}>
      <Kpi id="vlu" label={`${t.lang === "ko" ? "래치업" : t("kpi.vlu")} · ${t("kpi.mean")}`} sym="V_{\mathrm{LU}}" value={mc ? <>{v3(lu?.mean)}<span className="u">± {mV(lu?.sd)} mV</span></> : "—"} color="var(--sto)" loading={mLoading} sub={mc || f ? `${t("kpi.fold")}: ${v3(mc?.centre.V_LU ?? f?.V_LU)} V` : undefined} />
      <Kpi id="vld" label={`${t.lang === "ko" ? "래치다운" : t("kpi.vld")} · ${t("kpi.mean")}`} sym="V_{\mathrm{LD}}" value={mc ? <>{v3(ld?.mean)}<span className="u">± {mV(ld?.sd)} mV</span></> : "—"} color="var(--lrs)" loading={mLoading} sub={mc || f ? `${t("kpi.fold")}: ${v3(mc?.centre.V_LD ?? f?.V_LD)} V` : undefined} />
      <Kpi id="window" label={t("kpi.window")} sym="\Delta V" value={mV(stoWin)} unit={isNum(stoWin) ? "mV" : undefined} color="var(--det)" loading={mLoading} sub="⟨V_LU⟩ − ⟨V_LD⟩" />
      <Kpi id="cycles" label={t("kpi.cycles")} value={mc ? String(nCyc) : "—"} color="var(--border-strong)" loading={mLoading} sub={mc ? `${t("kpi.censored")}: ${lu?.censored ?? 0} (${fmtShare(nCyc ? (lu?.censored ?? 0) / nCyc : 0)})` : undefined} />
      <Kpi id="runtime" label={t("kpi.runtime")} value={fmtDuration(mc?.runtime_s)} color="var(--border-strong)" loading={mLoading} sub={staleSub ?? (mc ? `${t("kpi.engine")}: ${mc.engine}${me?.cached ? " · " + t("cached") : ""}` : undefined)} />
    </div>
    <StatsPanel />
    </>
  );
}
