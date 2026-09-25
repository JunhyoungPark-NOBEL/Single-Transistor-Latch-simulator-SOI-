// Answer bar of the Device tab: three cells in one card — V_LU (turn-on, --hrs), V_LD (turn-off, --lrs) and the
// window ΔV (--text), all in V with 3 decimals so 3.704 − 2.598 = 1.106 reads directly. Deterministic sub-line:
// the change against the previous completed run ("▲ +80.0 mV · 이전 대비", state/prevRuns.ts), else the plain
// meaning. Stochastic: mean ± σ, sub-line = the measured record (like for like) or the deterministic fold.
// Stale values dim, with one "변경됨" badge. The [간단히 | 모두 보기] switch sits at the right end.
import { useMemo, type ReactNode } from "react";
import type { BranchesResult, SweepMCResult, VgCurveResult } from "../api/types";
import { LayoutToggle } from "../components/LayoutToggle";
import { useT, type T } from "../i18n";
import { DEV } from "../i18n/strings.device";
import { fill, UX } from "../i18n/strings.ux";
import { signed, subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { usePrevRun } from "../state/prevRuns";
import { useStore } from "../state/store";
import { fmtDuration, fmtSI, isNum } from "../utils/format";
import { branchesPayload, sweepMcPayload } from "../utils/payload";
import { isStale, useCurrentKey, useEntry } from "./common";
import "./device.css";

const v3 = (v: number | null | undefined) => (isNum(v) ? v.toFixed(3) : "—");
/** Spread in mV with 3 significant digits (116 mV, 19.4 mV, 4.21 mV). */
const sdMV = (v: number | null | undefined) => {
  if (!isNum(v)) return "—";
  const m = v * 1e3;
  return Math.abs(m) >= 100 ? m.toFixed(0) : Math.abs(m) >= 10 ? m.toFixed(1) : m.toFixed(2);
};

/** "▲ +80.0 mV" / "▼ −1.2 mV" (null below 0.05 mV = no change). */
export function deltaText(d: number): { text: string; dir: "up" | "down" } | null {
  const mv = d * 1e3;
  if (!Number.isFinite(mv) || Math.abs(mv) < 0.05) return null;
  const abs = Math.abs(mv);
  const num = abs >= 1000 ? `${(abs / 1e3).toFixed(3)} V` : `${abs.toFixed(1)} mV`;
  return mv > 0 ? { text: `▲ +${num}`, dir: "up" } : { text: `▼ −${num}`, dir: "down" };
}

function Delta({ id, t, now, before }: { id: string; t: T; now: number | null | undefined; before: number | null | undefined }) {
  if (!isNum(now) || !isNum(before)) return null;
  const d = deltaText(now - before);
  if (!d)
    return (
      <span className="ans-delta flat" data-testid={`kpi-${id}-delta`}>
        {t.l(DEV["kpi.nochange"])}
      </span>
    );
  // "{d} · 이전 대비": the number in the mono chip, the words in the text font
  const [before_, after] = t.l(DEV["kpi.delta"]).split("{d}");
  return (
    <span className={`ans-delta ${d.dir}`} data-testid={`kpi-${id}-delta`} title={fill(t.l(DEV["kpi.delta.aria"]), { d: d.text.slice(2) })}>
      {before_ && <span className="ans-delta-note">{before_}</span>}
      <span className="ans-delta-num">{d.text}</span>
      {after && <span className="ans-delta-note">{after}</span>}
    </span>
  );
}

interface CellProps {
  id: string;
  label: string;
  sym: ReactNode;
  value: ReactNode;
  unit?: string;
  pm?: ReactNode;
  sub?: ReactNode;
  color: string;
  loading?: boolean;
  title?: string;
}

function Cell({ id, label, sym, value, unit, pm, sub, color, loading, title }: CellProps) {
  return (
    <div className={`ans-cell${loading ? " loading" : ""}`} data-testid={`kpi-${id}`} title={title} style={{ "--ans-color": color } as React.CSSProperties}>
      <div className="ans-label">
        <span>{label}</span> <span className="ans-sym">{sym}</span>
      </div>
      <div className="ans-value" data-testid={`kpi-${id}-value`}>
        <span className="ans-num">{value}</span>
        {unit && <span className="u">{unit}</span>}
        {pm && <span className="ans-pm">{pm}</span>}
      </div>
      <div className="ans-sub">{sub ?? " "}</div>
    </div>
  );
}

const SYM = {
  vlu: (
    <>
      V<sub>LU</sub>
    </>
  ),
  vld: (
    <>
      V<sub>LD</sub>
    </>
  ),
  win: <>ΔV</>,
};

export function KpiStrip() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const params = useStore((s) => s.params);
  const { entry: be, data: br } = useEntry<BranchesResult>("branches");
  const { entry: me, data: mc } = useEntry<SweepMCResult>("sweep_mc");
  const prevBr = usePrevRun<BranchesResult>("branches");
  const vgc = useStore((s) => s.results.vg_curve?.data as VgCurveResult | undefined);
  const vgNow = params.device.vg;
  const prevMc = usePrevRun<SweepMCResult>("sweep_mc");
  const bKey = useCurrentKey("branches", useMemo(() => branchesPayload(params), [params]));
  const mKey = useCurrentKey("sweep_mc", useMemo(() => sweepMcPayload(params), [params]));
  // values from an older parameter set are dimmed (same rule as the panels' "변경됨" badge)
  const stale = mode === "deterministic" ? isStale(be, bKey) : isStale(me, mKey);
  const staleBadge = stale ? (
    <span className="badge stale ans-badge" title={t("stale")}>
      {t.l(UX["badge.changed"])}
    </span>
  ) : null;
  const runtime = (s: number | undefined, cached?: boolean) =>
    isNum(s) ? `${fill(t.l(DEV["kpi.tip.runtime"]), { t: fmtDuration(s) })}${cached ? ` (${t.l(DEV["kpi.tip.cached"])})` : ""}` : "";

  let cells: ReactNode;
  if (mode === "deterministic") {
    const loading = !br && (be?.status === "running" || be?.status === "queued");
    const f = br?.folds;
    const noLatch = !!br && !br.latch;
    const win = f ? (f.window_V ?? (isNum(f.V_LU) && isNum(f.V_LD) ? f.V_LU - f.V_LD : null)) : null;
    const pf = prevBr?.data.folds;
    const pWin = pf ? (pf.window_V ?? (isNum(pf.V_LU) && isNum(pf.V_LD) ? pf.V_LU - pf.V_LD : null)) : null;
    const hasPrev = !!pf && !!br && !noLatch && prevBr!.data.latch;
    const tip = (lines: (string | false | null | undefined)[]) => lines.filter(Boolean).join("\n") || undefined;
    const common = br ? [isNum(br.iph_A) ? `I_PH = ${fmtSI(br.iph_A, "A", 3)}` : null, runtime(br.runtime_s, be?.cached)] : [];
    const noLatchVal = <span className="ans-nolatch">{t.l(DEV["kpi.nolatch"])}</span>;
    // no latch: say where the latch window is (from the V_G curve), or that the sweep stops too early
    const w = vgc?.window;
    const noLatchHint =
      w && isNum(w.vg_low) && isNum(w.vg_high)
        ? vgNow < w.vg_low || vgNow > w.vg_high
          ? fill(t.l(DEV["kpi.nolatch.window"]), { lo: signed(w.vg_low, 2), hi: signed(w.vg_high, 2) })
          : t.l(DEV["kpi.nolatch.vdmax"])
        : t.l(DEV["kpi.nolatch.hint"]);
    const sub = (id: string, meaning: string, now: number | null | undefined, before: number | null | undefined) =>
      noLatch ? (
        id === "vlu" ? (
          <span className="ans-hint">
            <SubText text={subs(noLatchHint)} />
          </span>
        ) : undefined
      ) : hasPrev ? (
        <Delta id={id} t={t} now={now} before={before} />
      ) : (
        <span className="ans-meaning">
          <SubText text={subs(meaning)} />
        </span>
      );
    cells = (
      <>
        <Cell
          id="vlu"
          label={t.l(DEV["kpi.vlu"])}
          sym={SYM.vlu}
          color="var(--hrs)"
          loading={loading}
          value={noLatch ? noLatchVal : v3(f?.V_LU)}
          unit={noLatch || !br ? undefined : "V"}
          sub={sub("vlu", t.l(DEV["kpi.meaning.vlu"]), f?.V_LU, pf?.V_LU)}
          title={f ? tip([isNum(f.u_LU) && `u_LU = ${v3(f.u_LU)} V`, isNum(f.I_LU) && `I_LU = ${fmtSI(f.I_LU, "A", 3)}`, ...common]) : undefined}
        />
        <Cell
          id="vld"
          label={t.l(DEV["kpi.vld"])}
          sym={SYM.vld}
          color="var(--lrs)"
          loading={loading}
          value={noLatch ? noLatchVal : v3(f?.V_LD)}
          unit={noLatch || !br ? undefined : "V"}
          sub={sub("vld", t.l(DEV["kpi.meaning.vld"]), f?.V_LD, pf?.V_LD)}
          title={f ? tip([isNum(f.u_LD) && `u_LD = ${v3(f.u_LD)} V`, isNum(f.I_LD) && `I_LD = ${fmtSI(f.I_LD, "A", 3)}`, ...common]) : undefined}
        />
        <Cell
          id="window"
          label={t.l(DEV["kpi.window"])}
          sym={SYM.win}
          color="var(--text)"
          loading={loading}
          value={noLatch ? "—" : v3(win)}
          unit={isNum(win) && !noLatch ? "V" : undefined}
          sub={
            noLatch ? undefined : hasPrev ? (
              <Delta id="window" t={t} now={win} before={pWin} />
            ) : (
              <span className="ans-meaning">
                <SubText text={subs(t.l(DEV["kpi.meaning.window"]))} />
              </span>
            )
          }
        />
      </>
    );
  } else {
    const loading = !mc && (me?.status === "running" || me?.status === "queued");
    const lu = mc?.stats.LU;
    const ld = mc?.stats.LD;
    const win = isNum(lu?.mean) && isNum(ld?.mean) ? lu!.mean! - ld!.mean! : null;
    const m = mc?.measured;
    const mLu = m?.stats.LU;
    const mLd = m?.stats.LD;
    const mWin = isNum(mLu?.mean) && isNum(mLd?.mean) ? mLu!.mean! - mLd!.mean! : null;
    const fold = (v: number | null | undefined) => (isNum(v) ? fill(t.l(DEV["kpi.fold"]), { v: `${v3(v)}\u00a0V` }) : undefined);
    const meas = (mean: number | null | undefined, sd?: number | null) =>
      isNum(mean) ? fill(t.l(DEV["kpi.measured"]), { v: `${v3(mean)}\u00a0V${isNum(sd) ? ` ±\u00a0${sdMV(sd)}\u00a0mV` : ""}` }) : undefined;
    const pl = prevMc?.data.stats;
    const prevWin = pl && isNum(pl.LU.mean) && isNum(pl.LD.mean) ? pl.LU.mean - pl.LD.mean : null;
    // sub-line: the measured record (like for like), else the change vs the previous run, else the fold
    const meaning = { vlu: "kpi.meaning.vlu", vld: "kpi.meaning.vld", window: "kpi.meaning.window" } as const;
    const sub = (id: keyof typeof meaning, measured: string | undefined, now: number | null | undefined, before: number | null | undefined, fb: string | undefined) =>
      !mc ? (
        <span className="ans-meaning">
          <SubText text={subs(t.l(DEV[meaning[id]]))} />
        </span>
      ) : measured ? (
        <span className="ans-meaning">{measured}</span>
      ) : pl && isNum(before) ? (
        <Delta id={id} t={t} now={now} before={before} />
      ) : fb ? (
        <span className="ans-meaning">{fb}</span>
      ) : undefined;
    const nCyc = mc ? mc.V_LU.length : 0;
    const tipSto = mc
      ? [`n = ${nCyc}`, fill(t.l(DEV["kpi.tip.noLu"]), { n: lu?.censored ?? 0 }), runtime(mc.runtime_s, me?.cached)].filter(Boolean).join("\n")
      : undefined;
    const labelMean = (k: keyof typeof DEV) => `${t.l(DEV[k])} · ${t.l(DEV["kpi.mean"])}`;
    cells = (
      <>
        <Cell
          id="vlu"
          label={labelMean("kpi.vlu")}
          sym={SYM.vlu}
          color="var(--hrs)"
          loading={loading}
          value={mc ? v3(lu?.mean) : "—"}
          unit={mc && isNum(lu?.mean) ? "V" : undefined}
          pm={mc && isNum(lu?.sd) ? `±\u00a0${sdMV(lu!.sd)}\u00a0mV` : undefined}
          sub={sub("vlu", meas(mLu?.mean, mLu?.sd), lu?.mean, pl?.LU.mean, fold(mc?.centre.V_LU ?? br?.folds.V_LU))}
          title={tipSto}
        />
        <Cell
          id="vld"
          label={labelMean("kpi.vld")}
          sym={SYM.vld}
          color="var(--lrs)"
          loading={loading}
          value={mc ? v3(ld?.mean) : "—"}
          unit={mc && isNum(ld?.mean) ? "V" : undefined}
          pm={mc && isNum(ld?.sd) ? `±\u00a0${sdMV(ld!.sd)}\u00a0mV` : undefined}
          sub={sub("vld", meas(mLd?.mean, mLd?.sd), ld?.mean, pl?.LD.mean, fold(mc?.centre.V_LD ?? br?.folds.V_LD))}
          title={tipSto}
        />
        <Cell
          id="window"
          label={labelMean("kpi.window")}
          sym={SYM.win}
          color="var(--text)"
          loading={loading}
          value={mc ? v3(win) : "—"}
          unit={isNum(win) ? "V" : undefined}
          sub={sub(
            "window",
            isNum(mWin) ? fill(t.l(DEV["kpi.measured"]), { v: `${v3(mWin)}\u00a0V` }) : undefined,
            win,
            prevWin,
            mc && isNum(mc.centre.V_LU) && isNum(mc.centre.V_LD) ? fold(mc.centre.V_LU - mc.centre.V_LD) : undefined,
          )}
          title={tipSto}
        />
      </>
    );
  }

  return (
    <div className="answer-row">
      <div className={`answer-bar${stale ? " stale" : ""}`} data-testid="kpis" role="group" aria-label={t.l(DEV["kpi.label"])}>
        {cells}
        {staleBadge}
      </div>
      <LayoutToggle className="answer-layout" />
    </div>
  );
}
