// Validation tab, 8 illumination conditions: per-condition model vs measured V_LU statistics (mean, SD,
// Δmean, SD ratio, two-sample KS) under the ⟨V_LU⟩/σ figure. Rows appear as the condition runs finish.
import { useMemo } from "react";
import type { SweepMCResult } from "../api/types";
import { usePalette } from "../device/common";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { StatsTable, type StatsColumn, type StatsRow } from "../stats/StatsTable";

const COLUMNS: StatsColumn[] = ["mean", "sd", "m_mean", "m_sd", "dmean", "sd_ratio", "ks_d", "ks_p"];

export function PhotoConditionsStats() {
  const t = useT();
  const c = usePalette();
  const measured = useStore((s) => s.measured.data?.photo);
  const conds = useStore((s) => s.meta.measured_photo_conditions);
  const results = useStore((s) => s.results);
  const rows = useMemo<StatsRow[]>(() => {
    const list = conds ?? [];
    const vgs = [...new Set(list.map((x) => x.vg))];
    const out: StatsRow[] = [];
    list.forEach((cd, k) => {
      const r = results[`val_photo_${k}`];
      const data = r?.status === "done" ? (r.data as SweepMCResult | undefined) : undefined;
      if (!data?.V_LU) return;
      const m = measured?.find((x) => Math.abs(x.vg - cd.vg) < 1e-6 && Math.abs(x.power_mW - cd.power_mW) < 5e-3);
      const gi = vgs.indexOf(cd.vg);
      out.push({
        key: `c${k}`,
        label: t("stats.val.cond", { vg: cd.vg.toFixed(1).replace("-", "−"), p: cd.power_mW.toFixed(2) }),
        unit: "V",
        values: data.V_LU,
        measured: m?.raw?.length ? m.raw : data.measured?.V_LU ?? null,
        color: gi === 0 ? c.sto : c.categorical[1],
      });
    });
    return out;
  }, [conds, results, measured, c, t]);
  if (!rows.length) return null;
  return (
    <div style={{ padding: "0 14px 12px" }}>
      <StatsTable
        rows={rows}
        columns={COLUMNS}
        measuredLayout="columns"
        csvName="validation_photo_conditions_stats"
        testId="val-photo-stats"
        caption={<strong style={{ color: "var(--text-2)" }}>{t("stats.val.title")}</strong>}
        optional={[]}
      />
      <div className="small muted" style={{ marginTop: 4 }}>{t("stats.val.desc")}</div>
    </div>
  );
}
