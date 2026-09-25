// Static snapshot mode (api/snapshot.ts): per-panel notices ("nearest precomputed V_G" / "not in the snapshot →
// example data") and the global banner. Everything renders nothing outside snapshot mode.
import { snapshotApprox, type SnapshotApprox } from "../api/snapshot";
import type { Health } from "../api/types";
import { useT, type T } from "../i18n";
import { useStore } from "../state/store";

/** Status-dot text in snapshot mode: number of precomputed results and the recording date. */
export function snapshotStatusText(t: T, health: Health | null): string {
  const info = (health as { snapshot?: { n?: number; created?: string } } | null)?.snapshot;
  return t("snapshot.status", { n: info?.n ?? "?", date: (info?.created ?? "").slice(0, 10) || "?" });
}

const num = (v: number, d: number) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d).replace("-", "−");

export function nearText(t: T, a: SnapshotApprox): string {
  const p = a.power_mW;
  const rp = a.requested.power_mW;
  if (p === null || rp === null || Math.abs(p - rp) < 1e-9) return t("snapshot.near.vg", { vg: num(a.vg, 2), req: num(a.requested.vg, 2) });
  return t("snapshot.near.vgp", { vg: num(a.vg, 2), p: num(p, 2), rvg: num(a.requested.vg, 2), rp: num(rp, 2) });
}

/**
 * Shown in a result panel while the page runs from the snapshot: a note when the data is the nearest
 * precomputed V_G (`data` marked by the snapshot backend), a warning when it is demo data (`show`).
 */
export function SnapshotMissNotice({ show, data }: { show: boolean; data?: unknown }) {
  const t = useT();
  const snapshot = useStore((s) => s.backend === "snapshot");
  if (!snapshot) return null;
  const approx = show ? undefined : snapshotApprox(data);
  if (!show && !approx) return null;
  return (
    <div className="panel-foot">
      {show ? (
        <div className="callout warn" role="status" data-testid="snapshot-miss">
          {t("snapshot.miss")}
        </div>
      ) : (
        <div className="callout info" role="status" data-testid="snapshot-near">
          {nearText(t, approx!)}
        </div>
      )}
    </div>
  );
}

/** Global banner in snapshot mode; adds a line while results on screen are approximate or example data. */
export function SnapshotBanner() {
  const t = useT();
  const snapshot = useStore((s) => s.backend === "snapshot");
  const anyMiss = useStore((s) => Object.values(s.results).some((r) => r.status === "done" && r.mock));
  const anyNear = useStore((s) => Object.values(s.results).some((r) => r.status === "done" && !!snapshotApprox(r.data)));
  if (!snapshot) return null;
  return (
    <div className="banner snapshot" role="status" data-testid="snapshot-banner">
      <span className="dot snapshot" aria-hidden />
      <span>
        <strong>{t("snapshot.banner.title")}</strong> — {t("snapshot.banner")}
        {anyNear && (
          <>
            {" "}
            <em className="near" data-testid="snapshot-banner-near">{t("snapshot.banner.near")}</em>
          </>
        )}
        {anyMiss && (
          <>
            {" "}
            <em data-testid="snapshot-banner-some">{t("snapshot.banner.some")}</em>
          </>
        )}
      </span>
    </div>
  );
}
