// Static snapshot mode (api/snapshot.ts): per-panel notices ("nearest precomputed V_G" / "not in the snapshot →
// example data") and the details behind the context-strip pill. Everything renders nothing outside snapshot mode.
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

/** Snapshot-mode flags for the status pill: results on screen that are the nearest precomputed V_G, or example data. */
export function useSnapshotState(): { snapshot: boolean; anyMiss: boolean; anyNear: boolean } {
  const snapshot = useStore((s) => s.backend === "snapshot");
  const anyMiss = useStore((s) => s.backend === "snapshot" && Object.values(s.results).some((r) => r.status === "done" && r.mock));
  const anyNear = useStore((s) => s.backend === "snapshot" && Object.values(s.results).some((r) => r.status === "done" && !!snapshotApprox(r.data)));
  return { snapshot, anyMiss, anyNear };
}

/** What the static snapshot is (the popover of the context-strip pill), plus a line while results are approximate. */
export function SnapshotDetails() {
  const t = useT();
  const { anyMiss, anyNear } = useSnapshotState();
  return (
    <>
      <p>
        {t("snapshot.banner.title")} — {t("snapshot.banner")}
      </p>
      {anyNear && (
        <p className="near" data-testid="snapshot-banner-near">
          {t("snapshot.banner.near")}
        </p>
      )}
      {anyMiss && (
        <p className="some" data-testid="snapshot-banner-some">
          {t("snapshot.banner.some")}
        </p>
      )}
    </>
  );
}
