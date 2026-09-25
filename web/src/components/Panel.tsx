// Result panel card. One 40 px title row: title (+ at most one status badge: 변경됨 > 오류 > 데모), at most one
// visible control (`toolbar`), 📖 (Details) and ⋯ (PanelMenu: 보기 toggles + 내보내기 CSV/PNG). The description
// shows only while the panel is empty (otherwise it is the title's tooltip) — except inside a MoreCard tab,
// where it stays as a one-line caption ("what to look for"), since the tab has no visible title. Below: the
// plot (or table children), then one muted footnote line (`foot`) that also carries the server notices.
// Empty state: a primary (hero) panel offers the big Run button; other panels show a quiet placeholder.
// Inside a MoreCard tab (MoreCardContext.embedded) the card border goes, the h3 is visually hidden (the tab
// label is the title) and the badge/📖/⋯ move into the tab row (MoreSlotContext); the toolbar sits under it.
import type { Data, Layout } from "plotly.js";
import { useContext, useRef, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { TopicId } from "../content/physics";
import { useT } from "../i18n";
import { DEV } from "../i18n/strings.device";
import { UX } from "../i18n/strings.ux";
import { SubText } from "../plots/SubText";
import { palette } from "../plots/theme";
import { runCurrent } from "../state/runner";
import { useStore, type ResultEntry } from "../state/store";
import { downloadText, tracesToCsv } from "../utils/csv";
import { fmtDuration } from "../utils/format";
import { DetailsButton } from "./DetailsButton";
import { ErrorBoundary } from "./ErrorBoundary";
import { IconChart, IconDownload, IconImage, IconPlay } from "./icons";
import { MoreCardContext, MoreSlotContext } from "./MoreCard";
import { PanelMenu, type PanelMenuItem } from "./PanelMenu";
import { exportPlotPng, Plot } from "./Plot";
import { SnapshotMissNotice } from "./SnapshotNotice";

export type { PanelMenuItem, PanelMenuSection } from "./PanelMenu";

/** `meta` of a Plotly trace that is left out of the panel's CSV export (e.g. the grey "이전" ghost curve). */
export const NO_CSV = "no-csv";

export interface PanelProps {
  id: string;
  title: string;
  desc?: string;
  topic?: TopicId;
  entry?: ResultEntry | { status: ResultEntry["status"]; progress?: number; message?: string; error?: string; mock?: boolean; dataKey?: string };
  hasData: boolean;
  currentKey?: string;
  /** At most one visible control (e.g. [로그 | 선형]); everything else belongs in `menu`. */
  toolbar?: ReactNode;
  badges?: ReactNode;
  plot?: { data: Data[]; layout: Partial<Layout>; className?: string };
  csvName?: string;
  empty?: ReactNode;
  children?: ReactNode;
  wide?: boolean;
  warnings?: string[];
  error?: string | null;
  /** Force the "parameters changed" badge (for panels that render a result owned by another panel's entry). */
  stale?: boolean;
  /** Hero panel of a screen (FocusLayout). Only a primary panel shows the big empty-state Run button. */
  primary?: boolean;
  /** ⋯ menu items (trigger `panel-menu-<id>`): 보기 toggles and 내보내기 actions (CSV/PNG are added here). */
  menu?: PanelMenuItem[];
  /** One muted footnote line under the plot (run metadata: fold currents, range, KS, solver stats …). */
  foot?: ReactNode;
  /**
   * A plain click (no drag) inside the plot area, outside any Plotly control: `x` in data units of the x axis
   * (e.g. the I–V hero offers "이 V_D에서 전하 균형 보기").
   */
  onPlotClick?: (pt: { x: number; clientX: number; clientY: number }) => void;
}

export function Progress({ value, indeterminate }: { value: number; indeterminate?: boolean }) {
  return (
    <div className={`progress${indeterminate ? " indet" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}>
      <span style={{ width: `${Math.max(3, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

/** Informational server notes: a small collapsible pill (most runs carry a few). `inline`: at the end of the
 *  footnote line (no second row under the card, so side-by-side cards keep even bottoms). */
export function Notices({ items, inline }: { items: string[]; inline?: boolean }) {
  const t = useT();
  const body = (
    <details className={`notices${inline ? " inline" : ""}`} data-testid="notices">
      <summary>
        {t("warnings")} · {items.length}
      </summary>
      <ul>
        {items.map((w, i) => (
          <li key={i}>
            <SubText text={symbolSubs(w)} />
          </li>
        ))}
      </ul>
    </details>
  );
  return inline ? body : <div className="panel-foot">{body}</div>;
}

const hasFoot = (f: ReactNode) => f != null && f !== false && f !== "";

/**
 * Physics symbols in a description as subscripts: "V_LU" → V<sub>LU</sub>, "V_D,max" → V<sub>D,max</sub>. Only a
 * lone symbol letter (V I Q R C σ) followed by "_" is converted, so code ids such as sweep_mc stay as they are.
 */
export function symbolSubs(text: string): string {
  return text.replace(/(^|[^A-Za-z0-9_])([VIQRCσ])_(\{[^}]+\}|[A-Za-z0-9]+(?:,max|,min)?)/g, (_, pre: string, a: string, b: string) => `${pre}${a}<sub>${b.replace(/^\{|\}$/g, "")}</sub>`);
}

/** The parts of a Plotly axis object used to map a pixel to a value. */
interface AxisPx {
  _offset: number;
  _length: number;
  p2l: (px: number) => number;
}

export function Panel(p: PanelProps) {
  const t = useT();
  const theme = useStore((s) => s.theme);
  const backend = useStore((s) => s.backend);
  const { embedded } = useContext(MoreCardContext);
  const slot = useContext(MoreSlotContext);
  const gdRef = useRef<HTMLElement | null>(null);
  const e = p.entry;
  const running = !!e && (e.status === "running" || e.status === "queued");
  const stale = p.stale ?? !!(p.hasData && p.currentKey && e && "dataKey" in e && e.dataKey && e.dataKey !== p.currentKey && !running);
  const err = p.error ?? (e?.status === "error" ? e.error : null);
  const progress = e?.progress ?? 0;
  const elapsed = e && "elapsed" in e ? (e as ResultEntry).elapsed : undefined;
  // one status badge: 변경됨 > 오류 > 데모. In demo mode (mock/offline) the context strip already says "demo" once;
  // the per-panel badge marks a single fallback result (e.g. a request missing from the static snapshot).
  const demo = !!e?.mock && p.hasData && backend !== "mock" && backend !== "offline";
  const status = stale ? (
    <span className="badge stale" title={t("stale")}>
      {t.l(UX["badge.changed"])}
    </span>
  ) : err ? (
    <span className="badge err">{t.l(UX["status.error"])}</span>
  ) : demo ? (
    <span className="badge demo">{t("demo")}</span>
  ) : null;

  const csv = () => {
    if (!p.plot) return;
    const rows = p.plot.data.filter((d) => (d as { meta?: unknown }).meta !== NO_CSV);
    downloadText(`${p.csvName ?? p.id}.csv`, tracesToCsv(rows as never));
  };
  const png = () => {
    if (!gdRef.current) return;
    void exportPlotPng(gdRef.current, p.csvName ?? p.id, palette(theme).surface);
  };
  const exportItems: PanelMenuItem[] =
    p.plot && p.hasData
      ? [
          { kind: "action", id: "csv", label: t.l(UX["menu.csv"]), title: t("csv.aria"), icon: <IconDownload size={13} />, onSelect: csv, testId: `csv-${p.id}` },
          { kind: "action", id: "png", label: t.l(UX["menu.png"]), title: t("png.aria"), icon: <IconImage size={13} />, onSelect: png, testId: `png-${p.id}` },
        ]
      : [];
  const items = [...(p.menu ?? []), ...exportItems];
  const toolbar = p.toolbar;
  const actions = (
    <>
      {p.topic && <DetailsButton topic={p.topic} testId={`details-panel-${p.id}`} />}
      {items.length > 0 && <PanelMenu panelId={p.id} items={items} />}
    </>
  );
  const showDesc = !!p.desc && !p.hasData && !(p.primary && !running);
  // a MoreCard tab has no visible title: its description stays as a one-line caption once data is in
  const caption = embedded && !!p.desc && p.hasData;

  // plot-area click → data x (Plotly's axis objects: _offset/_length in px, p2l pixel → value). Plotly covers
  // the page while a pointer is down (drag layer), so the release is caught on the document.
  const plotPointerDown = (ev: ReactPointerEvent<HTMLDivElement>) => {
    const cb = p.onPlotClick;
    if (!cb || ev.button !== 0 || (ev.target as Element).closest?.(".modebar, .legend, .annotation-text-g, .infolayer")) return;
    const gd = gdRef.current as (HTMLElement & { _fullLayout?: { xaxis?: AxisPx; yaxis?: AxisPx } }) | null;
    const x0 = ev.clientX;
    const y0 = ev.clientY;
    const up = (u: PointerEvent) => {
      if (!gd || Math.hypot(u.clientX - x0, u.clientY - y0) > 4) return; // a drag (zoom) is not a click
      const xa = gd._fullLayout?.xaxis;
      const ya = gd._fullLayout?.yaxis;
      if (!xa?.p2l || !ya) return;
      const r = gd.getBoundingClientRect();
      const px = x0 - r.left - xa._offset;
      const py = y0 - r.top - ya._offset;
      if (px < 0 || px > xa._length || py < 0 || py > ya._length) return;
      const x = xa.p2l(px);
      if (Number.isFinite(x)) cb({ x, clientX: x0, clientY: y0 });
    };
    document.addEventListener("pointerup", up, { once: true, capture: true });
  };

  const emptyBody = p.empty ?? (
    p.primary ? (
      <>
        {p.desc && (
          <span className="empty-desc">
            <SubText text={symbolSubs(p.desc)} />
          </span>
        )}
        {/* the sidebar Run button lives in a drawer below 1100 px — offer it here too */}
        <button type="button" className="btn primary empty-run" onClick={() => void runCurrent()} data-testid={`empty-run-${p.id}`}>
          <IconPlay size={12} /> {t.l(DEV["empty.runBtn"])}
        </button>
        <span className="empty-hint">{t.l(DEV["empty.runHint"])}</span>
      </>
    ) : (
      <span>{t.l(UX["panel.placeholder"])}</span>
    )
  );

  const cls = `panel${p.wide ? " wide" : ""}${p.primary ? " primary" : ""}${embedded ? " embedded" : ""}${toolbar ? " has-tools" : ""}`;
  return (
    <section className={cls} aria-labelledby={`panel-${p.id}-title`} data-testid={`panel-${p.id}`} aria-busy={running}>
      {embedded ? (
        <>
          {/* the MoreCard tab label is the visible title; badge, 📖 and ⋯ sit at the right end of the tab row */}
          <h3 className="panel-title sr-only" id={`panel-${p.id}-title`}>
            {p.title}
          </h3>
          {slot &&
            createPortal(
              <div className="panel-slot">
                {status}
                {p.badges}
                {actions}
              </div>,
              slot,
            )}
          {(showDesc || caption) && (
            <div className={`panel-desc embedded-desc${caption ? " caption" : ""}`} title={caption ? p.desc : undefined} data-testid={caption ? `panel-caption-${p.id}` : undefined}>
              <SubText text={symbolSubs(p.desc!)} />
            </div>
          )}
          {toolbar && <div className="panel-toolbar">{toolbar}</div>}
        </>
      ) : (
        <header className="panel-head">
          {/* title and the one control share a line while they fit; a wider toolbar wraps under the title */}
          <div className="panel-headmain">
            <h3 className="panel-title" id={`panel-${p.id}-title`} title={p.hasData && p.desc ? p.desc : undefined}>
              <span className="pt-text">
                <SubText text={symbolSubs(p.title)} />
              </span>
              {status}
              {p.badges}
            </h3>
            {toolbar && <div className="panel-tools">{toolbar}</div>}
          </div>
          <div className="panel-actions">{actions}</div>
          {showDesc && (
            <div className="panel-desc">
              <SubText text={symbolSubs(p.desc!)} />
            </div>
          )}
        </header>
      )}
      <SnapshotMissNotice show={demo} data={e && "data" in e ? e.data : undefined} />
      {err && (
        <div className="panel-foot">
          <div className="err-box" role="alert">
            <strong>{t("error")}:</strong> {err}
          </div>
        </div>
      )}
      {(p.plot || !p.hasData || running) && (
        <div className="panel-body">
          {p.plot && p.hasData ? (
            <div
              className={running ? "dim" : undefined}
              onPointerDownCapture={p.onPlotClick ? plotPointerDown : undefined}
            >
              <ErrorBoundary label={p.title} resetKey={p.plot}>
                <Plot data={p.plot.data} layout={p.plot.layout} className={p.plot.className ?? "plot"} onGraph={(gd) => (gdRef.current = gd)} />
              </ErrorBoundary>
            </div>
          ) : !p.hasData ? (
            running ? (
              <div className="skeleton-plot" />
            ) : (
              <div className={`empty${p.primary || p.empty ? "" : " quiet"}`}>
                <div className="empty-inner">
                  <IconChart size={p.primary ? 28 : 22} className="empty-icon" />
                  {emptyBody}
                </div>
              </div>
            )
          ) : null}
          {running && (
            <div className="overlay" aria-live="polite">
              <div className="overlay-card">
                <strong>{t("loading")}</strong>
                <Progress value={progress} indeterminate={progress <= 0.001} />
                <span className="small muted mono">
                  {e?.message || (e?.status === "queued" ? t("run.queued") : "")}
                  {elapsed ? ` · ${fmtDuration(elapsed)}` : ""}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
      {p.children}
      {hasFoot(p.foot) ? (
        <div className="panel-footnote">
          {p.foot}
          {p.warnings && p.warnings.length > 0 && (
            <>
              {" "}
              <Notices items={p.warnings} inline />
            </>
          )}
        </div>
      ) : (
        p.warnings && p.warnings.length > 0 && <Notices items={p.warnings} />
      )}
    </section>
  );
}
