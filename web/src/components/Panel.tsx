// Result panel card: title + "상세" + toolbar (CSV/PNG), loading skeleton / progress overlay, empty
// state, inline errors, stale + demo badges. The plot is optional (tables can be passed as children).
// Inside a MoreCard tab (MoreCardContext.embedded) the card border goes and the h3 is visually hidden.
import type { Data, Layout } from "plotly.js";
import { useContext, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import type { TopicId } from "../content/physics";
import { useT } from "../i18n";
import { UX } from "../i18n/strings.ux";
import { palette } from "../plots/theme";
import { useStore, type ResultEntry } from "../state/store";
import { downloadText, tracesToCsv } from "../utils/csv";
import { fmtDuration } from "../utils/format";
import { DetailsButton } from "./DetailsButton";
import { ErrorBoundary } from "./ErrorBoundary";
import { IconChart, IconCheck, IconDownload, IconImage, IconMore, IconPlay } from "./icons";
import { MoreCardContext } from "./MoreCard";
import { runCurrent } from "../state/runner";
import { exportPlotPng, Plot } from "./Plot";
import { SnapshotMissNotice } from "./SnapshotNotice";

/** Section of the ⋯ menu: 보기 (view toggles) or 내보내기 (export actions). */
export type PanelMenuSection = "view" | "export";

interface PanelMenuBase {
  /** Unique within the menu (React key). */
  id: string;
  /** Localised label. */
  label: string;
  /** Default: "view" for check/radio items, "export" for actions. */
  section?: PanelMenuSection;
  disabled?: boolean;
  title?: string;
  testId?: string;
}

/**
 * One item of a panel's ⋯ menu: a checkbox (e.g. ☑ 측정), a radio (consecutive items with the same `group`
 * form one set, `group` is its caption, e.g. 로그 | 선형) or an action (e.g. CSV). Check and radio items keep
 * the menu open; an action closes it.
 */
export type PanelMenuItem =
  | (PanelMenuBase & { kind: "check"; checked: boolean; onChange: (checked: boolean) => void })
  | (PanelMenuBase & { kind: "radio"; group: string; checked: boolean; onSelect: () => void })
  | (PanelMenuBase & { kind: "action"; onSelect: () => void; icon?: ReactNode });

const sectionOf = (it: PanelMenuItem): PanelMenuSection => it.section ?? (it.kind === "action" ? "export" : "view");

export interface PanelProps {
  id: string;
  title: string;
  desc?: string;
  topic?: TopicId;
  entry?: ResultEntry | { status: ResultEntry["status"]; progress?: number; message?: string; error?: string; mock?: boolean; dataKey?: string };
  hasData: boolean;
  currentKey?: string;
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
  /** Hero panel of a screen (FocusLayout). Only a primary panel will show the big empty-state Run button (WP1). */
  primary?: boolean;
  /** ⋯ menu items (trigger `panel-menu-<id>`): 보기 toggles and 내보내기 actions. */
  menu?: PanelMenuItem[];
  /** One muted footnote line under the plot (run metadata: fold currents, range, KS, solver stats …). */
  foot?: ReactNode;
}

export function Progress({ value, indeterminate }: { value: number; indeterminate?: boolean }) {
  return (
    <div className={`progress${indeterminate ? " indet" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}>
      <span style={{ width: `${Math.max(3, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

/** Informational server warnings: a small collapsible notice (most runs carry a few). */
export function Notices({ items }: { items: string[] }) {
  const t = useT();
  return (
    <div className="panel-foot">
      <details className="notices" data-testid="notices">
        <summary>
          {t("warnings")} · {items.length}
        </summary>
        <ul>
          {items.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      </details>
    </div>
  );
}

/** Minimal ⋯ menu (role=menu): sections 보기 / 내보내기, arrow keys, Esc and outside click close it.
 *  Step 0 placeholder: WP1 replaces it with components/PanelMenu.tsx (same PanelMenuItem contract). */
function BasicPanelMenu({ panelId, items }: { panelId: string; items: PanelMenuItem[] }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement | null>(null);
  const trigger = useRef<HTMLButtonElement | null>(null);
  const menuId = useId();

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    wrap.current?.querySelector<HTMLButtonElement>(".pmenu-item:not(:disabled)")?.focus();
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  const close = (refocus: boolean) => {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation(); // Esc closes the menu before any window around it
      close(true);
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const els = [...(wrap.current?.querySelectorAll<HTMLButtonElement>(".pmenu-item:not(:disabled)") ?? [])];
    if (!els.length) return;
    const i = els.indexOf(document.activeElement as HTMLButtonElement);
    const n = els.length;
    const next = e.key === "Home" ? 0 : e.key === "End" ? n - 1 : e.key === "ArrowDown" ? (i + 1) % n : (i - 1 + n) % n;
    els[next].focus();
  };

  const sections: { id: PanelMenuSection; title: string; items: PanelMenuItem[] }[] = (
    [
      { id: "view", title: t.l(UX["menu.view"]), items: items.filter((it) => sectionOf(it) === "view") },
      { id: "export", title: t.l(UX["menu.export"]), items: items.filter((it) => sectionOf(it) === "export") },
    ] as const
  ).filter((sec) => sec.items.length > 0);

  const renderItem = (it: PanelMenuItem) => {
    const common = { type: "button" as const, className: "pmenu-item", disabled: it.disabled, title: it.title, "data-testid": it.testId };
    if (it.kind === "check")
      return (
        <button key={it.id} {...common} role="menuitemcheckbox" aria-checked={it.checked} onClick={() => it.onChange(!it.checked)}>
          <span className={`pmenu-mark pm-box${it.checked ? " is-on" : ""}`} aria-hidden>
            {it.checked && <IconCheck size={11} />}
          </span>
          {it.label}
        </button>
      );
    if (it.kind === "radio")
      return (
        <button key={it.id} {...common} role="menuitemradio" aria-checked={it.checked} onClick={() => it.onSelect()}>
          <span className={`pmenu-mark pm-radio${it.checked ? " is-on" : ""}`} aria-hidden />
          {it.label}
        </button>
      );
    return (
      <button key={it.id} {...common} role="menuitem" onClick={() => { it.onSelect(); close(true); }}>
        <span className="pmenu-mark pm-icon" aria-hidden>
          {it.icon}
        </span>
        {it.label}
      </button>
    );
  };

  return (
    <div
      className="pmenu-wrap"
      ref={wrap}
      onBlur={(e) => {
        if (open && wrap.current && !wrap.current.contains(e.relatedTarget as Node | null)) setOpen(false);
      }}
    >
      <button
        ref={trigger}
        type="button"
        className="icon-btn xs"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={t.l(UX["menu.open"])}
        title={t.l(UX["menu.open"])}
        data-testid={`panel-menu-${panelId}`}
        onClick={() => setOpen((o) => !o)}
      >
        <IconMore size={14} />
      </button>
      {open && (
        <div className="pmenu" role="menu" id={menuId} aria-label={t.l(UX["menu.open"])} onKeyDown={onKey}>
          {sections.map((sec) => (
            <div key={sec.id} className="pmenu-sec" role="group" aria-label={sec.title}>
              <div className="pmenu-title" aria-hidden>
                {sec.title}
              </div>
              {sec.items.map((it, i) => {
                const prev = sec.items[i - 1];
                const caption = it.kind === "radio" && !(prev?.kind === "radio" && prev.group === it.group) ? it.group : null;
                return (
                  <div key={it.id} role="none">
                    {caption && (
                      <div className="pmenu-group" aria-hidden>
                        {caption}
                      </div>
                    )}
                    {renderItem(it)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function Panel(p: PanelProps) {
  const t = useT();
  const theme = useStore((s) => s.theme);
  const { embedded } = useContext(MoreCardContext);
  const gdRef = useRef<HTMLElement | null>(null);
  const e = p.entry;
  const running = !!e && (e.status === "running" || e.status === "queued");
  const stale = p.stale ?? !!(p.hasData && p.currentKey && e && "dataKey" in e && e.dataKey && e.dataKey !== p.currentKey && !running);
  const err = p.error ?? (e?.status === "error" ? e.error : null);
  const progress = e?.progress ?? 0;
  const elapsed = e && "elapsed" in e ? (e as ResultEntry).elapsed : undefined;

  const csv = () => {
    if (!p.plot) return;
    downloadText(`${p.csvName ?? p.id}.csv`, tracesToCsv(p.plot.data as never));
  };
  const png = () => {
    if (!gdRef.current) return;
    void exportPlotPng(gdRef.current, p.csvName ?? p.id, palette(theme).surface);
  };
  const badges = (
    <>
      {p.badges}
      {e?.mock && p.hasData && <span className="badge demo">{t("demo")}</span>}
      {stale && <span className="badge stale" title={t("stale")}>{t("stale").split("—")[0].trim()}</span>}
    </>
  );

  return (
    <section className={`panel${p.wide ? " wide" : ""}${p.primary ? " primary" : ""}${embedded ? " embedded" : ""}`} aria-labelledby={`panel-${p.id}-title`} data-testid={`panel-${p.id}`} aria-busy={running}>
      <header className="panel-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          {embedded ? (
            <>
              {/* the MoreCard tab label is the visible title */}
              <h3 className="panel-title sr-only" id={`panel-${p.id}-title`}>
                {p.title}
              </h3>
              <div className="panel-badges">{badges}</div>
            </>
          ) : (
            <h3 className="panel-title" id={`panel-${p.id}-title`} style={{ margin: 0 }}>
              <span className="pt-text">{p.title}</span>
              {badges}
            </h3>
          )}
          {p.desc && <div className="panel-desc">{p.desc}</div>}
        </div>
        <div className="panel-actions">
          {p.plot && p.hasData && (
            <>
              <button type="button" className="icon-btn xs" onClick={csv} aria-label={t("csv.aria")} title={t("csv.aria")} data-testid={`csv-${p.id}`}>
                <IconDownload size={14} />
              </button>
              <button type="button" className="icon-btn xs" onClick={png} aria-label={t("png.aria")} title={t("png.aria")} data-testid={`png-${p.id}`}>
                <IconImage size={14} />
              </button>
            </>
          )}
          {p.menu && p.menu.length > 0 && <BasicPanelMenu panelId={p.id} items={p.menu} />}
          {p.topic && <DetailsButton topic={p.topic} testId={`details-panel-${p.id}`} />}
        </div>
      </header>
      {p.toolbar && <div className="panel-toolbar">{p.toolbar}</div>}
      <SnapshotMissNotice show={!!e?.mock && p.hasData} data={e && "data" in e ? e.data : undefined} />
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
          <div className={running ? "dim" : undefined}>
            <ErrorBoundary label={p.title} resetKey={p.plot}>
              <Plot data={p.plot.data} layout={p.plot.layout} className={p.plot.className ?? "plot"} onGraph={(gd) => (gdRef.current = gd)} />
            </ErrorBoundary>
          </div>
        ) : !p.hasData ? (
          running ? (
            <div className="skeleton-plot" />
          ) : (
            <div className="empty">
              <div className="empty-inner">
                <IconChart size={28} className="empty-icon" />
                {p.empty ?? (
                  <>
                    <span>{t("empty.run")}</span>
                    {/* the sidebar Run button lives in a drawer below 1100 px — offer it here too */}
                    <button type="button" className="btn primary sm" onClick={() => void runCurrent()} data-testid={`empty-run-${p.id}`}>
                      <IconPlay size={11} /> {t("run")}
                    </button>
                  </>
                )}
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
      {p.foot != null && p.foot !== false && p.foot !== "" && <div className="panel-footnote">{p.foot}</div>}
      {p.warnings && p.warnings.length > 0 && <Notices items={p.warnings} />}
    </section>
  );
}
