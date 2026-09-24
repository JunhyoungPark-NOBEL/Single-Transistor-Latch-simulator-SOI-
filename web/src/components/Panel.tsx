// Result panel card: title + "상세" + toolbar (CSV/PNG), loading skeleton / progress overlay, empty
// state, inline errors, stale + demo badges. The plot is optional (tables can be passed as children).
import type { Data, Layout } from "plotly.js";
import { useRef, type ReactNode } from "react";
import type { TopicId } from "../content/physics";
import { useT } from "../i18n";
import { palette } from "../plots/theme";
import { useStore, type ResultEntry } from "../state/store";
import { downloadText, tracesToCsv } from "../utils/csv";
import { fmtDuration } from "../utils/format";
import { DetailsButton } from "./DetailsButton";
import { IconChart, IconDownload, IconImage } from "./icons";
import { exportPlotPng, Plot } from "./Plot";

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
}

export function Progress({ value, indeterminate }: { value: number; indeterminate?: boolean }) {
  return (
    <div className={`progress${indeterminate ? " indet" : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}>
      <span style={{ width: `${Math.max(3, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

export function Panel(p: PanelProps) {
  const t = useT();
  const theme = useStore((s) => s.theme);
  const gdRef = useRef<HTMLElement | null>(null);
  const e = p.entry;
  const running = !!e && (e.status === "running" || e.status === "queued");
  const stale = !!(p.hasData && p.currentKey && e && "dataKey" in e && e.dataKey && e.dataKey !== p.currentKey && !running);
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

  return (
    <section className={`panel${p.wide ? " wide" : ""}`} aria-labelledby={`panel-${p.id}-title`} data-testid={`panel-${p.id}`} aria-busy={running}>
      <header className="panel-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 className="panel-title" id={`panel-${p.id}-title`} style={{ margin: 0 }}>
            {p.title}
            {p.badges}
            {e?.mock && p.hasData && <span className="badge demo">{t("demo")}</span>}
            {stale && <span className="badge stale" title={t("stale")}>{t("stale").split("—")[0].trim()}</span>}
          </h3>
          {p.desc && <div className="panel-desc">{p.desc}</div>}
        </div>
        {p.topic && <DetailsButton topic={p.topic} testId={`details-panel-${p.id}`} />}
      </header>
      {(p.toolbar || (p.plot && p.hasData)) && (
        <div className="panel-toolbar">
          {p.toolbar}
          <span className="spacer" />
          {p.plot && p.hasData && (
            <>
              <button type="button" className="btn sm ghost" onClick={csv} aria-label={t("csv.aria")} title={t("csv.aria")}>
                <IconDownload size={13} /> {t("csv")}
              </button>
              <button type="button" className="btn sm ghost" onClick={png} aria-label={t("png.aria")} title={t("png.aria")}>
                <IconImage size={13} /> {t("png")}
              </button>
            </>
          )}
        </div>
      )}
      {err && (
        <div className="panel-foot">
          <div className="err-box" role="alert">
            <strong>{t("error")}:</strong> {err}
          </div>
        </div>
      )}
      <div className="panel-body">
        {p.plot && p.hasData ? (
          <div className={running ? "dim" : undefined}>
            <Plot data={p.plot.data} layout={p.plot.layout} className={p.plot.className ?? "plot"} onGraph={(gd) => (gdRef.current = gd)} />
          </div>
        ) : p.plot ? (
          running ? (
            <div className="skeleton-plot" />
          ) : (
            <div className="empty">
              <div className="empty-inner">
                <IconChart size={28} className="empty-icon" />
                {p.empty ?? <span>{t("empty.run")}</span>}
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
      {p.children}
      {p.warnings && p.warnings.length > 0 && (
        <div className="panel-foot">
          <ul className="warn-list" aria-label={t("warnings")}>
            {p.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
