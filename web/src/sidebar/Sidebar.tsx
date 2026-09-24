// Left sidebar: preset selector, grouped parameter cards, sticky Run bar. Collapses to a drawer < 1100 px.
import { useEffect, useMemo, useState } from "react";
import type { PresetId } from "../api/types";
import { Progress } from "../components/Panel";
import { IconPlay, IconStop, IconX } from "../components/icons";
import { useT } from "../i18n";
import { GROUPS, groupVisible, type Ctx } from "../params/schema";
import { PRESET_IDS } from "../state/presets";
import { cancelActive, runContext, runCurrent } from "../state/runner";
import { presetDefaults, useStore } from "../state/store";
import { fmtDuration } from "../utils/format";
import { deepEqual } from "../utils/object";
import { ParamGroup } from "./ParamGroup";

function PresetCard() {
  const t = useT();
  const preset = useStore((s) => s.preset);
  const meta = useStore((s) => s.meta);
  const load = useStore((s) => s.loadPreset);
  const modified = useStore((s) => {
    const d = presetDefaults(s);
    return !(deepEqual(s.params.device, d.device) && deepEqual(s.params.sweep, d.sweep) && deepEqual(s.params.stochastic, d.stochastic));
  });
  const baseName = t(`preset.${preset}` as never);
  return (
    <div className="preset-card" data-testid="preset-card">
      <div className="preset-head">{t("preset.label")}</div>
      <div className="preset-options" role="radiogroup" aria-label={t("preset.label")}>
        {PRESET_IDS.map((id: PresetId) => (
          <button key={id} type="button" role="radio" aria-checked={preset === id && !modified} className="preset-opt" data-testid={`preset-${id}`} onClick={() => load(id)} title={meta.presets[id]?.label?.[t.lang] ?? id}>
            {t(`preset.${id}` as never)}
          </button>
        ))}
      </div>
      <div className="preset-note" data-testid="preset-label" aria-live="polite">
        {modified ? (
          <>
            <span className="chg" aria-hidden />
            <span style={{ flex: 1 }}>{t("preset.modified", { base: baseName })}</span>
            <button type="button" className="link-btn" onClick={() => load(preset)}>
              {t("reset")}
            </button>
          </>
        ) : (
          <span className="muted">{meta.presets[preset]?.label?.[t.lang] ?? preset}</span>
        )}
      </div>
    </div>
  );
}

function useTick(active: boolean, ms = 200) {
  const [, set] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => set((x) => x + 1), ms);
    return () => clearInterval(id);
  }, [active, ms]);
}

export function RunBar() {
  const t = useT();
  const tab = useStore((s) => s.tab);
  const mode = useStore((s) => s.mode);
  const autoRun = useStore((s) => s.autoRun);
  const setAutoRun = useStore((s) => s.setAutoRun);
  const lastRun = useStore((s) => s.activeRun);
  const results = useStore((s) => s.results);
  const anyRunning = (lastRun?.keys ?? []).some((k) => results[k]?.status === "running" || results[k]?.status === "queued");
  // report a finished run only in its own context (tab + mode); a running one is always shown (cancellable)
  const active = lastRun && (anyRunning || lastRun.label === runContext(tab, mode)) ? lastRun : null;
  const entries = useMemo(() => (active?.keys ?? []).map((k) => results[k]).filter(Boolean), [active, results]);
  const running = entries.some((e) => e.status === "running" || e.status === "queued");
  useTick(running);
  const progress = entries.length ? entries.reduce((a, e) => a + (e.status === "done" ? 1 : e.status === "running" || e.status === "queued" ? e.progress : 0), 0) / entries.length : 0;
  const runningEntry = entries.find((e) => e.status === "running" || e.status === "queued");
  const failed = entries.filter((e) => e.status === "error").length;
  const cancelled = entries.some((e) => e.status === "cancelled");
  const elapsed = active ? ((active.finishedAt ?? performance.now()) - active.startedAt) / 1000 : 0;
  const label = tab === "circuit" ? t("run.circuit") : mode === "stochastic" ? t("run.sto") : t("run.det");

  let status: string;
  if (running) status = `${runningEntry?.kind ?? ""}${runningEntry?.message ? ": " + runningEntry.message : ""}`;
  else if (!active) status = t("run.idle");
  else if (failed) status = `${t("run.failed")} (${failed}/${entries.length})`;
  else if (cancelled) status = t("run.cancelled");
  else status = t("run.done", { t: fmtDuration(elapsed) });

  return (
    <div className="runbar" data-testid="runbar">
      <div className="runbar-main">
        <button type="button" className="btn primary run-btn" onClick={() => void runCurrent()} data-testid="run-button" aria-keyshortcuts="Control+Enter" title="Ctrl/⌘ + Enter">
          <IconPlay size={14} /> {label}
        </button>
        {running && (
          <button type="button" className="btn danger" onClick={cancelActive} data-testid="cancel-button" aria-label={t("run.cancel")}>
            <IconStop size={12} /> {t("run.cancel")}
          </button>
        )}
      </div>
      {(running || active) && <Progress value={running ? progress : failed ? 0 : 1} indeterminate={running && progress < 0.01} />}
      <div className="runbar-status" aria-live="polite">
        <span className="msg" data-testid="run-status" title={status}>
          {status}
        </span>
        {running && <span className="mono">{t("run.elapsed", { t: fmtDuration(elapsed) })}</span>}
      </div>
      {tab === "device" && mode === "deterministic" && (
        <div className="autorun">
          <button type="button" role="switch" aria-checked={autoRun} className="switch" onClick={() => setAutoRun(!autoRun)} aria-label={t("run.auto")} data-testid="autorun" />
          <span title={t("run.auto.hint")}>{t("run.auto")}</span>
          <span className="small muted" style={{ marginLeft: "auto" }}>Ctrl/⌘ + Enter</span>
        </div>
      )}
    </div>
  );
}

export function Sidebar() {
  const t = useT();
  const tab = useStore((s) => s.tab);
  const mode = useStore((s) => s.mode);
  const params = useStore((s) => s.params);
  const open = useStore((s) => s.sidebarOpen);
  const setOpen = useStore((s) => s.setSidebar);
  const ctx: Ctx = { root: params, mode, tab };
  // groups specific to the current tab (e.g. bench/solver on the circuit tab) come first
  const circuitFirst = (g: (typeof GROUPS)[number]) => Number(tab === "circuit" && g.tabs.length === 1 && g.tabs[0] === "circuit");
  const groups = GROUPS.filter((g) => groupVisible(g, ctx)).sort((a, b) => circuitFirst(b) - circuitFirst(a));
  return (
    <>
      {open && <div className="scrim" onClick={() => setOpen(false)} aria-hidden />}
      <aside className={`sidebar${open ? " open" : ""}`} id="sidebar" aria-label={t("sidebar.title")} data-testid="sidebar">
        <div className="drawer-head">
          <span>{t("sidebar.title")}</span>
          <button type="button" className="icon-btn" onClick={() => setOpen(false)} aria-label={t("sidebar.close")}>
            <IconX size={15} />
          </button>
        </div>
        <div className="sidebar-scroll">
          <PresetCard />
          {groups.map((g) => (
            <ParamGroup key={g.id} g={g} ctx={ctx} />
          ))}
        </div>
        <RunBar />
      </aside>
    </>
  );
}
