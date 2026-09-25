// Left sidebar: Device card (technology, geometry, calibration preset, saved devices), the basic parameter
// groups, the "고급 설정" disclosure with the advanced groups, and the sticky Run bar. On the Circuit tab's
// schematic editor it shows the device library and the simulation settings instead (lazy chunk).
// Collapses to a drawer < 1100 px.
import { lazy, Suspense, useEffect, useId, useMemo, useState } from "react";
import { useCircuitView } from "../circuit/view";
import { DeviceCard } from "../devices/DeviceCard";
import { useForcing } from "../device/forcing";
import { Progress } from "../components/Panel";
import { IconChevron, IconPlay, IconStop, IconX } from "../components/icons";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { fill } from "../i18n/strings.ux";
import { GROUPS, groupPaths, groupVisible, type Ctx, type GroupDef } from "../params/schema";
import { useIsAll } from "../state/layout";
import { cancelActive, runContext, runCurrent } from "../state/runner";
import { presetDefaults, useStore } from "../state/store";
import { fmtDuration } from "../utils/format";
import { deepEqual, getPath } from "../utils/object";
import { ParamGroup } from "./ParamGroup";
import { GeometryControls } from "./GeometryControls";
import { useSidebarUi } from "./sidebarState";
import "./sidebar.css";

const SchematicSidebar = lazy(() => import("../schematic/SchematicSidebar"));

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
  useForcing((s) => s.forcing); // keep progress context in sync with the forcing switch
  const autoRun = useStore((s) => s.autoRun);
  const setAutoRun = useStore((s) => s.setAutoRun);
  const lastRun = useStore((s) => s.activeRun);
  const results = useStore((s) => s.results);
  const view = useCircuitView((s) => s.view);
  const anyRunning = (lastRun?.keys ?? []).some((k) => results[k]?.status === "running" || results[k]?.status === "queued");
  // report a finished run only in its own context (tab + mode, schematic vs benches); a running one is always shown (cancellable)
  const ctx = tab === "circuit" && view === "schematic" ? `schematic ${mode}` : runContext(tab, mode);
  const active = lastRun && (anyRunning || lastRun.label === ctx) ? lastRun : null;
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

  const showProgress = running || failed > 0;
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
        {tab === "device" && mode === "deterministic" && (
          <label className="autorun" title={t("run.auto.hint")}>
            <button type="button" role="switch" aria-checked={autoRun} className="switch" onClick={() => setAutoRun(!autoRun)} aria-label={t("run.auto")} data-testid="autorun" />
            {!running && (
              <span className="autorun-label" aria-hidden>
                {t("run.auto")}
              </span>
            )}
          </label>
        )}
      </div>
      {showProgress && (
        <div className={`rb-progress${!running && failed ? " failed" : ""}`}>
          <Progress value={running ? progress : entries.length ? (entries.length - failed) / entries.length : 0} indeterminate={running && progress < 0.01} />
        </div>
      )}
      <div className="runbar-status" aria-live="polite">
        <span className={`msg${failed && !running ? " err" : ""}`} data-testid="run-status" title={status}>
          {status}
        </span>
        {running && <span className="mono">{t("run.elapsed", { t: fmtDuration(elapsed) })}</span>}
      </div>
    </div>
  );
}

const ADV_SHORT: Record<string, keyof typeof GUIDE> = { state: "adv.short.state", calib: "adv.short.calib", ext: "adv.short.ext", numerics: "adv.short.numerics", solver: "adv.short.solver" };

/** "고급 설정 ▸ 국소 상태 · 보정 · 확장 · 수치 · 2개 수정" + the advanced groups (closed heads) when open. */
function AdvancedGroups({ groups, ctx }: { groups: GroupDef[]; ctx: Ctx }) {
  const t = useT();
  const all = useIsAll();
  const open = useSidebarUi((s) => (all ? s.advOpenAll : s.advOpen));
  const setOpen = useSidebarUi((s) => s.setAdvOpen);
  const bench = useStore((s) => s.params.circuit.bench);
  const changed = useStore((s) => {
    const d = presetDefaults(s);
    let n = 0;
    for (const g of groups) for (const p of g.id === "bench" ? [["circuit", "bench_params", bench]] : groupPaths(g)) if (!deepEqual(getPath(s.params, p), getPath(d, p))) n++;
    return n;
  });
  const bodyId = useId();
  if (!groups.length) return null;
  const names = groups.map((g) => (ADV_SHORT[g.id] ? t.l(GUIDE[ADV_SHORT[g.id]]) : t(g.title))).join("\u2009·\u2009");
  return (
    <section className={`adv-groups${open ? " open" : ""}`} data-testid="adv-groups" aria-label={t.l(GUIDE["adv.title"])}>
      <button type="button" className="adv-head" aria-expanded={open} aria-controls={bodyId} onClick={() => setOpen(!open, all)}>
        <IconChevron size={14} className="chev" />
        <span className="adv-title">{t.l(GUIDE["adv.title"])}</span>
        {changed > 0 && <span className="chg-count">{fill(t.l(GUIDE["adv.changed"]), { n: changed })}</span>}
        <span className="adv-names">{names}</span>
      </button>
      {open && (
        <div className="adv-body" id={bodyId}>
          {groups.map((g) => (
            <ParamGroup key={g.id} g={g} ctx={ctx} />
          ))}
        </div>
      )}
    </section>
  );
}

export function Sidebar() {
  const t = useT();
  const tab = useStore((s) => s.tab);
  const mode = useStore((s) => s.mode);
  const forcing = useForcing((s) => s.forcing);
  const params = useStore((s) => s.params);
  const open = useStore((s) => s.sidebarOpen);
  const setOpen = useStore((s) => s.setSidebar);
  const view = useCircuitView((s) => s.view);
  const schematic = tab === "circuit" && view === "schematic";
  const ctx: Ctx = { root: params, mode, tab };
  // groups specific to the current tab (e.g. bench/solver on the circuit tab) come first
  const circuitFirst = (g: (typeof GROUPS)[number]) => Number(tab === "circuit" && g.tabs.length === 1 && g.tabs[0] === "circuit");
  const csvm = tab === "device" && forcing === "csvm";
  const groups = GROUPS.filter((g) => groupVisible(g, ctx)).map((g) => {
    if (!csvm) return g;
    // CSVM is one time-domain trace; voltage-sweep/MC sweep controls have no effect here.
    const hidden = new Set(["n_cycles", "n_traces", "engine", "fold_nodes", "hazard_nodes", "ls_trend"]);
    return { ...g, ...(g.id === "bias" ? { title: "g.bias.csvm" as const } : {}), fields: g.fields.filter((f) => f.path[0] !== "sweep" && !hidden.has(f.key)) };
  }).sort((a, b) => circuitFirst(b) - circuitFirst(a));
  const basic = groups.filter((g) => !g.advanced);
  const advanced = groups.filter((g) => g.advanced);
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
          {schematic ? (
            <Suspense fallback={<div className="skeleton-plot" style={{ height: 220 }} />}>
              <SchematicSidebar />
            </Suspense>
          ) : (
            <>
              <GeometryControls />
              <DeviceCard />
              {basic.map((g) => (
                <ParamGroup key={g.id} g={g} ctx={ctx} />
              ))}
              <AdvancedGroups groups={advanced} ctx={ctx} />
            </>
          )}
        </div>
        <RunBar />
      </aside>
    </>
  );
}
