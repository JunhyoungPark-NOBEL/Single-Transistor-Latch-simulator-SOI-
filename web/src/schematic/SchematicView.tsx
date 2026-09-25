// Schematic editor view (lazy chunk): toolbar, canvas + properties panel, ERC bar, results (waveforms,
// time cursor, statistics) and the read-only netlist (folded in 간단히). The sidebar shows the device library
// and the simulation settings (./SchematicSidebar.tsx); Run / Ctrl+Enter run this circuit.
import { useEffect, useMemo, useRef, useState } from "react";
import { Panel } from "../components/Panel";
import { IconCopy } from "../components/icons";
import { isStale, useEntry } from "../device/common";
import { Modal } from "../devices/Modal";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { useIsAll } from "../state/layout";
import { useStore } from "../state/store";
import { fmtDuration, fmtInt } from "../utils/format";
import { Canvas } from "./Canvas";
import { runErc, type ErcItem } from "./erc";
import { Inspector } from "./Inspector";
import { buildRequest, netlistText } from "./netlist";
import { extractNets, type Connectivity } from "./nets";
import { annotationsAt, Results } from "./Results";
import { RESULT_KEY, requestKey, runSchematic, type SchematicRunData } from "./run";
import { fmtSI } from "./si";
import { useSch } from "./store";
import { handleEditorKey, Toolbar } from "./Toolbar";
import { Segmented, Switch } from "./ui";
import "./schematic.css";

function ErcBar({ erc, hasResult, stale }: { erc: ErcItem[]; hasResult: boolean; stale: boolean }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const annotate = useSch((s) => s.annotate);
  const cursorT = useSch((s) => s.cursorT);
  const set = useSch((s) => s.set);
  const errs = erc.filter((i) => i.level === "error");
  const warns = erc.filter((i) => i.level === "warning");
  useEffect(() => {
    const onOpen = () => setOpen(true);
    window.addEventListener("sch-open-erc", onOpen);
    return () => window.removeEventListener("sch-open-erc", onOpen);
  }, []);
  const status = errs.length ? "err" : warns.length ? "warn" : "ok";
  return (
    <div className="sch-status">
      <div className="sch-status-row">
        <button type="button" className={`erc-pill ${status}`} onClick={() => setOpen(!open)} aria-expanded={open} data-testid="erc-pill" title={t("schematic.erc.full")}>
          <span className="erc-dot" aria-hidden />
          <strong>{t("schematic.erc.title")}</strong>
          <span>{erc.length ? t("schematic.erc.count", { e: errs.length, w: warns.length }) : t("schematic.erc.ok")}</span>
        </button>
        <span className="spacer" />
        {hasResult && (
          <label className="tb-toggle" title={stale ? t("schematic.res.staleHint") : undefined}>
            <Switch on={annotate && !stale} onChange={(v) => set({ annotate: v })} label={t("schematic.res.annotate")} testId="toggle-annotate" />
            <span className={stale ? "muted" : undefined}>
              {t("schematic.res.annotate")}
              {/* the instant the values are read (the time cursor, far below in the waveform card) */}
              {annotate && !stale && cursorT != null && <span className="mono" data-testid="annotate-time"> · t = {fmtSI(cursorT, "s", 4)}</span>}
            </span>
          </label>
        )}
      </div>
      {open && erc.length > 0 && (
        <ul className="erc-list" data-testid="erc-list">
          {erc.map((i, k) => (
            <li key={k}>
              <button
                type="button"
                className={`erc-item ${i.level}`}
                onClick={() => {
                  const ids = [...i.elementIds, ...(i.wireIds ?? [])];
                  useSch.getState().flash(ids);
                }}
                data-testid={`erc-item-${i.code}`}
              >
                <span className="erc-lvl" aria-hidden>
                  {i.level === "error" ? "!" : "i"}
                </span>
                <span>{t(`schematic.erc.${i.code}` as StrKey, i.vars)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function NetlistPanel({ conn }: { conn: Connectivity }) {
  const t = useT();
  const doc = useSch((s) => s.doc);
  const traces = useSch((s) => s.traces);
  const mode = useStore((s) => s.mode);
  const [view, setView] = useState<"spice" | "json">("spice");
  const [copied, setCopied] = useState(false);
  const text = useMemo(() => (view === "spice" ? netlistText(doc, conn, mode) : JSON.stringify(buildRequest(doc, conn, mode, traces), null, 1)), [view, doc, conn, mode, traces]);
  return (
    <Panel
      id="sch-netlist"
      wide
      title={t("schematic.netlist.title")}
      desc={t("schematic.netlist.desc")}
      hasData
      toolbar={
        <>
          <Segmented value={view} label={t("schematic.netlist.title")} onChange={setView} options={[{ v: "spice", label: t("schematic.netlist.spice") }, { v: "json", label: t("schematic.netlist.json") }]} testId="netlist-view" />
          <button
            type="button"
            className="btn sm ghost"
            onClick={() => {
              void navigator.clipboard?.writeText(text).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1400);
              });
            }}
          >
            <IconCopy size={13} /> {copied ? t("copied") : t("copy")}
          </button>
        </>
      }
    >
      <div className="panel-foot">
        <pre className="netlist" data-testid="netlist-text">
          {text}
        </pre>
      </div>
    </Panel>
  );
}

/** 간단히: the netlist behind "넷리스트 (SPICE · JSON) ▸" (the text stays in the DOM); 모두 보기: the open panel. */
function NetlistBlock({ conn }: { conn: Connectivity }) {
  const t = useT();
  const all = useIsAll();
  if (all) return <NetlistPanel conn={conn} />;
  return (
    <details className="sch-fold sch-netlist-fold">
      <summary data-testid="netlist-toggle">{t("schematic.netlist.toggle")}</summary>
      <NetlistPanel conn={conn} />
    </details>
  );
}

function ConfirmRun() {
  const t = useT();
  const confirm = useSch((s) => s.confirm);
  if (!confirm) return null;
  const close = () => useSch.getState().set({ confirm: null });
  return (
    <Modal
      title={t("schematic.run.confirmTitle")}
      onClose={close}
      width={460}
      testId="sch-confirm"
      footer={
        <>
          <span className="spacer" />
          <button type="button" className="btn sm ghost" onClick={close}>
            {t("schematic.dev.cancel")}
          </button>
          <button type="button" className="btn sm primary" onClick={() => void runSchematic({ confirmed: true })} data-testid="sch-confirm-run">
            {t("schematic.run.anyway")}
          </button>
        </>
      }
    >
      <p>{t("schematic.run.confirmBody", { steps: fmtInt(confirm.totalSteps), time: fmtDuration(confirm.seconds) })}</p>
      {confirm.level === "refuse" && <div className="callout warn">{t("schematic.run.confirmRefuse")}</div>}
      <p className="small muted">{t("schematic.sim.hintHeavy")}</p>
    </Modal>
  );
}

function Toast() {
  const toast = useSch((s) => s.toast);
  const [shown, setShown] = useState<typeof toast>(null);
  useEffect(() => {
    if (!toast) return;
    setShown(toast);
    const id = setTimeout(() => setShown(null), 3200);
    return () => clearTimeout(id);
  }, [toast]);
  if (!shown) return null;
  return (
    <div className={`sch-toast ${shown.kind ?? "ok"}`} role="status" data-testid="sch-toast">
      {shown.msg}
    </div>
  );
}

export default function SchematicView() {
  const doc = useSch((s) => s.doc);
  const traces = useSch((s) => s.traces);
  const cursorT = useSch((s) => s.cursorT);
  const annotate = useSch((s) => s.annotate);
  const source = useSch((s) => s.annotateSource);
  const mode = useStore((s) => s.mode);
  const method = useStore((s) => s.params.circuit.solver.method);
  const setParam = useStore((s) => s.setParam);
  const conn = useMemo(() => extractNets(doc), [doc]);
  const erc = useMemo(() => runErc(doc, conn), [doc, conn]);
  const { entry } = useEntry<SchematicRunData>(RESULT_KEY);
  const data = entry?.data as SchematicRunData | undefined;
  const reqKey = useMemo(() => requestKey(buildRequest(doc, conn, mode, traces)), [doc, conn, mode, traces]);
  const stale = isStale(entry, reqKey);
  const res = data?.result;
  const ann = useMemo(() => (res && annotate && !stale && cursorT != null ? annotationsAt(res, cursorT, res.mode === "stochastic" ? source : "run0") : null), [res, annotate, stale, cursorT, source]);
  const cells = useMemo(() => doc.elements.filter((e) => e.kind === "STL").map((e) => e.name), [doc.elements]);
  // keep the header's integrator hint in step with this circuit's .tran method
  useEffect(() => {
    if (method !== doc.tran.method) setParam(["circuit", "solver", "method"], doc.tran.method);
  }, [doc.tran.method, method, setParam]);
  const all = useIsAll();
  // 간단히: when a run finishes and its waveform (e.g. the oscillator's sawtooth) is below the fold, bring the
  // result summary to the top of the view once — the answer should not hide under the editor
  const wasRunning = useRef(false);
  const status = entry?.status;
  useEffect(() => {
    const running = status === "running" || status === "queued";
    if (running) {
      wasRunning.current = true;
      return;
    }
    if (!wasRunning.current || status !== "done" || all) return;
    wasRunning.current = false;
    const id = requestAnimationFrame(() => {
      const waves = document.querySelector('[data-testid="panel-sch-waves"]');
      if (!waves || waves.getBoundingClientRect().top < window.innerHeight - 160) return;
      const target = document.querySelector('[data-testid="sch-summary"]') ?? waves;
      target.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [status, all]);
  const [tall, setTall] = useState(() => (typeof window !== "undefined" ? window.innerHeight > 860 : true));
  useEffect(() => {
    const on = () => setTall(window.innerHeight > 860);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  const canvasH = tall ? (all ? 540 : 470) : 440;
  return (
    <div className="sch-view" onKeyDown={handleEditorKey} data-testid="schematic-view">
      <section className="panel wide sch-editor" aria-label="schematic editor">
        <Toolbar />
        {/* 간단히: a slightly shorter canvas keeps the result summary above the fold at 1440 × 900 */}
        <div className="sch-body" style={{ "--sch-h": `${canvasH}px` } as React.CSSProperties}>
          <Canvas conn={conn} erc={erc} ann={ann} height={canvasH} />
          <aside className="sch-side">
            <Inspector conn={conn} />
          </aside>
        </div>
        <ErcBar erc={erc} hasResult={!!res} stale={stale} />
      </section>
      <Results entry={entry} stale={stale} cells={cells} />
      <NetlistBlock conn={conn} />
      <ConfirmRun />
      <Toast />
    </div>
  );
}
