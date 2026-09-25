// Sidebar content while the schematic editor is active: device library (place devices), .tran settings
// with a live feasibility estimate, and the stochastic settings (runs, seed, noise, local states).
// 간단히: the stop time and the number of runs stay visible; the other settings fold behind "고급 항목 n개 ▸"
// (open in 모두 보기).
import { useMemo, useState, type ReactNode } from "react";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { deviceName, type LibDevice } from "../devices/library";
import { SaveDeviceForm } from "../devices/DeviceCard";
import { useDeviceLib } from "../devices/store";
import { LOCAL_ACTION_OPTIONS, LOCAL_MODE_OPTIONS } from "../params/schema";
import { useIsAll } from "../state/layout";
import { useStore } from "../state/store";
import { fmtDuration, fmtSI as fmtSIu } from "../utils/format";
import { iphPA } from "../utils/payload";
import { estimate } from "./feasibility";
import type { SchematicDoc } from "./model";
import { autoDtMax, autoDtMin } from "./netlist";
import { fmtSI } from "./si";
import { libraryEntries, useSch } from "./store";
import { PartIcon } from "./Symbols";
import { Card, Row, Segmented, SIInput, Switch } from "./ui";
import "./schematic.css";

function patchDoc(fn: (d: SchematicDoc) => SchematicDoc) {
  useSch.getState().commit(fn);
}

/** "고급 항목 n개 ▸ …" disclosure (closed in 간단히, open in 모두 보기); the fields stay in the DOM. */
function Advanced({ n, list, testId, children }: { n: number; list: string; testId: string; children: ReactNode }) {
  const t = useT();
  const all = useIsAll();
  return (
    <details className="sch-adv" open={all}>
      <summary data-testid={testId}>
        <span className="sch-adv-n">{t("schematic.sim.adv", { n })}</span>
        <span className="sch-adv-list">{list}</span>
      </summary>
      <div className="sch-adv-body">{children}</div>
    </details>
  );
}

function LibraryCard() {
  const t = useT();
  const lang = useStore((s) => s.lang);
  useStore((s) => s.meta);
  useStore((s) => s.params.device);
  useDeviceLib((s) => s.devices);
  const tool = useSch((s) => s.tool);
  const stlChoice = useSch((s) => s.stlChoice);
  const [saving, setSaving] = useState(false);
  const entries = libraryEntries();
  const placing = tool.kind === "place" && tool.el === "STL";
  const place = (d: LibDevice) => {
    useSch.getState().set({ stlChoice: d.id, tool: { kind: "place", el: "STL", rot: 0, mirror: false }, selection: [] });
    document.querySelector<HTMLElement>("[data-testid=sch-canvas]")?.focus({ preventScroll: true });
  };
  return (
    <Card title={t("schematic.lib.title")} desc={t("schematic.lib.desc")} testId="sch-library">
      <ul className="lib-list" data-testid="lib-list">
        {entries.map((d) => {
          const iph = iphPA(d.device);
          const active = placing && stlChoice === d.id;
          return (
            <li key={d.id} className={`lib-item${active ? " active" : ""}`} data-testid={`lib-${d.builtin ? d.id : d.id === "current" ? "current" : d.name}`}>
              <span className="lib-icon" aria-hidden>
                <PartIcon kind="STL" size={20} />
              </span>
              <span className="lib-text">
                <span className="lib-name">{d.id === "current" ? t("schematic.lib.current") : deviceName(d, lang)}</span>
                <span className="lib-meta mono">
                  {d.builtin && <span className="lib-badge">{t("schematic.lib.builtin")}</span>}
                  V<sub>G</sub> {d.device.vg.toFixed(2)} V · {iph ? <>I<sub>PH</sub> {fmtSIu(iph * 1e-12, "A", 3)}</> : t("schematic.lib.dark")} · {d.stochastic.local_state.mode}
                </span>
              </span>
              <button type="button" className={`btn sm${active ? " primary" : ""}`} onClick={() => place(d)} aria-label={t("schematic.lib.placeAria", { name: d.id === "current" ? t("schematic.lib.current") : deviceName(d, lang) })} data-testid="lib-place">
                {t("schematic.lib.place")}
              </button>
            </li>
          );
        })}
      </ul>
      {saving ? (
        <SaveDeviceForm onDone={(d) => {
          setSaving(false);
          if (d) useSch.getState().notify(t("schematic.dev.savedToast", { name: d.name }));
        }} />
      ) : (
        <button type="button" className="btn sm ghost lib-save" onClick={() => setSaving(true)} data-testid="lib-save-current" title={t("schematic.lib.currentHint")}>
          + {t("schematic.dev.saveAs")}
        </button>
      )}
    </Card>
  );
}

function SimCard() {
  const t = useT();
  const tran = useSch((s) => s.doc.tran);
  const detect = useSch((s) => s.doc.detect);
  const saveAll = useSch((s) => s.doc.save_all);
  const doc = useSch((s) => s.doc);
  const mode = useStore((s) => s.mode);
  const est = useMemo(() => estimate(doc, mode), [doc, mode]);
  const setTran = (p: Partial<SchematicDoc["tran"]>) => patchDoc((d) => ({ ...d, tran: { ...d.tran, ...p } }));
  const level = est.level;
  return (
    <Card title={t("schematic.sim.title")} desc={t("schematic.sim.desc")} testId="sch-sim">
      <Row label={t("schematic.sim.tStop")}>
        <SIInput value={tran.t_stop_s} unit="s" min={1e-12} onCommit={(v) => v && setTran({ t_stop_s: v })} testId="sim-tstop" ariaLabel={t("schematic.sim.tStop")} />
      </Row>
      <Advanced n={8} list={t("schematic.sim.advList")} testId="sch-sim-adv">
      <Row label={t("schematic.sim.tStart")}>
        <SIInput value={tran.t_start_save_s} unit="s" min={0} onCommit={(v) => setTran({ t_start_save_s: v ?? 0 })} testId="sim-tstart" ariaLabel={t("schematic.sim.tStart")} />
      </Row>
      <Row label={t("schematic.sim.dtMax")}>
        <SIInput value={tran.dt_max_s} unit="s" min={1e-15} allowEmpty placeholder={t("schematic.sim.autoVal", { v: fmtSI(autoDtMax(tran.t_stop_s), "s", 3) })} onCommit={(v) => setTran({ dt_max_s: v })} testId="sim-dtmax" ariaLabel={t("schematic.sim.dtMax")} />
      </Row>
      <Row label={t("schematic.sim.dtMin")}>
        <SIInput value={tran.dt_min_s} unit="s" min={1e-18} allowEmpty placeholder={t("schematic.sim.autoVal", { v: fmtSI(autoDtMin(tran.t_stop_s), "s", 2) })} onCommit={(v) => setTran({ dt_min_s: v })} testId="sim-dtmin" ariaLabel={t("schematic.sim.dtMin")} />
      </Row>
      <Row label={t("schematic.sim.method")}>
        <Segmented value={tran.method} label={t("schematic.sim.method")} onChange={(m) => setTran({ method: m })} options={[{ v: "BE", label: "BE" }, { v: "TRAP", label: "TRAP" }]} testId="sim-method" />
      </Row>
      <Row label={t("schematic.sim.reltol")}>
        <SIInput value={tran.reltol} min={1e-9} max={0.1} onCommit={(v) => v && setTran({ reltol: v })} testId="sim-reltol" ariaLabel={t("schematic.sim.reltol")} />
      </Row>
      <Row label={t("schematic.sim.ith")}>
        <SIInput value={detect.i_threshold_A} unit="A" min={1e-15} max={1e-3} onCommit={(v) => v && patchDoc((d) => ({ ...d, detect: { ...d.detect, i_threshold_A: v } }))} ariaLabel={t("schematic.sim.ith")} />
      </Row>
      <Row label={t("schematic.sim.hyst")} hint={t("schematic.sim.hystHint")}>
        <SIInput value={detect.hysteresis} min={1} max={1e6} onCommit={(v) => v && patchDoc((d) => ({ ...d, detect: { ...d.detect, hysteresis: v } }))} ariaLabel={t("schematic.sim.hyst")} />
      </Row>
      <div className="toggle-row">
        <span className="small" title={t("schematic.sim.saveAllHint")}>{t("schematic.sim.saveAll")}</span>
        <Switch on={saveAll} onChange={(v) => patchDoc((d) => ({ ...d, save_all: v }))} label={t("schematic.sim.saveAll")} testId="sim-saveall" />
      </div>
      {!saveAll && <div className="small muted">{t("schematic.sim.saveAllHint")}</div>}
      </Advanced>
      <div className={`feas ${level}`} data-testid="feasibility" data-level={level}>
        <div className="feas-head">
          <span className="feas-dot" aria-hidden />
          <strong>{t(`schematic.sim.level.${level}` as StrKey)}</strong>
          <span className="mono">
            {mode === "stochastic"
              ? t("schematic.sim.estimateSto", { steps: compact(est.stepsPerRun), runs: est.runs, time: fmtDuration(est.seconds) })
              : t("schematic.sim.estimate", { steps: compact(est.stepsPerRun), time: fmtDuration(est.seconds) })}
          </span>
        </div>
        {(level === "heavy" || level === "refuse") && <div className="feas-hint">{t("schematic.sim.hintHeavy")}</div>}
        {mode === "stochastic" && level !== "ok" && <div className="feas-hint">{t("schematic.sim.hintSto")}</div>}
        <div className="feas-note">{t("schematic.sim.roughNote")}</div>
      </div>
      <div className="small muted">{t("schematic.sim.modeNote")}</div>
    </Card>
  );
}

function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(n));
}

function StochCard() {
  const t = useT();
  const st = useSch((s) => s.doc.stoch);
  const setSt = (p: Partial<SchematicDoc["stoch"]>) => patchDoc((d) => ({ ...d, stoch: { ...d.stoch, ...p } }));
  const ls = st.local_state;
  const volt = ls.action === "gidl" || ls.action === "junction";
  return (
    <Card title={t("schematic.sto.title")} desc={t("schematic.sto.desc")} testId="sch-sto" className="sto-only">
      <Row label={t("schematic.sto.runs")}>
        <SIInput value={st.n_runs} min={1} max={200} integer onCommit={(v) => v && setSt({ n_runs: v })} testId="sto-runs" ariaLabel={t("schematic.sto.runs")} />
      </Row>
      <Advanced n={st.local_source === "device" ? 4 : ls.mode === "none" ? 5 : ls.mode === "evolving" ? 8 : 7} list={t("schematic.sto.advList")} testId="sch-sto-adv">
      <Row label={t("schematic.sto.seed")}>
        <SIInput value={st.seed} min={0} max={2 ** 32 - 1} integer onCommit={(v) => v != null && setSt({ seed: v })} ariaLabel={t("schematic.sto.seed")} />
      </Row>
      <div className="toggle-row">
        <span className="small">{t("schematic.sto.noise")}</span>
        <Switch on={st.carrier_noise} onChange={(v) => setSt({ carrier_noise: v })} label={t("schematic.sto.noise")} />
      </div>
      <div className="toggle-row">
        <span className="small">{t("schematic.sto.ldNoise")}</span>
        <Switch on={st.ld_carrier_noise} onChange={(v) => setSt({ ld_carrier_noise: v })} label={t("schematic.sto.ldNoise")} />
      </div>
      <div className="toggle-row">
        <span className="small">{t("schematic.sto.override")}</span>
        <Switch on={st.local_source === "override"} onChange={(v) => setSt({ local_source: v ? "override" : "device" })} label={t("schematic.sto.override")} testId="sto-override" />
      </div>
      {st.local_source === "device" ? (
        <div className="small muted">{t("schematic.sto.localDeviceHint")}</div>
      ) : (
        <>
          <Segmented
            value={ls.mode}
            full
            label={t("schematic.sto.local")}
            onChange={(v) => setSt({ local_state: { ...ls, mode: v } })}
            options={LOCAL_MODE_OPTIONS.map((o) => ({ v: o.value as typeof ls.mode, label: typeof o.label === "string" ? t(o.label as StrKey) : t.l(o.label) }))}
          />
          {ls.mode !== "none" && (
            <>
              <Row label={t("schematic.sto.action")}>
                <select className="select" value={ls.action} onChange={(e) => setSt({ local_state: { ...ls, action: e.target.value as typeof ls.action } })}>
                  {LOCAL_ACTION_OPTIONS.map((o) => (
                    <option key={String(o.value)} value={String(o.value)}>
                      {typeof o.label === "string" ? t(o.label as StrKey) : t.l(o.label)}
                    </option>
                  ))}
                </select>
              </Row>
              <Row label={t("schematic.sto.sigma")}>
                <SIInput value={ls.sigma} unit={volt ? "V" : "ln"} min={0} onCommit={(v) => v != null && setSt({ local_state: { ...ls, sigma: v } })} ariaLabel={t("schematic.sto.sigma")} />
              </Row>
              {ls.mode === "evolving" && (
                <Row label={t("schematic.sto.tau")}>
                  <SIInput value={ls.tau_s} unit="s" min={1e-9} onCommit={(v) => v && setSt({ local_state: { ...ls, tau_s: v } })} ariaLabel={t("schematic.sto.tau")} />
                </Row>
              )}
            </>
          )}
        </>
      )}
      </Advanced>
    </Card>
  );
}

export default function SchematicSidebar() {
  const mode = useStore((s) => s.mode);
  return (
    <>
      <LibraryCard />
      <SimCard />
      {mode === "stochastic" && <StochCard />}
    </>
  );
}
