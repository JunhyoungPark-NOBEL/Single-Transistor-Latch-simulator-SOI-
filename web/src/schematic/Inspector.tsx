import { modelLabel } from "../params/model";
// Properties panel for the selection: name, value (SPICE numbers), source waveforms, STL device from the
// library (snapshot + "update from library") and its light waveform, net-label text, orientation.
import { useEffect, useMemo, useRef } from "react";
import type { Wave } from "../api/circuitCustom";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { deviceName, isSupportedTechnology, type LibDevice } from "../devices/library";
import { useDeviceLib } from "../devices/store";
import { useStore } from "../state/store";
import { deepEqual } from "../utils/object";
import { iphPA } from "../utils/payload";
import { deleteItems, duplicateItems, mirrorItems, rotateItems } from "./edit";
import { LABEL_RE, NAME_RE } from "./erc";
import { CIRCUIT_KINDS, DEFAULT_CMP, DEFAULT_MOS, DEFAULT_DIODE, DEFAULT_BJT, pinPositions, type CmpParams, type SElement } from "./model";
import type { Connectivity } from "./nets";
import { isCircuitNet, optionalPinConnected, pinId } from "./nets";
import { fmtSI } from "./si";
import type { SchematicRunData } from "./run";
import { libraryEntries, stlRefFor, useSch } from "./store";
import { PartIcon } from "./Symbols";
import { Row, Segmented, SIInput, TextInput } from "./ui";
import { WaveEditor } from "./WaveEditor";
import { SubText } from "../plots/SubText";

const KIND_KEY: Record<SElement["kind"], StrKey> = {
  MOS: "schematic.tool.MOS", D: "schematic.tool.D", BJT: "schematic.tool.BJT",
  R: "schematic.tool.R", C: "schematic.tool.C", V: "schematic.tool.V", I: "schematic.tool.I", STL: "schematic.tool.STL", CMP: "schematic.tool.CMP", GND: "schematic.tool.GND", LABEL: "schematic.tool.LABEL",
};

function SemiconductorSection({ el }: { el: SElement }) {
  const t = useT();
  const model = el.kind === "MOS" ? el.mos ?? DEFAULT_MOS : el.kind === "D" ? el.diode ?? DEFAULT_DIODE : el.bjt ?? DEFAULT_BJT;
  const field = el.kind === "MOS" ? "mos" : el.kind === "D" ? "diode" : "bjt";
  const set = (key: string, value: number | string) => update(el.id, { [field]: { ...model, [key]: value } });
  const fields: [string, string, string, number, number][] = el.kind === "MOS" ? [
    ["L_um", "L", "µm", 0.001, 10000], ["W_um", "W", "µm", 0.001, 100000], ["Vth_V", "|Vth|", "V", 0, 100], ["SS_mV_dec", "SS", "mV/dec", 60, 1000],
    ["k_uA_V2", "k", "µA/V²", 0.001, 1e6], ["lambda_per_V", "λ", "1/V", 0, 10],
  ] : el.kind === "D" ? [["Is_A", "I_s", "A", 1e-30, 1], ["n", "n", "", 0.1, 10]] : [["Is_A", "I_s", "A", 1e-30, 1], ["beta_F", "β_F", "", 0.01, 1e6], ["beta_R", "β_R", "", 0.01, 1e6]];
  const row = ([key, label, unit, min, max]: typeof fields[number]) => <Row key={key} label={<SubText text={["k", "n", "λ"].includes(label) ? `<i>${label}</i>` : label} />}><SIInput value={(model as unknown as Record<string, number>)[key]} unit={unit} min={min} max={max} onCommit={(v) => v != null && set(key, v)} testId={`insp-${key}`} ariaLabel={`${el.name} ${label}`} /></Row>;
  return <>
    {"polarity" in model && <Row label={t("schematic.insp.polarity")}><Segmented value={model.polarity} options={el.kind === "MOS" ? [{ v: "nmos", label: "NMOS" }, { v: "pmos", label: "PMOS" }] : [{ v: "npn", label: "NPN" }, { v: "pnp", label: "PNP" }]} onChange={(v) => set("polarity", v)} label={t("schematic.insp.polarity")} /></Row>}
    {fields.slice(0, el.kind === "MOS" ? 4 : 3).map(row)}
    {el.kind === "MOS" && <details className="insp-help"><summary>{t("schematic.insp.moreModel")}</summary>{fields.slice(4).map(row)}</details>}
    <details className="insp-help"><summary>{t("schematic.insp.modelGuide")}</summary><p>{t(el.kind === "MOS" ? "schematic.insp.mosGuide" : el.kind === "D" ? "schematic.insp.diodeGuide" : "schematic.insp.bjtGuide")}</p><a href={`${import.meta.env.BASE_URL}docs/basic-circuit-devices.md`} target="_blank" rel="noreferrer">{t("schematic.insp.modelEquations")} ↗</a></details>
  </>;
}

/** Comparator parameters: V_ref, output levels and hysteresis (SPICE numbers). */
function CmpSection({ el }: { el: SElement }) {
  const t = useT();
  const c = el.cmp ?? DEFAULT_CMP;
  const set = (patch: Partial<CmpParams>) => update(el.id, { cmp: { ...c, ...patch } });
  return (
    <>
      <Row label={t("schematic.insp.cmpRef")} hint={t("schematic.insp.cmpRefHint")}>
        <SIInput value={c.v_ref} unit="V" min={-1000} max={1000} onCommit={(v) => set({ v_ref: v ?? 0 })} testId="insp-cmp-ref" ariaLabel={t("schematic.insp.cmpRef")} />
      </Row>
      <Row label={t("schematic.insp.cmpHigh")}>
        <SIInput value={c.v_high} unit="V" min={-1000} max={1000} onCommit={(v) => set({ v_high: v ?? 1 })} testId="insp-cmp-high" ariaLabel={t("schematic.insp.cmpHigh")} />
      </Row>
      <Row label={t("schematic.insp.cmpLow")}>
        <SIInput value={c.v_low} unit="V" min={-1000} max={1000} onCommit={(v) => set({ v_low: v ?? 0 })} testId="insp-cmp-low" ariaLabel={t("schematic.insp.cmpLow")} />
      </Row>
      <Row label={t("schematic.insp.cmpHyst")} hint={t("schematic.insp.cmpHystHint")}>
        <SIInput value={c.hysteresis} unit="V" min={0} max={10} onCommit={(v) => set({ hysteresis: v ?? 0 })} testId="insp-cmp-hyst" ariaLabel={t("schematic.insp.cmpHyst")} />
      </Row>
      <div className="insp-note subtle">{t("schematic.insp.cmpNote")}</div>
    </>
  );
}

function update(id: string, patch: Partial<SElement>) {
  useSch.getState().commit((d) => ({ ...d, elements: d.elements.map((e) => (e.id === id ? { ...e, ...patch } : e)) }));
}

function OrientButtons({ ids }: { ids: string[] }) {
  const t = useT();
  const commit = useSch((s) => s.commit);
  return (
    <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
      <button type="button" className="btn sm" onClick={() => commit((d) => rotateItems(d, ids))} title="Ctrl+R">
        ⟳ {t("schematic.tool.rotate")}
      </button>
      <button type="button" className="btn sm" onClick={() => commit((d) => mirrorItems(d, ids))} title="Ctrl+E">
        ⇋ {t("schematic.tool.mirror")}
      </button>
      <button
        type="button"
        className="btn sm"
        onClick={() => {
          const st = useSch.getState();
          const r = duplicateItems(st.doc, ids);
          st.commit(() => r.doc);
          st.select(r.ids);
        }}
        title="Ctrl+D"
      >
        ⧉ {t("schematic.tool.duplicate")}
      </button>
      <button
        type="button"
        className="btn sm danger"
        data-testid="insp-delete"
        onClick={() => {
          const st = useSch.getState();
          st.commit((d) => deleteItems(d, ids));
          st.select([]);
        }}
        title="Del"
      >
        {t("schematic.tool.delete")}
      </button>
    </div>
  );
}

function StlSection({ el, conn }: { el: SElement; conn: Connectivity }) {
  const t = useT();
  const lang = useStore((s) => s.lang);
  useStore((s) => s.meta);
  useStore((s) => s.params.device);
  useDeviceLib((s) => s.devices);
  const tStop = useSch((s) => s.doc.tran.t_stop_s);
  const entries = libraryEntries();
  const ref = el.stl;
  const lib = ref ? entries.find((d) => d.id === ref.libId) : undefined;
  const noTrend = (l: LibDevice["stochastic"]["local_state"] | undefined) => (l ? { ...l, acquisition_trend: false } : null);
  const supported = !ref || isSupportedTechnology(ref.technology ?? "FDSOI");
  const same = !!(ref && lib && supported && isSupportedTechnology(lib.technology) && deepEqual(ref.device, lib.device) && deepEqual(noTrend(ref.local_state), noTrend(lib.stochastic.local_state)));
  // resolved per-cell data echoed by the server for the last run (folds at the circuit's V_GS)
  const run = useStore((s) => s.results.schematic?.data as SchematicRunData | undefined);
  const echo = run?.result.elements?.find((x) => x.name === el.name && (x.type ?? x.kind) === "STL") as
    | { nodes?: Record<string, string> | string[]; vgs_V?: number; folds?: { V_LU?: number | null; V_LD?: number | null }; latch_window?: boolean }
    | undefined;
  const d = ref?.device;
  const ls = ref?.local_state;
  const light = el.light ?? null;
  const lightMode: "device" | "dc" | "pulse" = !light ? "device" : light.kind === "pulse" ? "pulse" : "dc";
  const builtins = entries.filter((x) => x.builtin);
  const mine = entries.filter((x) => !x.builtin && x.id !== "current");
  return (
    <>
      <Row label={t("schematic.stl.device")} htmlFor="insp-stl-dev">
        <select
          id="insp-stl-dev"
          className="select"
          value={ref && entries.some((x) => x.id === ref.libId) ? ref.libId : ""}
          data-testid="insp-stl-device"
          onChange={(e) => { if (isSupportedTechnology(entries.find((d) => d.id === e.target.value)?.technology)) update(el.id, { stl: stlRefFor(e.target.value) }); }}
        >
          {!lib && <option value="">{ref?.name ?? "—"}</option>}
          <optgroup label={t("schematic.lib.builtin")}>
            {builtins.map((x) => (
              <option key={x.id} value={x.id}>
                {deviceName(x, lang)}
              </option>
            ))}
          </optgroup>
          {mine.length > 0 && (
            <optgroup label={t("schematic.lib.mine")}>
              {mine.map((x) => (
                <option key={x.id} value={x.id} disabled={!isSupportedTechnology(x.technology)}>
                  {x.name}{!isSupportedTechnology(x.technology) ? ` · ${x.technology} · ${t.lang === "ko" ? "미지원" : "Unsupported"}` : ""}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      </Row>
      <div className="insp-note">
        <span className={`dot-status ${!lib ? "warn" : same ? "ok" : "warn"}`} aria-hidden />
        <span>{!supported ? `${ref?.technology} · ${t.lang === "ko" ? "미지원" : "Unsupported"}` : !lib ? t("schematic.stl.missing") : same ? t("schematic.stl.upToDate") : t("schematic.stl.differs")}</span>
        {lib && isSupportedTechnology(lib.technology) && !same && (
          <button type="button" className="btn sm" data-testid="insp-stl-update" onClick={() => update(el.id, { stl: stlRefFor(lib.id) })}>
            {t("schematic.stl.update")}
          </button>
        )}
      </div>
      {d && (
        <div className="insp-meta mono" title={t("schematic.stl.snapshot")}><strong>{modelLabel(d)} Model</strong> · 
          {t("schematic.stl.summary", { iph: fmtSI(iphPA(d) * 1e-12, "A", 3), ls: ls?.mode ?? "none" })}
        </div>
      )}
      {d && <div className="insp-note subtle" data-testid="insp-gate-note">{t("schematic.stl.gateNote")}</div>}
      {el.stlTerminalMode === "legacy3" ? (
        <div className="insp-note">
          <span>{t("schematic.stl.legacy3")}</span>
          <button type="button" className="btn sm" data-testid="insp-expand-terminals" onClick={() => update(el.id, { stlTerminalMode: undefined })}>{t("schematic.stl.expand5")}</button>
        </div>
      ) : <div className="insp-terminal-defaults" data-testid="insp-terminal-defaults">
        <span>BG <strong>{optionalPinConnected(el.id, "bg", conn) ? t("schematic.stl.wired") : `${Number(d?.vbg ?? 0).toFixed(2)} V`}</strong></span>
        <span>B <strong>{optionalPinConnected(el.id, "b", conn) ? t("schematic.stl.wired") : t("schematic.stl.floating")}</strong></span>
      </div>}
      <details className="insp-help"><summary>{t("schematic.stl.terminalGuide")}</summary><p>{d?.model === "simple" ? (t.lang === "ko" ? "B는 소스측 바디 접점 u입니다. X.vb는 저장 전하의 등가 바디 전위 w = u + ID·RLRS로 표시합니다. Simple Model의 접점 확장이며 결정론적 BE를 사용합니다." : "B is the source-side body contact u. X.vb is the reservoir potential w = u + ID·RLRS. This Simple Model terminal extension uses deterministic BE.") : t("schematic.stl.terminalHelp")}</p></details>
      {echo && !(echo.nodes && ("bg" in echo.nodes || "b" in echo.nodes)) && (
        <div className="insp-echo" data-testid="insp-stl-echo">
          {echo.latch_window === false
            ? t("schematic.stl.noWindow")
            : t("schematic.stl.lastRun", {
                vgs: typeof echo.vgs_V === "number" ? echo.vgs_V.toFixed(2) : "—",
                vlu: typeof echo.folds?.V_LU === "number" ? `${echo.folds.V_LU.toFixed(3)} V` : "—",
                vld: typeof echo.folds?.V_LD === "number" ? `${echo.folds.V_LD.toFixed(3)} V` : "—",
              })}
        </div>
      )}
      <div className="insp-sub">{t("schematic.stl.light")}</div>
      <Segmented
        full
        value={lightMode}
        label={t("schematic.stl.light")}
        testId="insp-light-mode"
        options={[
          { v: "device", label: t("schematic.stl.lightDevice") },
          { v: "dc", label: t("schematic.stl.lightDc") },
          { v: "pulse", label: t("schematic.stl.lightPulse") },
        ]}
        onChange={(m) => {
          const base = d ? iphPA(d) : 0;
          const w: Wave | null = m === "device" ? null : m === "dc" ? { kind: "dc", value: base || 1 } : { kind: "pulse", v1: 0, v2: base || 1, td: 0, tr: 1e-6, tf: 1e-6, pw: tStop / 4, per: tStop / 2, ncycles: 0 };
          update(el.id, { light: w });
        }}
      />
      {light ? (
        <WaveEditor wave={light} unit="pA" tStop={tStop} kinds={["dc", "pulse"]} onChange={(w) => update(el.id, { light: w })} testId="insp-light" />
      ) : (
        d && <div className="small muted">{t("schematic.stl.lightNow", { v: fmtSI(iphPA(d) * 1e-12, "A", 3) })}</div>
      )}
    </>
  );
}

export function Inspector({ conn }: { conn: Connectivity }) {
  const t = useT();
  const doc = useSch((s) => s.doc);
  const selection = useSch((s) => s.selection);
  const nameRef = useRef<HTMLDivElement>(null);
  const els = doc.elements.filter((e) => selection.includes(e.id));
  const wires = doc.wires.filter((w) => selection.includes(w.id));
  const el = els.length === 1 && wires.length === 0 ? els[0] : null;
  useEffect(() => {
    const onFocus = () => nameRef.current?.querySelector<HTMLInputElement>("input")?.focus();
    window.addEventListener("sch-focus-inspector", onFocus);
    return () => window.removeEventListener("sch-focus-inspector", onFocus);
  }, []);
  const counts = useMemo(() => {
    const parts = doc.elements.filter((e) => CIRCUIT_KINDS.includes(e.kind));
    return { e: parts.length, n: conn.nets.filter((n) => isCircuitNet(n, conn) && !n.ground).length, s: parts.filter((e) => e.kind === "STL").length };
  }, [doc.elements, conn]);

  if (!selection.length) {
    return (
      <div className="insp" data-testid="inspector">
        <div className="insp-head">
          <span className="insp-title">{t("schematic.insp.circuit")}</span>
        </div>
        <Row label={t("schematic.file.name")}>
          <TextInput value={doc.name} placeholder={t("schematic.file.untitled")} onCommit={(v) => useSch.getState().commit((d) => ({ ...d, name: v.slice(0, 80) }))} testId="insp-circuit-name" />
        </Row>
        <div className="insp-meta">{t("schematic.insp.counts", counts)}</div>
        <p className="insp-empty">{t("schematic.insp.none")}</p>
        <details className="insp-help">
          <summary>{t("schematic.shortcuts")}</summary>
          <p>{t("schematic.shortcuts.body")}</p>
        </details>
      </div>
    );
  }
  if (!el) {
    const wireNet = wires.length === 1 && !els.length ? conn.wireNet.get(wires[0].id)?.name : undefined;
    return (
      <div className="insp" data-testid="inspector">
        <div className="insp-head">
          <span className="insp-title">{wireNet ? t("schematic.insp.wire", { node: wireNet }) : t("schematic.insp.multi", { n: selection.length })}</span>
        </div>
        <OrientButtons ids={selection} />
      </div>
    );
  }
  const nameOk = (v: string) => NAME_RE.test(v.trim()) && !doc.elements.some((e) => e.id !== el.id && e.name.toUpperCase() === v.trim().toUpperCase());
  const tStop = doc.tran.t_stop_s;
  return (
    <div className="insp" data-testid="inspector">
      <div className="insp-head">
        <span className="insp-icon">
          <PartIcon kind={el.kind} size={18} />
        </span>
        <span className="insp-title">{t(KIND_KEY[el.kind])}</span>
        {el.kind !== "GND" && el.kind !== "LABEL" && <span className="insp-name mono">{el.name}</span>}
      </div>
      {el.kind !== "GND" && el.kind !== "LABEL" && (
        <div ref={nameRef}>
          <Row label={t("schematic.insp.name")} htmlFor="insp-name">
            <TextInput id="insp-name" value={el.name} validate={nameOk} onCommit={(v) => update(el.id, { name: v.trim() })} testId="insp-name" />
          </Row>
        </div>
      )}
      {el.kind === "R" && (
        <Row label={t("schematic.insp.resistance")} hint={t("schematic.insp.siHint")}>
          <SIInput value={el.value} unit="Ω" min={1e-9} onCommit={(v) => update(el.id, { value: v ?? 1 })} testId="insp-value" ariaLabel={t("schematic.insp.resistance")} />
        </Row>
      )}
      {el.kind === "C" && (
        <Row label={t("schematic.insp.capacitance")} hint={t("schematic.insp.siHint")}>
          <SIInput value={el.value} unit="F" min={0} onCommit={(v) => update(el.id, { value: v ?? 0 })} testId="insp-value" ariaLabel={t("schematic.insp.capacitance")} />
        </Row>
      )}
      {(el.kind === "V" || el.kind === "I") && el.wave && <WaveEditor wave={el.wave} unit={el.kind === "V" ? "V" : "A"} tStop={tStop} onChange={(w) => update(el.id, { wave: w })} testId="insp-wave" />}
      {["MOS", "D", "BJT"].includes(el.kind) && <SemiconductorSection el={el} />}
      {el.kind === "STL" && <StlSection el={el} conn={conn} />}
      {el.kind === "CMP" && <CmpSection el={el} />}
      {el.kind === "LABEL" && (
        <div ref={nameRef}>
          <Row label={t("schematic.insp.label")} hint={t("schematic.insp.labelHint")} htmlFor="insp-label">
            <TextInput id="insp-label" value={el.label ?? ""} validate={(v) => LABEL_RE.test(v.trim())} onCommit={(v) => update(el.id, { label: v.trim() })} testId="insp-label" />
          </Row>
        </div>
      )}
      {el.kind === "GND" && <div className="insp-note subtle">{t("schematic.insp.gnd")}</div>}
      {el.kind !== "GND" && el.kind !== "LABEL" && (
        <div className="insp-conn">
          <span className="insp-sub">{t("schematic.insp.nodes")}</span>
          <div className="chips">
            {pinPositions(el).map((p) => (
              <span key={p.pin} className="chip" title={t((el.kind === "STL" && p.pin === "b" ? "schematic.pin.body" : `schematic.pin.${p.pin}`) as StrKey)}>
                {p.pin === "p" ? "+" : p.pin === "n" ? "−" : p.pin === "i" ? "IN" : p.pin === "q" ? "OUT" : p.pin.toUpperCase()} → {el.kind === "STL" && (p.pin === "b" || p.pin === "bg") && !optionalPinConnected(el.id, p.pin, conn) ? t(p.pin === "b" ? "schematic.stl.floating" : "schematic.stl.savedBias") : conn.pinNet.get(pinId(el.id, p.pin))?.name ?? "—"}
              </span>
            ))}
          </div>
        </div>
      )}
      <OrientButtons ids={[el.id]} />
    </div>
  );
}
