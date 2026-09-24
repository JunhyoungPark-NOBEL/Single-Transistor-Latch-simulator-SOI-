// Properties panel for the selection: name, value (SPICE numbers), source waveforms, STL device from the
// library (snapshot + "update from library") and its light waveform, net-label text, orientation.
import { useEffect, useMemo, useRef } from "react";
import type { Wave } from "../api/circuitCustom";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { deviceName, type LibDevice } from "../devices/library";
import { useDeviceLib } from "../devices/store";
import { useStore } from "../state/store";
import { deepEqual } from "../utils/object";
import { iphPA } from "../utils/payload";
import { deleteItems, duplicateItems, mirrorItems, rotateItems } from "./edit";
import { LABEL_RE, NAME_RE } from "./erc";
import { CIRCUIT_KINDS, pinPositions, type SElement } from "./model";
import type { Connectivity } from "./nets";
import { pinId } from "./nets";
import { fmtSI } from "./si";
import type { SchematicRunData } from "./run";
import { libraryEntries, stlRefFor, useSch } from "./store";
import { PartIcon } from "./Symbols";
import { Row, Segmented, SIInput, TextInput } from "./ui";
import { WaveEditor } from "./WaveEditor";

const KIND_KEY: Record<SElement["kind"], StrKey> = {
  R: "schematic.tool.R", C: "schematic.tool.C", V: "schematic.tool.V", I: "schematic.tool.I", STL: "schematic.tool.STL", GND: "schematic.tool.GND", LABEL: "schematic.tool.LABEL",
};

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

function StlSection({ el }: { el: SElement }) {
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
  const same = !!(ref && lib && deepEqual(ref.device, lib.device) && deepEqual(noTrend(ref.local_state), noTrend(lib.stochastic.local_state)));
  // resolved per-cell data echoed by the server for the last run (folds at the circuit's V_GS)
  const run = useStore((s) => s.results.schematic?.data as SchematicRunData | undefined);
  const echo = run?.result.elements?.find((x) => x.name === el.name && (x.type ?? x.kind) === "STL") as
    | { vgs_V?: number; folds?: { V_LU?: number | null; V_LD?: number | null }; latch_window?: boolean }
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
          onChange={(e) => update(el.id, { stl: stlRefFor(e.target.value) })}
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
                <option key={x.id} value={x.id}>
                  {x.name}
                </option>
              ))}
            </optgroup>
          )}
          <option value="current">{t("schematic.lib.current")}</option>
        </select>
      </Row>
      <div className="insp-note">
        <span className={`dot-status ${!lib ? "warn" : same ? "ok" : "warn"}`} aria-hidden />
        <span>{!lib ? t("schematic.stl.missing") : same ? t("schematic.stl.upToDate") : t("schematic.stl.differs")}</span>
        {lib && !same && (
          <button type="button" className="btn sm" data-testid="insp-stl-update" onClick={() => update(el.id, { stl: stlRefFor(lib.id) })}>
            {t("schematic.stl.update")}
          </button>
        )}
      </div>
      {d && (
        <div className="insp-meta mono" title={t("schematic.stl.snapshot")}>
          {t("schematic.stl.summary", { iph: fmtSI(iphPA(d) * 1e-12, "A", 3), ls: ls?.mode ?? "none" })}
        </div>
      )}
      {d && <div className="insp-note subtle" data-testid="insp-gate-note">{t("schematic.stl.gateNote", { vg: d.vg.toFixed(2) })}</div>}
      {echo && (
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
    return { e: parts.length, n: conn.nets.filter((n) => n.pins.length && !n.ground).length, s: parts.filter((e) => e.kind === "STL").length };
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
      {el.kind === "STL" && <StlSection el={el} />}
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
              <span key={p.pin} className="chip" title={t(`schematic.pin.${p.pin}` as StrKey)}>
                {p.pin === "p" ? "+" : p.pin === "n" ? "−" : p.pin.toUpperCase()} → {conn.pinNet.get(pinId(el.id, p.pin))?.name ?? "—"}
              </span>
            ))}
          </div>
        </div>
      )}
      <OrientButtons ids={[el.id]} />
    </div>
  );
}
