import { modelLabel } from "../params/model";
// Compact device identity and explicit reset to the single default device.
import { useState } from "react";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { resolveBackGate } from "../params/geometry";
import { clone, deepEqual } from "../utils/object";
import { geometryFromDevice, isSupportedTechnology, stochOf, type Geometry, type LibDevice } from "./library";
import { MAX_USER_DEVICES, useDeviceLib } from "./store";
import "./devices.css";
import { SubText } from "../plots/SubText";


const SEP = "\u2009·\u2009";

export function GeometryLine({ g }: { g: Geometry }) {
  const f = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
  return <SubText text={`L ${f(g.Lg_nm)} nm${SEP}W ${f(g.W_nm)} nm${SEP}T<sub>Si</sub> ${f(g.Tsi_nm)} nm${SEP}EOT ${f(g.EOT_nm)} nm${SEP}T<sub>box</sub> ${f(g.Tbox_nm)} nm${SEP}N<sub>body</sub> ${g.Nbody_cm3.toExponential(2).replace("e+", "e")} cm⁻³`} />;
}

/** Load a library device into the Device tab parameters (device block + stochastic local-state settings). */
export function loadDeviceIntoParams(d: LibDevice): boolean {
  if (!isSupportedTechnology(d.technology)) return false;
  const st = useStore.getState();
  if (d.builtin && (d.device.preset === "paper" || d.device.preset === "photo")) {
    st.loadPreset(d.device.preset);
    return true;
  }
  useStore.setState((s) => ({
    preset: d.device.preset,
    mode: d.device.model === "simple" ? "deterministic" : s.mode,
    params: {
      ...s.params,
      device: clone({ ...d.device, geometry: geometryFromDevice(d.device) }),
      stochastic: { ...s.params.stochastic, local_state: clone(d.stochastic.local_state), carrier_noise: d.stochastic.carrier_noise, ld_carrier_noise: d.stochastic.ld_carrier_noise },
    },
  }));
  return true;
}

export function SaveDeviceForm({ onDone }: { onDone: (d: LibDevice | null) => void }) {
  const t = useT();
  const count = useDeviceLib((s) => s.devices.length);
  const n = count + 1;
  const [name, setName] = useState(t("schematic.dev.defaultName", { n }));
  const [notes, setNotes] = useState("");
  const save = () => {
    if (useDeviceLib.getState().devices.length >= MAX_USER_DEVICES) return;
    const s = useStore.getState();
    const base = s.meta.presets[s.params.device.preset] ?? s.meta.presets[s.preset];
    const dev = useDeviceLib.getState().add({
      name: name.trim() || t("schematic.dev.defaultName", { n }),
      technology: "FDSOI",
      geometry: geometryFromDevice(s.params.device, s.meta),
      calibration_label: base?.label ?? { ko: "사용자 정의", en: "Custom" },
      device: clone({ ...s.params.device, geometry: geometryFromDevice(s.params.device) }),
      stochastic: stochOf(s.params.stochastic),
      notes: notes.trim(),
    });
    onDone(dev);
  };
  return (
    <form
      className="dev-save"
      data-testid="dev-save-form"
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
    >
      <label className="dev-field">
        <span>{t("schematic.dev.name")}</span>
        <input className="input text" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} autoFocus data-testid="dev-save-name" />
      </label>
      <label className="dev-field">
        <span>{t("schematic.dev.notes")}</span>
        <textarea className="input text" rows={2} value={notes} maxLength={2000} placeholder={t("schematic.dev.notesPh")} onChange={(e) => setNotes(e.target.value)} />
      </label>
      <p className="dev-hint">{t.lang === "ko" ? "현재 모델·형상·보정값을 저장합니다." : "Saves the model, geometry and calibration."}</p>
      <div className="dev-actions">
        <button type="button" className="btn sm ghost" onClick={() => onDone(null)}>
          {t("schematic.dev.cancel")}
        </button>
        <button type="submit" disabled={count >= MAX_USER_DEVICES} className="btn sm primary" data-testid="dev-save-submit">
          {t("schematic.dev.save")}
        </button>
      </div>
    </form>
  );
}

export function DeviceCard() {
  const t = useT();
  const device = useStore((s) => s.params.device);
  const meta = useStore((s) => s.meta);
  const load = useStore((s) => s.loadPreset);
  const saved = useDeviceLib((s) => s.devices);
  const current = { ...device, geometry: geometryFromDevice(device), vbg: resolveBackGate(device.vbg) };
  const matched = saved.find((d) => isSupportedTechnology(d.technology) && deepEqual({ ...d.device, geometry: geometryFromDevice(d.device), vbg: resolveBackGate(d.device.vbg) }, current));
  const reference = meta.presets.paper.device;
  const modified = !deepEqual(current, { ...reference, geometry: geometryFromDevice(reference), vbg: resolveBackGate(reference.vbg) });
  return <section className="preset-card dev-card compact" data-testid="preset-card">
    <div className="dev-head"><span className="preset-head">{matched?.name ?? "Device 1"}{modified && !matched ? " ·" : ""}</span><span className="tech-chip">{modelLabel(device)} · FDSOI</span></div>
    {modified && <div className="preset-note" data-testid="preset-label"><span className="chg" /><span>{matched ? (t.lang === "ko" ? "저장된 소자" : "Saved device") : (t.lang === "ko" ? "수정됨" : "Modified")}</span><button type="button" className="link-btn" data-testid="preset-paper" onClick={() => load("paper")}>{t("reset")}</button></div>}
  </section>;
}
