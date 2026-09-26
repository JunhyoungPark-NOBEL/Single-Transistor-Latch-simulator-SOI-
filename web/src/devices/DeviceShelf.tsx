import { modelLabel } from "../params/model";
import { lazy, Suspense, useState } from "react";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { clone, deepEqual } from "../utils/object";
import { BiristorGlyph } from "../components/Logo";
import { loadDeviceIntoParams, SaveDeviceForm } from "./DeviceCard";
import { geometryFromDevice, isSupportedTechnology, stochOf, type LibDevice } from "./library";
import { MAX_USER_DEVICES, useDeviceLib } from "./store";
import { Modal } from "./Modal";
import { ModelExportButton } from "./ModelExportButton";
import "./workspace.css";
const DeviceManager = lazy(() => import("./DeviceManager"));

export function DeviceShelf() {
  const t = useT();
  const devices = useDeviceLib((s) => s.devices);
  const device = useStore((s) => s.params.device);
  const sweep = useStore((s) => s.params.sweep);
  const [saving, setSaving] = useState(false);
  const [manage, setManage] = useState(false);
  const [replace, setReplace] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const L = (ko: string, en: string) => t.lang === "ko" ? ko : en;
  const full = devices.length >= MAX_USER_DEVICES;
  const load = (d: LibDevice) => { if (!loadDeviceIntoParams(d)) return; setMessage(t("schematic.dev.loadedToast", { name: d.name })); };
  const overwrite = (d: LibDevice) => {
    if (!isSupportedTechnology(d.technology)) return;
    const s = useStore.getState();
    useDeviceLib.getState().update(d.id, { device: clone({ ...s.params.device, geometry: geometryFromDevice(s.params.device) }), stochastic: stochOf(s.params.stochastic), technology: "FDSOI", calibration_label: clone(s.meta.presets[s.params.device.preset]?.label ?? {ko:"사용자 정의",en:"Custom"}), geometry: geometryFromDevice(s.params.device) });
    setReplace(null); setMessage(t("schematic.dev.savedToast", { name: d.name }));
  };
  return <aside className="device-shelf" aria-label={L("사용자 소자", "My devices")} data-testid="device-shelf">
    <div className="shelf-head"><h2>{L("사용자 소자", "My devices")}</h2><span className="shelf-count">{devices.length} / 5</span></div>
    <button className="btn primary shelf-save" type="button" disabled={full} onClick={() => setSaving(true)} data-testid="dev-save-open"><span aria-hidden>＋</span>{L("현재 소자 저장", "Save current device")}</button>
    <p className="shelf-caption">{full ? L("저장된 소자를 선택해 업데이트할 수 있습니다.", "Update a saved device to keep this calibration.") : L("보정한 소자를 저장하고 회로에서 사용하세요.", "Keep calibrations ready for your circuits.")}</p>
    <div className="shelf-slots">
      {Array.from({ length: Math.max(MAX_USER_DEVICES, devices.length) }, (_, i) => {
        const d = devices[i];
        return d ? <article key={d.id} className="device-slot filled" data-testid={`device-slot-${i + 1}`}>
          <div className="slot-heading"><BiristorGlyph size={23} /><strong title={d.name}>{d.name}</strong><span className="slot-number">{String(i + 1).padStart(2, "0")}</span></div>
          <span className="slot-meta">{modelLabel(d.device)} · {d.technology}{!isSupportedTechnology(d.technology) && <> · {L("미지원", "Unsupported")}</>} · <i>V</i><sub>G</sub> {d.device.vg} V</span>
          <div className="slot-actions"><button type="button" className="link-btn" disabled={!isSupportedTechnology(d.technology)} onClick={() => load(d)}>{L("불러오기", "Load")}</button><button type="button" className="link-btn" disabled={!isSupportedTechnology(d.technology)} onClick={() => setReplace(replace === d.id ? null : d.id)}>{L("업데이트", "Update")}</button></div>
          {replace === d.id && <div className="slot-confirm"><span>{L("현재 보정값으로 변경할까요?", "Replace with the current calibration?")}</span><button type="button" className="btn sm" onClick={() => setReplace(null)}>{L("취소", "Cancel")}</button><button type="button" className="btn sm primary" onClick={() => overwrite(d)}>{L("저장", "Save")}</button></div>}
        </article> : <button type="button" className="device-slot empty" key={i} onClick={() => setSaving(true)} data-testid={`device-slot-${i + 1}`} aria-label={L(`소자 ${i + 1} 저장`, `Save device ${i + 1}`)}><span className="slot-number">{String(i + 1).padStart(2, "0")}</span><span className="slot-empty-plus">＋</span><span>{L("소자 저장", "Save device")}</span></button>;
      })}
    </div>
    <button type="button" className="btn sm ghost shelf-manage" onClick={() => setManage(true)} data-testid="dev-manage">{L("소자 관리 · 가져오기", "Manage & import")}</button>
    <div className="shelf-export"><ModelExportButton device={device} sweep={sweep} name={devices.find((d) => isSupportedTechnology(d.technology) && deepEqual(d.device, device))?.name} /></div>
    <span className="shelf-local-note">{L("이 브라우저에 저장됩니다.", "Saved in this browser.")}</span>
    {message && <div className="dev-toast" role="status" data-testid="dev-toast">{message}<button type="button" className="icon-btn xs" aria-label={L("닫기", "Close")} onClick={() => setMessage("")}>×</button></div>}
    {saving && <Modal title={L("현재 소자 저장", "Save current device")} onClose={() => setSaving(false)} width={400} testId="save-device-dialog"><SaveDeviceForm onDone={(d) => {setSaving(false); if(d) setMessage(t("schematic.dev.savedToast",{name:d.name}));}} /></Modal>}
    {manage && <Suspense fallback={null}><DeviceManager onClose={() => setManage(false)} /></Suspense>}
  </aside>;
}
