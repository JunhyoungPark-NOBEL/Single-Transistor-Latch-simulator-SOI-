// Device library manager (modal): built-in read-only devices + the user's devices — load into the Device
// tab, place in the schematic, rename, duplicate, delete, export/import JSON (validated).
import { useRef, useState } from "react";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { downloadText } from "../utils/csv";
import { fmtSI } from "../utils/format";
import { iphPA } from "../utils/payload";
import { useCircuitView } from "../circuit/view";
import { loadDeviceIntoParams, GeometryLine } from "./DeviceCard";
import { builtinDevices, calibPart, deviceName, exportLibraryJson, parseLibraryJson, TECHNOLOGIES, type LibDevice } from "./library";
import { Modal } from "./Modal";
import { useDeviceLib, validationBase } from "./store";

function Row({ d, onPlace, onLoad }: { d: LibDevice; onPlace: (d: LibDevice) => void; onLoad: (d: LibDevice) => void }) {
  const t = useT();
  const lib = useDeviceLib();
  const [name, setName] = useState(d.name);
  const [confirm, setConfirm] = useState(false);
  const dev = d.device;
  const iph = iphPA(dev);
  const created = d.created ? new Date(d.created) : null;
  return (
    <li className={`dm-row${d.builtin ? " builtin" : ""}`} data-testid={`dm-row-${d.builtin ? d.id : d.name}`}>
      <div className="dm-main">
        <div className="dm-title">
          {d.builtin ? (
            <strong>{deviceName(d, t.lang)}</strong>
          ) : (
            <input
              className="input text dm-name"
              value={name}
              aria-label={t("schematic.lib.rename")}
              maxLength={80}
              onChange={(e) => setName(e.target.value)}
              onBlur={() => (name.trim() && name !== d.name ? lib.rename(d.id, name) : setName(d.name))}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                if (e.key === "Escape") setName(d.name);
              }}
              data-testid="dm-name"
            />
          )}
          <span className="badge">{d.technology}</span>
          {d.builtin && <span className="badge det">{t("schematic.lib.builtin")}</span>}
        </div>
        <div className="dm-meta">
          <GeometryLine g={d.geometry} />
        </div>
        <div className="dm-meta mono">
          V<sub>G</sub> {dev.vg.toFixed(2)} V · {iph ? <>I<sub>PH</sub> {fmtSI(iph * 1e-12, "A", 3)}</> : t("schematic.lib.dark")} · {t("schematic.lib.local", { mode: d.stochastic.local_state.mode })}
          {!d.builtin && ` · ${t("schematic.dev.fromDevice", { base: calibPart(t.l(d.calibration_label)) })}`}
        </div>
        {d.notes && <div className="dm-notes">{d.notes}</div>}
        {created && <div className="dm-date">{t("schematic.lib.created")}: {created.toLocaleString(t.lang === "ko" ? "ko-KR" : "en-GB")}</div>}
      </div>
      <div className="dm-actions">
        <button type="button" className="btn sm" onClick={() => onLoad(d)} data-testid="dm-load">
          {t("schematic.lib.load")}
        </button>
        <button type="button" className="btn sm primary" onClick={() => onPlace(d)} data-testid="dm-place">
          {t("schematic.lib.place")}
        </button>
        <button type="button" className="btn sm ghost" onClick={() => lib.duplicate(d, t("schematic.lib.copySuffix"))} data-testid="dm-duplicate">
          {t("schematic.lib.duplicate")}
        </button>
        <button type="button" className="btn sm ghost" onClick={() => downloadText(`${d.name.replace(/[^\w.-]+/g, "_")}.stl-device.json`, exportLibraryJson([d]))}>
          {t("schematic.lib.exportOne")}
        </button>
        {!d.builtin &&
          (confirm ? (
            <button type="button" className="btn sm danger" onClick={() => lib.remove(d.id)} onBlur={() => setConfirm(false)} autoFocus data-testid="dm-delete-confirm" title={t("schematic.lib.deleteConfirm", { name: d.name })}>
              {t("schematic.lib.delete")}?
            </button>
          ) : (
            <button type="button" className="btn sm ghost danger" onClick={() => setConfirm(true)} data-testid="dm-delete">
              {t("schematic.lib.delete")}
            </button>
          ))}
      </div>
    </li>
  );
}

export default function DeviceManager({ onClose }: { onClose: () => void }) {
  const t = useT();
  const meta = useStore((s) => s.meta);
  const devices = useDeviceLib((s) => s.devices);
  const importMany = useDeviceLib((s) => s.importMany);
  const fileRef = useRef<HTMLInputElement>(null);
  const [msg, setMsg] = useState<{ text: string; err?: boolean } | null>(null);
  const builtins = builtinDevices(meta);
  const place = (d: LibDevice) => {
    onClose();
    useStore.getState().setTab("circuit");
    useCircuitView.getState().setView("schematic");
    void import("../schematic/store").then((m) => {
      m.useSch.getState().set({ stlChoice: d.id, tool: { kind: "place", el: "STL", rot: 0, mirror: false } });
    });
  };
  const loadDev = (d: LibDevice) => {
    loadDeviceIntoParams(d);
    setMsg({ text: t("schematic.dev.loadedToast", { name: deviceName(d, t.lang) }) });
  };
  return (
    <Modal
      title={t("schematic.lib.title")}
      onClose={onClose}
      width={720}
      testId="device-manager"
      footer={
        <>
          <span className="small muted">{t("schematic.lib.desc")}</span>
          <span className="spacer" />
          <button type="button" className="btn sm" onClick={() => fileRef.current?.click()} data-testid="dm-import">
            {t("schematic.lib.import")}
          </button>
          <button type="button" className="btn sm" disabled={!devices.length} onClick={() => downloadText("stl-device-library.json", exportLibraryJson(devices))} data-testid="dm-export">
            {t("schematic.lib.exportAll")}
          </button>
        </>
      }
    >
      <div className="dm-tech" aria-label={t("schematic.dev.tech")}>
        {TECHNOLOGIES.map((x) => (
          <span key={x.id} className={`techsel-opt${x.active ? " on" : ""}`} aria-disabled={!x.active} title={x.active ? x.id : t("schematic.dev.techSoon", { tech: x.id })}>
            {x.id}
            {!x.active && <span className="soon">{t("schematic.dev.soon")}</span>}
          </span>
        ))}
        <span className="small muted">{t("schematic.dev.geometryFixed")}</span>
      </div>
      {msg && (
        <div className={`callout ${msg.err ? "err" : "info"}`} role="status" data-testid="dm-msg">
          {msg.text}
        </div>
      )}
      <h3 className="dm-h">{t("schematic.lib.builtin")}</h3>
      <ul className="dm-list">
        {builtins.map((d) => (
          <Row key={d.id} d={d} onPlace={place} onLoad={loadDev} />
        ))}
      </ul>
      <h3 className="dm-h">{t("schematic.lib.mine")}</h3>
      {devices.length === 0 ? (
        <p className="dm-empty">{t("schematic.lib.empty")}</p>
      ) : (
        <ul className="dm-list" data-testid="dm-user-list">
          {devices.map((d) => (
            <Row key={d.id} d={d} onPlace={place} onLoad={loadDev} />
          ))}
        </ul>
      )}
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        hidden
        data-testid="dm-import-input"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const r = parseLibraryJson(await f.text(), validationBase());
          if (r.error) {
            setMsg({ text: t(`schematic.lib.err.${r.error}`), err: true });
            return;
          }
          const n = importMany(r.devices);
          setMsg({ text: `${t("schematic.lib.imported", { n })}${r.skipped ? ` · ${t("schematic.lib.skipped", { n: r.skipped })}` : ""}` });
        }}
      />
    </Modal>
  );
}
