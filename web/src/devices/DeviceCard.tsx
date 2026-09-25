// Compact "Device" card for the sidebar (4 rows): "소자 · FDSOI  [라이브러리]", the geometry line, the calibration
// preset selector (loads the preset defaults), and the one-line preset state ("암조건 · V_G −2 V · 0.4 V/s", or
// "● 기준 보정에서 수정 · 초기화") with the "기술 · 저장 ▸" disclosure (technology chips, saved devices, "Save
// as device"; closed in 간단히, open in 모두 보기, always in the DOM).
import { lazy, Suspense, useEffect, useState } from "react";
import type { PresetId } from "../api/types";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { fill } from "../i18n/strings.ux";
import { useIsAll } from "../state/layout";
import { PRESET_IDS } from "../state/presets";
import { presetDefaults, useStore } from "../state/store";
import { GuideText } from "../components/GuidePopover";
import { clone, deepEqual } from "../utils/object";
import { iphPA } from "../utils/payload";
import { geometryFromMeta, stochOf, TECHNOLOGIES, type Geometry, type LibDevice } from "./library";
import { useDeviceLib } from "./store";
import "./devices.css";

const DeviceManager = lazy(() => import("./DeviceManager"));

const SEP = "\u2009·\u2009";

export function GeometryLine({ g }: { g: Geometry }) {
  const f = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
  return (
    <>
      L<sub>g</sub> {f(g.Lg_nm)} nm{SEP}W {f(g.W_nm)} nm{SEP}T<sub>Si</sub> {f(g.Tsi_nm)} nm{SEP}EOT {f(g.EOT_nm)} nm
    </>
  );
}

/** Load a library device into the Device tab parameters (device block + stochastic local-state settings). */
export function loadDeviceIntoParams(d: LibDevice) {
  const st = useStore.getState();
  if (d.builtin && (d.device.preset === "paper" || d.device.preset === "photo")) {
    st.loadPreset(d.device.preset);
    return;
  }
  useStore.setState((s) => ({
    preset: d.device.preset,
    params: {
      ...s.params,
      device: clone(d.device),
      stochastic: { ...s.params.stochastic, local_state: clone(d.stochastic.local_state), carrier_noise: d.stochastic.carrier_noise, ld_carrier_noise: d.stochastic.ld_carrier_noise },
    },
  }));
}

export function SaveDeviceForm({ onDone }: { onDone: (d: LibDevice | null) => void }) {
  const t = useT();
  const n = useDeviceLib((s) => s.devices.length) + 1;
  const [name, setName] = useState(t("schematic.dev.defaultName", { n }));
  const [notes, setNotes] = useState("");
  const save = () => {
    const s = useStore.getState();
    const base = s.meta.presets[s.params.device.preset] ?? s.meta.presets[s.preset];
    const dev = useDeviceLib.getState().add({
      name: name.trim() || t("schematic.dev.defaultName", { n }),
      technology: "FDSOI",
      geometry: geometryFromMeta(s.meta),
      calibration_label: base?.label ?? { ko: "사용자 정의", en: "Custom" },
      device: clone(s.params.device),
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
      <p className="dev-hint">{t("schematic.dev.captures")}</p>
      <div className="dev-actions">
        <button type="button" className="btn sm ghost" onClick={() => onDone(null)}>
          {t("schematic.dev.cancel")}
        </button>
        <button type="submit" className="btn sm primary" data-testid="dev-save-submit">
          {t("schematic.dev.save")}
        </button>
      </div>
    </form>
  );
}

const IconLib = () => (
  <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="3" y="4" width="5" height="16" rx="1" />
    <rect x="10" y="4" width="5" height="16" rx="1" />
    <path d="m17 5 4 1-3 14-4-1" />
  </svg>
);

const num = (v: number) => String(Number(v.toPrecision(3))).replace(/^-/, "−");
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** "암조건 · V_G −2 V · 0.4 V/s" (the current condition; equals the preset while it is unmodified). */
function useConditionLine(): string {
  const t = useT();
  const device = useStore((s) => s.params.device);
  const rate = useStore((s) => s.params.sweep.rate_V_per_s);
  const preset = useStore((s) => s.preset);
  const conds = useStore((s) => s.meta.measured_photo_conditions ?? []);
  const iph = iphPA(device);
  // the illumination calibration starts with the light off: say so, or "암조건" reads as "the light did not apply"
  const pMax = Math.max(0, ...conds.map((c) => c.power_mW));
  const light =
    iph > 0
      ? device.light.mode === "power"
        ? `P ${num(device.light.power_mW)} mW`
        : `I_PH ${num(iph)} pA`
      : preset === "photo" && pMax > 0
        ? fill(t.l(GUIDE["dev.photoDark"]), { max: num(pMax) })
        : cap(t.l(GUIDE["light.dark"]));
  return `${light} · V_G ${num(device.vg)} V · ${num(rate)} V/s`;
}

export function DeviceCard() {
  const t = useT();
  const all = useIsAll();
  const preset = useStore((s) => s.preset);
  const meta = useStore((s) => s.meta);
  const load = useStore((s) => s.loadPreset);
  const modified = useStore((s) => {
    const d = presetDefaults(s);
    return !(deepEqual(s.params.device, d.device) && deepEqual(s.params.sweep, d.sweep) && deepEqual(s.params.stochastic, d.stochastic));
  });
  const devices = useDeviceLib((s) => s.devices);
  const [saving, setSaving] = useState(false);
  const [manage, setManage] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  // "기술 · 저장": follows the layout (open in 모두 보기) until the user toggles it
  const [moreOpen, setMoreOpen] = useState<boolean | null>(null);
  const more = moreOpen ?? all;
  useEffect(() => setMoreOpen(null), [all]);
  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 2600);
    return () => clearTimeout(id);
  }, [toast]);
  const condition = useConditionLine();
  const geometry = geometryFromMeta(meta);
  const baseName = t(`preset.${preset}` as never);
  const fullLabel = meta.presets[preset]?.label?.[t.lang] ?? preset;
  return (
    <div className="preset-card dev-card compact" data-testid="preset-card">
      <div className="dev-head">
        <span className="preset-head">{fill(t.l(GUIDE["dev.head"]), { tech: "FDSOI" })}</span>
        <button type="button" className="btn sm ghost dev-lib-btn" onClick={() => setManage(true)} aria-label={t("schematic.dev.manageAria")} title={t("schematic.dev.manageAria")} data-testid="dev-manage">
          <IconLib />
          {t("schematic.dev.manage")}
        </button>
      </div>
      <div className="dev-geom" title={t("schematic.dev.geometryFixed")} data-testid="dev-geometry">
        <GeometryLine g={geometry} />
      </div>
      <div className="preset-options" role="radiogroup" aria-label={t("schematic.dev.calibration")}>
        {PRESET_IDS.map((id: PresetId) => (
          <button key={id} type="button" role="radio" aria-checked={preset === id && !modified} className="preset-opt" data-testid={`preset-${id}`} onClick={() => load(id)} title={`${meta.presets[id]?.label?.[t.lang] ?? id}${id === "photo" ? `\n${t.l(GUIDE["dev.photo.tip"])}` : ""}`}>
            {t(`preset.${id}` as never)}
          </button>
        ))}
      </div>
      <div className={`dev-foot${more ? " more-open" : ""}`}>
        <div className="preset-note" data-testid="preset-label" aria-live="polite" title={fullLabel}>
          {modified ? (
            <>
              <span className="chg" aria-hidden />
              <span className="pn-text">{fill(t.l(GUIDE["dev.modified"]), { base: baseName })}</span>
              <span className="pn-sep" aria-hidden>·</span>
              <button type="button" className="link-btn" onClick={() => load(preset)} aria-label={fill(t.l(GUIDE["dev.reset.aria"]), { base: baseName })}>
                {t("reset")}
              </button>
            </>
          ) : (
            <span className="pn-text muted">
              <GuideText text={condition} plain />
            </span>
          )}
        </div>
        <details
          className="dev-more"
          data-testid="dev-more"
          open={more}
          onToggle={(e) => {
            const o = (e.currentTarget as HTMLDetailsElement).open;
            if (o !== more) setMoreOpen(o);
          }}
        >
          <summary title={t.l(GUIDE["dev.more.title"])}>{t.l(GUIDE["dev.more"])}</summary>
          <div className="dev-more-body">
            <div className="techsel" role="radiogroup" aria-label={t("schematic.dev.tech")} data-testid="tech-select">
              {TECHNOLOGIES.map((x) => (
                <button
                  key={x.id}
                  type="button"
                  role="radio"
                  aria-checked={x.id === "FDSOI"}
                  disabled={!x.active}
                  className="techsel-opt"
                  title={x.active ? x.id : t("schematic.dev.techSoon", { tech: x.id })}
                  data-testid={`tech-${x.id}`}
                >
                  {x.id}
                  {!x.active && <span className="soon">{t("schematic.dev.soon")}</span>}
                </button>
              ))}
            </div>
            <div className="dev-row">
              <select
                className="select"
                value=""
                aria-label={t("schematic.dev.saved")}
                data-testid="dev-load"
                disabled={!devices.length}
                onChange={(e) => {
                  const d = devices.find((x) => x.id === e.target.value);
                  if (d) {
                    loadDeviceIntoParams(d);
                    setToast(t("schematic.dev.loadedToast", { name: d.name }));
                  }
                }}
              >
                <option value="">{devices.length ? t("schematic.dev.loadSaved") : t("schematic.dev.noneSaved")}</option>
                {devices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
              <button type="button" className={`btn sm${saving ? " active" : ""}`} onClick={() => setSaving(!saving)} aria-expanded={saving} data-testid="dev-save-open">
                + {t("schematic.dev.saveAs")}
              </button>
            </div>
            {saving && (
              <SaveDeviceForm
                onDone={(d) => {
                  setSaving(false);
                  if (d) setToast(t("schematic.dev.savedToast", { name: d.name }));
                }}
              />
            )}
          </div>
        </details>
      </div>
      {toast && (
        <div className="dev-toast" role="status" data-testid="dev-toast">
          {toast}
        </div>
      )}
      {manage && (
        <Suspense fallback={null}>
          <DeviceManager onClose={() => setManage(false)} />
        </Suspense>
      )}
    </div>
  );
}
