import { useEffect, useId, useState } from "react";
import { GuidePopover } from "../components/GuidePopover";
import { useT } from "../i18n";
import { GEOMETRY_FIELDS } from "../params/schema";
import { GEOMETRY_LIMITS, isGeometryValue, resolveGeometry, usesGeometryModel, type GeometryKey } from "../params/geometry";
import { presetDefaults, useStore } from "../state/store";
import { parseNumber, toInputString } from "../utils/format";
import { deepEqual } from "../utils/object";
import { SubText } from "../plots/SubText";
import { GeometrySchematic } from "./GeometrySchematic";
import "./geometry-schematic.css";

const SYMBOLS: Record<GeometryKey, string> = {
  Lg_nm: "L", W_nm: "W", Tsi_nm: "T<sub>Si</sub>", EOT_nm: "T<sub>ox</sub> (EOT)",
  Tbox_nm: "T<sub>box</sub>", Nbody_cm3: "N<sub>body</sub>",
};

function GeometryInput({ field, value, onChange, id, active, onFocus }: {
  field: typeof GEOMETRY_FIELDS[number]; value: number; onChange: (key: GeometryKey, value: number) => void;
  id: string; active: boolean; onFocus: () => void;
}) {
  const t = useT();
  const [text, setText] = useState(toInputString(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => { if (!editing) setText(toInputString(value)); }, [value, editing]);
  const key = field.geometryKey;
  const parsed = parseNumber(text);
  const invalid = parsed === null || !isGeometryValue(key, parsed);
  const { min, max, step } = GEOMETRY_LIMITS[key];
  const unit = String(field.unit);
  const range = `${toInputString(min)}–${toInputString(max)} ${unit}`;
  return <div className="geometry-field" data-testid={`field-${field.key}`} data-active={active}>
    <label htmlFor={id} title={t.l(field.label)}><span><SubText text={SYMBOLS[key]} /></span><span className="geometry-unit">{unit}</span></label>
    <input id={id} className={`input${invalid ? " invalid" : ""}`} type="text"
      inputMode={key === "Nbody_cm3" ? "text" : "decimal"} autoComplete="off" spellCheck={false}
      aria-label={`${field.sym} (${unit})`} aria-invalid={invalid}
      aria-describedby={editing && invalid ? `${id}-error` : undefined}
      data-testid={`geometry-${key}`} value={text}
      onFocus={() => { setEditing(true); onFocus(); }}
      onChange={(e) => {
        setText(e.target.value);
        const v = parseNumber(e.target.value);
        if (isGeometryValue(key, v)) onChange(key, v);
      }}
      onBlur={() => { setEditing(false); setText(toInputString(value)); }}
      onKeyDown={(e) => {
        if (e.key === "Escape" || e.key === "Enter") {
          setText(toInputString(value)); e.currentTarget.blur();
        } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
          e.preventDefault();
          const next = Number(((parsed ?? value) + (e.key === "ArrowUp" ? 1 : -1) * step * (e.shiftKey ? 10 : 1)).toPrecision(14));
          if (isGeometryValue(key, next)) { setText(toInputString(next)); onChange(key, next); }
        }
      }} />
    {editing && invalid && <span className="geometry-error" id={`${id}-error`} role="alert">{range}</span>}
  </div>;
}

export function GeometryControls() {
  const t = useT();
  const fieldId = useId();
  const [active, setActive] = useState<GeometryKey | null>(null);
  const current = useStore((s) => s.params.device.geometry);
  const vbg = useStore((s) => s.params.device.vbg);
  const simple = useStore((s) => s.params.device.model === "simple");
  const baseline = useStore((s) => presetDefaults(s).device.geometry);
  const update = useStore((s) => s.updateParams);
  const geometry = resolveGeometry(current);
  const changed = !deepEqual(geometry, resolveGeometry(baseline));
  const extended = usesGeometryModel({ geometry, vbg });
  const commit = (key: GeometryKey, value: number) => update((p) => ({
    ...p, device: { ...p.device, geometry: { ...resolveGeometry(p.device.geometry), [key]: value } },
  }));
  return <section className="geometry-controls" data-testid="geometry-controls" aria-label="Geometry">
    <div className="geometry-technology" role="group" data-testid="technology-selector" aria-label={t.lang === "ko" ? "소자 구조" : "Device structure"}>
      <button type="button" aria-pressed="true" data-testid="technology-FDSOI">FDSOI</button>
      {["PDSOI", "Bulk"].map((technology) => <button key={technology} type="button" disabled aria-pressed="false"
        data-testid={`technology-${technology}`} title={t.lang === "ko" ? "미지원" : "Not supported"}
        aria-label={`${technology} · ${t.lang === "ko" ? "미지원" : "Not supported"}`}>{technology}</button>)}
    </div>
    <div className="geometry-head">
      <h2>Geometry</h2>
      <GuidePopover id="geometry" testId="tip-geometry"
        ariaLabel={t.lang === "ko" ? "Geometry 설계 가이드" : "Geometry design guide"}
        label="Geometry"
        description={simple ? (t.lang === "ko" ? "Simple Model은 기준 파라미터를 현재 형상으로 환산합니다. 길이·도핑에 따라 β, 두께에 따라 수명이 달라집니다. 치수별 측정 보정은 필요합니다." : "Simple Model scales reference parameters to this geometry. Length/doping scale β; thickness scales lifetime. Measurements are still needed to calibrate each geometry.") : t.lang === "ko"
          ? "치수·백게이트 변경은 결정론적 VSCM·CSVM 해석을 지원합니다. 기준 보정점 밖은 미보정 확장입니다. EOT는 등가 두께이며, 완전 공핍 조건은 자동 판정하지 않습니다."
          : "Geometry/back-gate changes support deterministic VSCM and CSVM. Off-reference dimensions are uncalibrated. EOT is an equivalent thickness; full depletion is not checked automatically."}
        onMore={() => window.open(`${import.meta.env.BASE_URL}docs/${simple ? "simple-model" : "geometry-model"}.html`, "_blank", "noopener,noreferrer")} />
      {extended && <span className="geometry-scope" data-testid="geometry-extended" title={t.lang === "ko" ? "치수·백게이트 변경: 결정론적 해석" : "Geometry/back-gate changes: deterministic analysis"}>{t.lang === "ko" ? "확장 모델" : "Extended"}</span>}
      {changed && <button className="link-btn geometry-reset" type="button" data-testid="geometry-reset"
        onClick={() => update((p) => ({ ...p, device: { ...p.device, geometry: resolveGeometry(baseline) } }))}>{t("reset")}</button>}
    </div>
    <GeometrySchematic active={active} onHighlight={setActive} onSelect={(key) => {
      setActive(key);
      document.getElementById(`${fieldId}-${key}`)?.focus({ preventScroll: true });
    }} />
    <div className="geometry-grid">
      {GEOMETRY_FIELDS.map((field) => <GeometryInput key={field.key} id={`${fieldId}-${field.geometryKey}`} field={field}
        value={geometry[field.geometryKey]} onChange={commit} active={active === field.geometryKey} onFocus={() => setActive(field.geometryKey)} />)}
    </div>
  </section>;
}
