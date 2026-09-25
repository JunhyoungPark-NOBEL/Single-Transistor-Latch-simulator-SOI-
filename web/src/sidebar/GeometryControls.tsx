import { useEffect, useId, useState } from "react";
import { GuidePopover } from "../components/GuidePopover";
import { useT } from "../i18n";
import { GEOMETRY_FIELDS } from "../params/schema";
import { GEOMETRY_LIMITS, isGeometryValue, resolveGeometry, usesGeometryModel, type GeometryKey } from "../params/geometry";
import { presetDefaults, useStore } from "../state/store";
import { parseNumber, toInputString } from "../utils/format";
import { deepEqual } from "../utils/object";

function GeometryInput({ field, value, onChange }: {
  field: typeof GEOMETRY_FIELDS[number]; value: number; onChange: (key: GeometryKey, value: number) => void;
}) {
  const t = useT();
  const id = useId();
  const [text, setText] = useState(toInputString(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => { if (!editing) setText(toInputString(value)); }, [value, editing]);
  const key = field.geometryKey;
  const parsed = parseNumber(text);
  const invalid = parsed === null || !isGeometryValue(key, parsed);
  const { min, max, step } = GEOMETRY_LIMITS[key];
  const unit = String(field.unit);
  const range = `${toInputString(min)}–${toInputString(max)} ${unit}`;
  return <div className="geometry-field" data-testid={`field-${field.key}`}>
    <label htmlFor={id} title={t.l(field.label)}><span>{field.sym}</span><span className="geometry-unit">{unit}</span></label>
    <input id={id} className={`input${invalid ? " invalid" : ""}`} type="text"
      inputMode={key === "Nbody_cm3" ? "text" : "decimal"} autoComplete="off" spellCheck={false}
      aria-label={`${field.sym} (${unit})`} aria-invalid={invalid}
      aria-describedby={editing && invalid ? `${id}-error` : undefined}
      data-testid={`geometry-${key}`} value={text}
      onFocus={() => setEditing(true)}
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
  const current = useStore((s) => s.params.device.geometry);
  const vbg = useStore((s) => s.params.device.vbg);
  const baseline = useStore((s) => presetDefaults(s).device.geometry);
  const update = useStore((s) => s.updateParams);
  const geometry = resolveGeometry(current);
  const changed = !deepEqual(geometry, resolveGeometry(baseline));
  const extended = usesGeometryModel({ geometry, vbg });
  const commit = (key: GeometryKey, value: number) => update((p) => ({
    ...p, device: { ...p.device, geometry: { ...resolveGeometry(p.device.geometry), [key]: value } },
  }));
  return <section className="geometry-controls" data-testid="geometry-controls" aria-label="Geometry">
    <div className="geometry-head">
      <h2>Geometry</h2>
      <GuidePopover id="geometry" testId="tip-geometry"
        ariaLabel={t.lang === "ko" ? "Geometry 설계 가이드" : "Geometry design guide"}
        label="Geometry"
        description={t.lang === "ko"
          ? "치수·백게이트 변경은 결정론 VSCM·CSVM 해석을 지원하며, 기준 보정점 밖에서는 확장 모델입니다."
          : "Geometry and back-gate changes support deterministic VSCM and CSVM beyond the reference calibration."}
        onMore={() => window.open(`${import.meta.env.BASE_URL}docs/geometry-model.html`, "_blank", "noopener,noreferrer")} />
      {extended && <span className="geometry-scope" data-testid="geometry-extended" title={t.lang === "ko" ? "치수·백게이트 변경: 결정론 해석" : "Geometry/back-gate changes: deterministic analysis"}>{t.lang === "ko" ? "확장 모델" : "Extended"}</span>}
      {changed && <button className="link-btn geometry-reset" type="button" data-testid="geometry-reset"
        onClick={() => update((p) => ({ ...p, device: { ...p.device, geometry: resolveGeometry(baseline) } }))}>{t("reset")}</button>}
    </div>
    <div className="geometry-grid">
      {GEOMETRY_FIELDS.map((field) => <GeometryInput key={field.key} field={field} value={geometry[field.geometryKey]} onChange={commit} />)}
    </div>
  </section>;
}
