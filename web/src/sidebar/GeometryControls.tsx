// Geometry block at the top of the sidebar (D7): six dimension inputs, each with a click-only ⓘ guide (the
// parameter guide when content/params has an entry for it, else the one-line definition, reference value,
// range and the geometry-model document), plus one ⓘ for the block as a whole.
import { useEffect, useId, useRef, useState } from "react";
import { GuidePopover } from "../components/GuidePopover";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { useT } from "../i18n";
import { guideFor } from "../params/guideUi";
import { GEOMETRY_FIELDS } from "../params/schema";
import { GEOMETRY_LIMITS, isGeometryValue, REFERENCE_GEOMETRY, resolveGeometry, usesGeometryModel, type GeometryKey } from "../params/geometry";
import { TechContent } from "./Field";
import { presetDefaults, useStore } from "../state/store";
import { parseNumber, sciText, toInputString } from "../utils/format";
import { deepEqual } from "../utils/object";

const GEOMETRY_DOC = "docs/geometry-model.html";
const openGeometryDoc = () => window.open(`${import.meta.env.BASE_URL}${GEOMETRY_DOC}`, "_blank", "noopener,noreferrer");
const geomValue = (key: GeometryKey, v: number) => (key === "Nbody_cm3" ? sciText(v) : toInputString(v));

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
  const trigger = useRef<HTMLButtonElement | null>(null);
  const label = t.l(field.label);
  const sym = <SubText text={subs(field.sym ?? "")} />;
  return <div className="geometry-field" data-testid={`field-${field.key}`}>
    <div className="geometry-label">
      <label htmlFor={id} title={label}>{sym}</label>
      <GuidePopover id={field.key} testId={`tip-${field.key}`} triggerRef={trigger}
        ariaLabel={`${label} ${t.lang === "ko" ? "설계 가이드" : "design guide"}`}
        sym={sym} label={label}
        guide={guideFor(field.key) ?? guideFor(field.geometryKey)}
        technical={<TechContent help={t.l(field.help)}
          def={`${geomValue(key, REFERENCE_GEOMETRY[key])}\u00a0${unit}`}
          range={`${t("range", { min: geomValue(key, min), max: geomValue(key, max) })}\u00a0${unit}`} />}
        onMore={openGeometryDoc} />
      <span className="geometry-unit">{unit}</span>
    </div>
    <input id={id} className={`input${invalid ? " invalid" : ""}`} type="text"
      inputMode={key === "Nbody_cm3" ? "text" : "decimal"} autoComplete="off" spellCheck={false}
      aria-label={`${label} (${unit})`} aria-invalid={invalid}
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
  const title = t("g.geometry");
  return <section className="geometry-controls" data-testid="geometry-controls" aria-label={title}>
    <div className="geometry-head">
      <h2>{title}</h2>
      <GuidePopover id="geometry" testId="tip-geometry"
        ariaLabel={`${title} ${t.lang === "ko" ? "설계 가이드" : "design guide"}`}
        label={title}
        description={t("g.geometry.desc")}
        onMore={openGeometryDoc} />
      {extended && <span className="geometry-scope" data-testid="geometry-extended" title={t("g.geometry.extended.title")}>{t("g.geometry.extended")}</span>}
      {changed && <button className="link-btn geometry-reset" type="button" data-testid="geometry-reset"
        aria-label={t("reset.aria", { group: title })}
        onClick={() => update((p) => ({ ...p, device: { ...p.device, geometry: resolveGeometry(baseline) } }))}>{t("reset")}</button>}
    </div>
    <div className="geometry-grid">
      {GEOMETRY_FIELDS.map((field) => <GeometryInput key={field.key} field={field} value={geometry[field.geometryKey]} onChange={commit} />)}
    </div>
  </section>;
}
