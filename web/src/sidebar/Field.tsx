// One parameter field: label with KaTeX symbol, ⓘ tooltip (symbol, meaning, code index, default),
// changed-from-default dot, numeric input with range validation (+ slider), toggle, select, segmented.
import { useEffect, useId, useState } from "react";
import { InfoTip } from "../components/Tooltip";
import { Tex } from "../components/Tex";
import { useT, type T } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { scaleOf, unitOf, type Ctx, type FieldDef, type Option } from "../params/schema";
import { nearlyEqual, parseNumber, toInputString } from "../utils/format";

export interface FieldProps {
  f: FieldDef;
  ctx: Ctx;
  value: unknown;
  def: unknown;
  onChange: (v: unknown) => void;
}

const optLabel = (t: T, o: Option) => (typeof o.label === "string" ? t(o.label as StrKey) : t.l(o.label));

function fmtDefault(t: T, f: FieldDef, ctx: Ctx, def: unknown): string {
  if (def === null) return t("auto");
  if (Array.isArray(def)) return def.length ? def.join(", ") : "[ ]";
  if (typeof def === "boolean") return def ? t("on") : t("off");
  if (typeof def === "number") {
    if (f.options) {
      const o = f.options.find((x) => x.value === def);
      if (o) return optLabel(t, o);
    }
    const u = unitOf(f, ctx);
    return `${toInputString(def * scaleOf(f, ctx), 5)}${u ? " " + u : ""}`;
  }
  if (typeof def === "string" && f.options) {
    const o = f.options.find((x) => x.value === def);
    if (o) return optLabel(t, o);
  }
  return String(def ?? "—");
}

function TipContent({ f, ctx, def }: { f: FieldDef; ctx: Ctx; def: unknown }) {
  const t = useT();
  return (
    <>
      <div className="tip-head">
        {f.sym && <Tex tex={f.sym} />}
        <span>{t.l(f.label)}</span>
        {f.code && <span className="tip-code">{f.code}</span>}
      </div>
      <div>{t.l(f.help)}</div>
      <div className="tip-def">
        {t("default")}: {fmtDefault(t, f, ctx, def)}
        {f.min !== undefined && f.max !== undefined && f.type !== "toggle" && !f.options ? ` · ${t("range", { min: toInputString(f.min, 4), max: toInputString(f.max, 4) })}` : ""}
      </div>
    </>
  );
}

function Label({ f, ctx, def, changed, htmlFor }: { f: FieldDef; ctx: Ctx; def: unknown; changed: boolean; htmlFor?: string }) {
  const t = useT();
  return (
    <div className="field-label">
      {changed && <span className="field-changed" title={t("changed")} aria-label={t("changed")} role="img" />}
      {f.sym && <Tex tex={f.sym} className="sym" />}
      <label htmlFor={htmlFor} className="lbl" title={t.l(f.label)}>
        {t.l(f.label)}
      </label>
      {f.experimental && <span className="exp" title={t("experimental")}>{t("experimental.short")}</span>}
      <InfoTip label={`${t.l(f.label)} — ${t.l(f.help)}`} content={<TipContent f={f} ctx={ctx} def={def} />} testId={`tip-${f.key}`} />
    </div>
  );
}

function toSlider(v: number, f: FieldDef): number {
  if (f.slider === "log") return Math.log10(Math.max(v, f.min && f.min > 0 ? f.min : 1e-30));
  return v;
}
function fromSlider(s: number, f: FieldDef): number {
  if (f.slider === "log") {
    const v = 10 ** s;
    const p = Number(v.toPrecision(3));
    return p;
  }
  return s;
}

function NumberField({ f, ctx, value, def, onChange }: FieldProps) {
  const t = useT();
  const id = useId();
  const scale = scaleOf(f, ctx);
  const unit = unitOf(f, ctx);
  const isAuto = value === null && !!f.auto;
  const num = typeof value === "number" ? value : Number(value);
  const disp = isAuto ? NaN : num * scale;
  const show = (v: number) => (!Number.isFinite(v) ? "" : f.int ? String(Math.round(v)) : toInputString(v));
  const [text, setText] = useState(show(disp));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(show(disp));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disp, editing]);
  const parsed = parseNumber(text);
  const outOfRange = parsed !== null && ((f.min !== undefined && parsed < f.min - 1e-12) || (f.max !== undefined && parsed > f.max + 1e-12));
  const emptyAuto = f.auto && text.trim() === "";
  const error = emptyAuto ? null : parsed === null ? t("invalid") : outOfRange ? t("range", { min: toInputString(f.min, 4), max: toInputString(f.max, 4) }) : null;
  const changed = def === null ? value !== null : typeof def === "number" && !nearlyEqual(num, def);

  const commit = (v: number) => {
    let x = f.int ? Math.round(v) : v;
    if (f.min !== undefined) x = Math.max(f.min, x);
    if (f.max !== undefined) x = Math.min(f.max, x);
    const stored = x / scale;
    if (isAuto || !nearlyEqual(stored, num, 1e-12)) onChange(stored);
  };
  const stepBy = (dir: number, big: boolean) => {
    const base = parsed ?? disp;
    let step = f.step ?? (f.slider === "log" ? Math.max(Math.abs(base) * 0.1, 1e-30) : Math.max(Math.abs(base) * 0.05, 0.01));
    if (big) step *= 10;
    const nv = Number((base + dir * step).toPrecision(10));
    setText(toInputString(nv));
    if (!(f.min !== undefined && nv < f.min) && !(f.max !== undefined && nv > f.max)) commit(nv);
  };

  const sMin = f.slider ? toSlider(f.min ?? 0, f) : 0;
  const sMax = f.slider ? toSlider(f.max ?? 1, f) : 1;
  const sStep = f.slider === "log" ? (sMax - sMin) / 200 : f.step ?? (sMax - sMin) / 200;

  return (
    <div className="field" data-testid={`field-${f.key}`}>
      <div className="field-row">
        <Label f={f} ctx={ctx} def={def} changed={changed} htmlFor={id} />
        <div className="input-wrap">
          <input
            id={id}
            className={`input${error ? " invalid" : ""}`}
            type="text"
            inputMode="decimal"
            spellCheck={false}
            autoComplete="off"
            value={text}
            aria-invalid={!!error}
            aria-describedby={error ? `${id}-err` : undefined}
            onFocus={() => setEditing(true)}
            placeholder={f.auto ? t("auto") : undefined}
            onChange={(e) => {
              setText(e.target.value);
              if (f.auto && e.target.value.trim() === "") {
                if (value !== null) onChange(null);
                return;
              }
              const v = parseNumber(e.target.value);
              if (v !== null && !((f.min !== undefined && v < f.min) || (f.max !== undefined && v > f.max))) commit(v);
            }}
            onBlur={() => {
              setEditing(false);
              if (error) setText(show(disp));
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              else if (e.key === "Escape") {
                setText(show(disp));
                (e.target as HTMLInputElement).blur();
              } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
                e.preventDefault();
                stepBy(e.key === "ArrowUp" ? 1 : -1, e.shiftKey);
              }
            }}
            style={unit ? { paddingRight: `${Math.min(64, 12 + unit.length * 6.5)}px` } : { paddingRight: 8 }}
          />
          {unit && <span className="unit">{unit}</span>}
        </div>
      </div>
      {f.auto && (
        <div className="auto-row">
          <button type="button" className="chip" aria-pressed={isAuto} onClick={() => { setText(""); onChange(null); }} title={t("auto.hint")} data-testid={`auto-${f.key}`}>
            {t("auto")}
          </button>
          {isAuto && <span className="small muted">{t("auto.hint")}</span>}
        </div>
      )}
      {error && editing && (
        <div className="field-err" id={`${id}-err`} role="alert">
          {error}
        </div>
      )}
      {f.slider && !isAuto && (
        <div className="field-slider">
          <input
            type="range"
            min={sMin}
            max={sMax}
            step={sStep}
            value={Math.min(sMax, Math.max(sMin, toSlider(disp, f)))}
            aria-label={`${t.l(f.label)} (${unit})`}
            onChange={(e) => commit(fromSlider(Number(e.target.value), f))}
          />
        </div>
      )}
    </div>
  );
}

function ToggleField({ f, ctx, value, def, onChange }: FieldProps) {
  const id = useId();
  const on = !!value;
  return (
    <div className="field" data-testid={`field-${f.key}`}>
      <div className="toggle-row">
        <Label f={f} ctx={ctx} def={def} changed={typeof def === "boolean" && def !== on} htmlFor={id} />
        <button id={id} type="button" role="switch" aria-checked={on} className="switch" onClick={() => onChange(!on)} />
      </div>
    </div>
  );
}

function SelectField({ f, ctx, value, def, onChange }: FieldProps) {
  const t = useT();
  const id = useId();
  const opts = f.options ?? [];
  return (
    <div className="field" data-testid={`field-${f.key}`}>
      <div className="field-stack">
        <Label f={f} ctx={ctx} def={def} changed={def !== undefined && def !== value} htmlFor={id} />
        <select
          id={id}
          className="select"
          value={String(value)}
          onChange={(e) => {
            const o = opts.find((x) => String(x.value) === e.target.value);
            if (o) onChange(o.value);
          }}
        >
          {opts.map((o) => (
            <option key={String(o.value)} value={String(o.value)} disabled={o.disabled} title={o.note ? t.l(o.note) : undefined}>
              {optLabel(t, o)}
              {o.experimental ? ` — ${t("experimental")}` : ""}
              {o.disabled && o.note ? ` (${t.l(o.note)})` : ""}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function SegmentedField({ f, ctx, value, def, onChange }: FieldProps) {
  const t = useT();
  const opts = f.options ?? [];
  return (
    <div className="field" data-testid={`field-${f.key}`}>
      <Label f={f} ctx={ctx} def={def} changed={def !== undefined && def !== value} />
      <div className="seg full" role="radiogroup" aria-label={t.l(f.label)}>
        {opts.map((o) => (
          <button key={String(o.value)} type="button" role="radio" aria-checked={o.value === value} onClick={() => onChange(o.value)}>
            {optLabel(t, o).replace(/\s*\(.*\)$/, "")}
          </button>
        ))}
      </div>
    </div>
  );
}

function ListField({ f, ctx, value, def, onChange }: FieldProps) {
  const t = useT();
  const id = useId();
  const scale = scaleOf(f, ctx);
  const unit = unitOf(f, ctx);
  const arr = Array.isArray(value) ? (value as number[]) : [];
  const toText = (a: number[]) => a.map((v) => toInputString(v * scale)).join(", ");
  const [text, setText] = useState(toText(arr));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(toText(arr));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(arr), editing, scale]);
  const parts = text.split(/[,;\s]+/).filter(Boolean);
  const nums = parts.map(parseNumber);
  const bad = nums.some((v) => v === null || (f.min !== undefined && v < f.min) || (f.max !== undefined && v > f.max));
  const changed = JSON.stringify(arr) !== JSON.stringify(def ?? []);
  const commit = () => {
    if (bad) return;
    onChange((nums as number[]).map((v) => v / scale));
  };
  return (
    <div className="field" data-testid={`field-${f.key}`}>
      <div className="field-stack">
        <Label f={f} ctx={ctx} def={def} changed={changed} htmlFor={id} />
        <div className="input-wrap">
          <input
            id={id}
            className={`input${bad ? " invalid" : ""}`}
            style={{ textAlign: "left", paddingRight: 40 }}
            value={text}
            placeholder={t("list.placeholder")}
            spellCheck={false}
            aria-invalid={bad}
            onFocus={() => setEditing(true)}
            onChange={(e) => setText(e.target.value)}
            onBlur={() => {
              setEditing(false);
              commit();
            }}
            onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          />
          {unit && <span className="unit">{unit}</span>}
        </div>
        {bad && <div className="field-err">{t("invalid")}{f.min !== undefined ? ` · ${t("range", { min: toInputString(f.min, 4), max: toInputString(f.max, 4) })}` : ""}</div>}
      </div>
    </div>
  );
}

export function Field(props: FieldProps) {
  switch (props.f.type) {
    case "list": return <ListField {...props} />;
    case "toggle": return <ToggleField {...props} />;
    case "select": return <SelectField {...props} />;
    case "segmented": return <SegmentedField {...props} />;
    default: return <NumberField {...props} />;
  }
}
