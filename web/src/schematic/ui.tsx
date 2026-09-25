// Small form widgets for the schematic editor: SPICE-number input with the interpreted value, labelled
// rows, switch, segmented control and a compact card (sidebar style).
import { useEffect, useId, useState, type ReactNode } from "react";
import { useT } from "../i18n";
import { fmtSI, parseSI, toSpice } from "./si";

export function SIInput({
  value, onCommit, unit, allowEmpty, placeholder, min, max, integer, testId, ariaLabel, id, showParsed = true,
}: {
  value: number | null | undefined;
  onCommit: (v: number | null) => void;
  unit?: string;
  allowEmpty?: boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  integer?: boolean;
  testId?: string;
  ariaLabel?: string;
  id?: string;
  showParsed?: boolean;
}) {
  const t = useT();
  // integers (runs, seeds, cycles) are shown plainly, other values in SPICE notation
  const show = (v: number | null | undefined) => (v == null ? "" : integer ? String(Math.round(v)) : toSpice(v));
  const [text, setText] = useState(show(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(show(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, editing]);
  const empty = text.trim() === "";
  let parsed = empty ? null : parseSI(text);
  if (parsed != null && integer) parsed = Math.round(parsed);
  const outOfRange = parsed != null && ((min != null && parsed < min) || (max != null && parsed > max));
  const invalid = (!empty && parsed === null) || (empty && !allowEmpty) || outOfRange;
  const commit = () => {
    setEditing(false);
    if (invalid) {
      setText(show(value));
      return;
    }
    const v = empty ? null : parsed;
    if (v !== value) onCommit(v);
  };
  const shown = parsed != null && !invalid ? fmtSI(parsed, unit ?? "", 4) : null;
  return (
    <span className="si-field">
      <span className="input-wrap">
        <input
          id={id}
          className={`input${invalid && editing ? " invalid" : ""}`}
          style={unit && unit.length > 2 ? { paddingRight: `calc(${unit.length}ch + 10px)` } : undefined}
          value={text}
          placeholder={placeholder}
          aria-label={ariaLabel}
          aria-invalid={invalid && editing}
          data-testid={testId}
          spellCheck={false}
          autoComplete="off"
          onFocus={() => setEditing(true)}
          onChange={(e) => {
            setEditing(true);
            setText(e.target.value);
          }}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              commit();
              (e.target as HTMLInputElement).blur();
            } else if (e.key === "Escape") {
              setEditing(false);
              setText(show(value));
              (e.target as HTMLInputElement).blur();
            }
            e.stopPropagation();
          }}
        />
        {unit && <span className="unit">{unit}</span>}
      </span>
      {showParsed && editing && (
        <span className={`si-parsed${invalid ? " bad" : ""}`} aria-live="polite">
          {invalid ? t("schematic.insp.invalid") : shown ? `= ${shown}` : ""}
        </span>
      )}
    </span>
  );
}

export function Row({ label, children, hint, htmlFor }: { label: ReactNode; children: ReactNode; hint?: ReactNode; htmlFor?: string }) {
  const t = useT();
  return (
    <div className="sch-row">
      <div className="sch-row-name"><label className="sch-row-label" htmlFor={htmlFor}>{label}</label>
        {hint && <details className="sch-field-help"><summary aria-label={t("details")}>i</summary><div>{hint}</div></details>}
      </div>
      <div className="sch-row-ctl">{children}</div>
    </div>
  );
}

export function Switch({ on, onChange, label, testId }: { on: boolean; onChange: (v: boolean) => void; label: string; testId?: string }) {
  return <button type="button" role="switch" aria-checked={on} className="switch" onClick={() => onChange(!on)} aria-label={label} data-testid={testId} />;
}

export function Segmented<V extends string>({ value, options, onChange, label, testId, full }: { value: V; options: { v: V; label: ReactNode; disabled?: boolean; title?: string }[]; onChange: (v: V) => void; label: string; testId?: string; full?: boolean }) {
  return (
    <div className={`seg${full ? " full" : ""}`} role="radiogroup" aria-label={label} data-testid={testId}>
      {options.map((o) => (
        <button key={o.v} type="button" role="radio" aria-checked={value === o.v} disabled={o.disabled} title={o.title} onClick={() => onChange(o.v)} data-v={o.v}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function TextInput({ value, onCommit, testId, ariaLabel, id, placeholder, validate }: { value: string; onCommit: (v: string) => void; testId?: string; ariaLabel?: string; id?: string; placeholder?: string; validate?: (v: string) => boolean }) {
  const [text, setText] = useState(value);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(value);
  }, [value, editing]);
  const bad = validate ? !validate(text) : false;
  const commit = () => {
    setEditing(false);
    if (!bad && text !== value) onCommit(text);
    else if (bad) setText(value);
  };
  return (
    <input
      id={id}
      className={`input text${bad && editing ? " invalid" : ""}`}
      value={text}
      placeholder={placeholder}
      aria-label={ariaLabel}
      data-testid={testId}
      spellCheck={false}
      autoComplete="off"
      onFocus={() => setEditing(true)}
      onChange={(e) => {
        setEditing(true);
        setText(e.target.value);
      }}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          commit();
          (e.target as HTMLInputElement).blur();
        } else if (e.key === "Escape") {
          setEditing(false);
          setText(value);
          (e.target as HTMLInputElement).blur();
        }
        e.stopPropagation();
      }}
    />
  );
}

/** Sidebar-style collapsible card (matches the parameter groups). In 간단히 the description is the head's tooltip. */
export function Card({ title, desc, children, testId, defaultOpen = true, actions, className }: { title: ReactNode; desc?: string; children: ReactNode; testId?: string; defaultOpen?: boolean; actions?: ReactNode; className?: string }) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  const t = useT();
  return (
    <section className={`group sch-card${className ? ` ${className}` : ""}`} data-testid={testId}>
      <div className="group-head">
        <button type="button" className="group-toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)} title={desc}>
          <svg className="chev" width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden>
            <path d="m6 9 6 6 6-6" />
          </svg>
          <span>
            <span className="group-title">{title}</span>
          </span>
        </button>
        {desc && <details className="sch-field-help"><summary aria-label={t("details")}>i</summary><div>{desc}</div></details>}
        {actions && <div className="group-actions">{actions}</div>}
      </div>
      {open && (
        <div className="group-body" id={id}>
          {children}
        </div>
      )}
    </section>
  );
}
