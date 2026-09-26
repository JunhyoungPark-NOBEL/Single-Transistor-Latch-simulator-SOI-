// Source waveform editor: DC / PULSE / PWL (table + paste) / SINE with a tiny preview over 0 … t_stop.
import { useMemo, useState } from "react";
import type { Wave, WaveKind } from "../api/circuitCustom";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { IconX } from "../components/icons";
import { fmtSI, parseSI } from "./si";
import { Row, Segmented, SIInput } from "./ui";
import { convertWave, wavePoints } from "./waves";

export function WavePreview({ wave, tStop, unit }: { wave: Wave; tStop: number; unit: string }) {
  const t = useT();
  const W = 260;
  const H = 58;
  const { path, lo, hi, truncated } = useMemo(() => {
    const pts = wavePoints(wave, tStop, 800);
    const vs = pts.v.length ? pts.v : [0];
    let lo = Math.min(...vs);
    let hi = Math.max(...vs);
    if (hi === lo) {
      hi += Math.abs(hi) * 0.1 || 1;
      lo -= Math.abs(lo) * 0.1 || 1;
    }
    const x = (tt: number) => 4 + ((W - 8) * tt) / Math.max(tStop, 1e-18);
    const y = (v: number) => H - 6 - ((H - 12) * (v - lo)) / (hi - lo);
    const d = pts.t.map((tt, i) => `${i ? "L" : "M"}${x(tt).toFixed(1)} ${y(pts.v[i]).toFixed(1)}`).join(" ");
    return { path: d, lo, hi, truncated: pts.truncated };
  }, [wave, tStop]);
  return (
    <figure className="wave-preview" aria-label={t("schematic.wave.preview")}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" data-testid="wave-preview">
        <line x1={4} x2={W - 4} y1={H - 6} y2={H - 6} className="wp-axis" />
        <path d={path} className="wp-line" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption>
        <span>{fmtSI(hi, unit, 3)}</span>
        <span className="muted">
          0 – {fmtSI(tStop, "s", 3)}
          {truncated ? ` · ${t("schematic.wave.truncated")}` : ""}
        </span>
        <span>{fmtSI(lo, unit, 3)}</span>
      </figcaption>
    </figure>
  );
}

const PULSE_FIELDS: { k: "v1" | "v2" | "td" | "tr" | "tf" | "pw" | "per" | "ncycles"; label: StrKey; unit: "v" | "s" | "" }[] = [
  { k: "v1", label: "schematic.wave.v1", unit: "v" },
  { k: "v2", label: "schematic.wave.v2", unit: "v" },
  { k: "td", label: "schematic.wave.td", unit: "s" },
  { k: "tr", label: "schematic.wave.tr", unit: "s" },
  { k: "tf", label: "schematic.wave.tf", unit: "s" },
  { k: "pw", label: "schematic.wave.pw", unit: "s" },
  { k: "per", label: "schematic.wave.per", unit: "s" },
  { k: "ncycles", label: "schematic.wave.ncycles", unit: "" },
];
const SINE_FIELDS: { k: "vo" | "va" | "freq" | "td" | "theta"; label: StrKey; unit: "v" | "s" | "Hz" | "1/s" }[] = [
  { k: "vo", label: "schematic.wave.vo", unit: "v" },
  { k: "va", label: "schematic.wave.va", unit: "v" },
  { k: "freq", label: "schematic.wave.freq", unit: "Hz" },
  { k: "td", label: "schematic.wave.td", unit: "s" },
  { k: "theta", label: "schematic.wave.theta", unit: "1/s" },
];

export function WaveEditor({ wave, unit, onChange, tStop, kinds = ["dc", "pulse", "pwl", "sine"], testId = "wave" }: { wave: Wave; unit: string; onChange: (w: Wave) => void; tStop: number; kinds?: WaveKind[]; testId?: string }) {
  const t = useT();
  const [paste, setPaste] = useState<string | null>(null);
  const u = (x: "v" | "s" | "" | "Hz" | "1/s") => (x === "v" ? unit : x);
  return (
    <div className="wave-editor" data-testid={testId}>
      <Segmented
        full
        value={wave.kind}
        label={t("schematic.wave.title")}
        testId={`${testId}-kind`}
        options={kinds.map((k) => ({ v: k, label: t(`schematic.wave.${k}` as StrKey) }))}
        onChange={(k) => onChange(convertWave(wave, k))}
      />
      {wave.kind === "dc" && (
        <Row label={t("schematic.wave.value")}>
          <SIInput value={wave.value} unit={unit} onCommit={(v) => onChange({ ...wave, value: v ?? 0 })} testId={`${testId}-value`} ariaLabel={t("schematic.wave.value")} />
        </Row>
      )}
      {wave.kind === "pulse" && (
        <div className="wave-grid">
          {PULSE_FIELDS.map((f) => (
            <Row key={f.k} label={t(f.label)} hint={f.k === "ncycles" ? t("schematic.wave.ncyclesHint") : undefined}>
              <SIInput
                value={wave[f.k]}
                unit={u(f.unit)}
                min={f.unit === "s" || f.k === "ncycles" ? 0 : undefined}
                integer={f.k === "ncycles"}
                onCommit={(v) => onChange({ ...wave, [f.k]: v ?? 0 })}
                testId={`${testId}-${f.k}`}
                ariaLabel={t(f.label)}
              />
            </Row>
          ))}
        </div>
      )}
      {wave.kind === "sine" && (
        <div className="wave-grid">
          {SINE_FIELDS.map((f) => (
            <Row key={f.k} label={t(f.label)}>
              <SIInput value={wave[f.k]} unit={u(f.unit)} min={f.k === "freq" || f.k === "td" ? 0 : undefined} onCommit={(v) => onChange({ ...wave, [f.k]: v ?? 0 })} testId={`${testId}-${f.k}`} ariaLabel={t(f.label)} />
            </Row>
          ))}
        </div>
      )}
      {wave.kind === "pwl" && (
        <div className="pwl">
          <div className="pwl-head">
            <span>{t("schematic.wave.pwlT")} (s)</span>
            <span>
              {t("schematic.wave.pwlV")} ({unit})
            </span>
            <span />
          </div>
          <div className="pwl-rows">
            {wave.t.map((tt, i) => (
              <div className="pwl-row" key={i}>
                <SIInput value={tt} unit="s" min={0} showParsed={false} onCommit={(v) => onChange({ ...wave, t: wave.t.map((x, j) => (j === i ? v ?? 0 : x)) })} ariaLabel={`${t("schematic.wave.pwlT")} ${i + 1}`} testId={`${testId}-pwl-t-${i}`} />
                <SIInput value={wave.v[i]} unit={unit} showParsed={false} onCommit={(v) => onChange({ ...wave, v: wave.v.map((x, j) => (j === i ? v ?? 0 : x)) })} ariaLabel={`${t("schematic.wave.pwlV")} ${i + 1}`} testId={`${testId}-pwl-v-${i}`} />
                <button type="button" className="icon-btn xs" disabled={wave.t.length <= 1} aria-label={t("schematic.wave.pwlRemove")} title={t("schematic.wave.pwlRemove")} onClick={() => onChange({ ...wave, t: wave.t.filter((_, j) => j !== i), v: wave.v.filter((_, j) => j !== i) })}>
                  <IconX size={12} />
                </button>
              </div>
            ))}
          </div>
          <div className="row" style={{ gap: 6 }}>
            <button
              type="button"
              className="btn sm"
              onClick={() => {
                const last = wave.t[wave.t.length - 1] ?? 0;
                const step = wave.t.length > 1 ? last - wave.t[wave.t.length - 2] : tStop / 10;
                onChange({ ...wave, t: [...wave.t, last + (step > 0 ? step : tStop / 10)], v: [...wave.v, wave.v[wave.v.length - 1] ?? 0] });
              }}
              disabled={wave.t.length >= 2000}
            >
              + {t("schematic.wave.pwlAdd")}
            </button>
            <button type="button" className="btn sm ghost" onClick={() => setPaste(paste === null ? wave.t.map((x, i) => `${x} ${wave.v[i]}`).join("\n") : null)} aria-expanded={paste !== null}>
              {t("schematic.wave.pwlPaste")}
            </button>
          </div>
          {paste !== null && (
            <div className="pwl-paste">
              <textarea className="input text" rows={5} value={paste} onChange={(e) => setPaste(e.target.value)} placeholder={t("schematic.wave.pwlPasteHint")} aria-label={t("schematic.wave.pwlPasteHint")} onKeyDown={(e) => e.stopPropagation()} />
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="small muted">{t("schematic.wave.pwlPasteHint")}</span>
                <button
                  type="button"
                  className="btn sm primary"
                  onClick={() => {
                    const tt: number[] = [];
                    const vv: number[] = [];
                    for (const line of paste.split(/\n+/)) {
                      const parts = line.trim().split(/[\s,;\t]+/).filter(Boolean);
                      if (parts.length < 2) continue;
                      const a = parseSI(parts[0]);
                      const b = parseSI(parts[1]);
                      if (a == null || b == null) continue;
                      tt.push(a);
                      vv.push(b);
                    }
                    if (tt.length) onChange({ kind: "pwl", t: tt.slice(0, 2000), v: vv.slice(0, 2000) });
                    setPaste(null);
                  }}
                >
                  {t("schematic.wave.pwlApply")}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      <WavePreview wave={wave} tStop={tStop} unit={unit} />
    </div>
  );
}
