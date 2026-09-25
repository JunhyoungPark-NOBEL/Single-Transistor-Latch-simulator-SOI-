import { useEffect, useState, type ReactNode } from "react";
import { useT } from "../i18n";
import { Seg } from "./DetPanels";
import { CSVM_LIMITS, useForcing, type CsvmSettings } from "./forcing";

function Setting({ name, symbol, label, scale, unit, title }: { name: keyof CsvmSettings; symbol: ReactNode; label: string; scale: number; unit: string; title: string }) {
  const value = useForcing((s) => s.settings[name]);
  const set = useForcing((s) => s.setSetting);
  const display = (v: number) => String(Number((v * scale).toPrecision(8)));
  const [draft, setDraft] = useState(display(value));
  useEffect(() => setDraft(display(value)), [value]);
  const [lo, hi] = CSVM_LIMITS[name];
  const commit = () => {
    const next = Number(draft) / scale;
    if (draft.trim() && Number.isFinite(next) && next >= lo && next <= hi) set(name, next);
    else setDraft(display(value));
  };
  return <label className="forcing-field" title={title}>
    <span>{symbol}</span>
    <input aria-label={`${label} (${unit})`} data-testid={`csvm-${name}`} type="number" step="any" min={lo * scale} max={hi * scale} value={draft}
      onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
    <span className="forcing-unit">{unit}</span>
  </label>;
}

export function ForcingControls() {
  const t = useT();
  const forcing = useForcing((s) => s.forcing);
  const set = useForcing((s) => s.setForcing);
  const ko = t.lang === "ko";
  return <div className="forcing-bar" data-testid="device-forcing">
    <Seg value={forcing} onChange={set} label={ko ? "구동 모드" : "Forcing mode"} options={[
      { v: "vscm", label: `${ko ? "전압" : "Voltage"} · VSCM` },
      { v: "csvm", label: `${ko ? "전류" : "Current"} · CSVM` },
    ]} />
    {forcing === "csvm" && <div className="forcing-fields">
      <Setting name="current_A" symbol={<>I<sub>in</sub></>} label={ko ? "입력 전류" : "Input current"} scale={1e9} unit="nA" title={ko ? "드레인에 넣는 DC 전류" : "DC current fed into the drain"} />
      <Setting name="capacitance_F" symbol={<>C<sub>drain</sub></>} label={ko ? "드레인 커패시턴스" : "Drain capacitance"} scale={1e12} unit="pF" title={ko ? "드레인–접지 커패시턴스" : "Drain-to-ground capacitance"} />
      <Setting name="duration_s" symbol={ko ? "시간" : "Time"} label={ko ? "계산 시간" : "Simulated time"} scale={1e3} unit="ms" title={ko ? "과도해석 시간" : "Transient duration"} />
    </div>}
    <a className="forcing-help" aria-label={ko ? "구동 모드 안내" : "Forcing mode guide"} title={ko ? "구동 모드 안내" : "Forcing mode guide"} href={`${import.meta.env.BASE_URL}docs/device-forcing.html`} target="_blank" rel="noreferrer">ⓘ</a>
  </div>;
}
