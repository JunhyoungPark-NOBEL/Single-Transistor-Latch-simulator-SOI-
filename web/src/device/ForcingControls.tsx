import { useEffect, useState } from "react";
import { useT } from "../i18n";
import { SubText } from "../plots/SubText";
import { Seg } from "./DetPanels";
import { CSVM_LIMITS, useForcing, type CsvmSettings } from "./forcing";

function Setting({ name, symbol, scale, unit, title }: { name: keyof CsvmSettings; symbol: string; scale: number; unit: string; title: string }) {
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
    <span><SubText text={symbol} /></span>
    <input aria-label={symbol} data-testid={`csvm-${name}`} type="number" step="any" min={lo * scale} max={hi * scale} value={draft}
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
      <Setting name="current_A" symbol="Iin" scale={1e9} unit="nA" title={ko ? "드레인에 공급하는 DC 전류" : "DC current supplied to drain"} />
      <Setting name="capacitance_F" symbol="Cdrain" scale={1e12} unit="pF" title={ko ? "드레인–접지 커패시턴스" : "Drain-to-ground capacitance"} />
      <Setting name="duration_s" symbol="Time" scale={1e3} unit="ms" title={ko ? "과도 해석 시간" : "Transient duration"} />
    </div>}
    <a className="forcing-help" aria-label={ko ? "구동 모드 안내" : "Forcing mode guide"} title={ko ? "구동 모드 안내" : "Forcing mode guide"} href={`${import.meta.env.BASE_URL}docs/device-forcing.md`} target="_blank" rel="noreferrer">ⓘ</a>
  </div>;
}
