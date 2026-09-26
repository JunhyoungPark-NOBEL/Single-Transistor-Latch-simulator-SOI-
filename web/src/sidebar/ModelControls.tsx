import { useEffect, useRef, useState } from "react";
import { JobAborted, runJob } from "../api/client";
import { geometryError } from "../api/geometryPolicy";
import type { DeviceModel, SimpleCalibrationResult, SimpleFit, SimpleModelBlock } from "../api/types";
import { GuidePopover } from "../components/GuidePopover";
import { Tex } from "../components/Tex";
import { Modal } from "../devices/Modal";
import { useT } from "../i18n";
import { modelOf, parseHrsPoints, SIMPLE_DEFAULTS, simpleOf } from "../params/model";
import type { Ctx, FieldDef } from "../params/schema";
import { getBackend } from "../state/runner";
import { useStore } from "../state/store";
import { clone, deepEqual } from "../utils/object";
import { Field } from "./Field";
import "./model-controls.css";

const L = (ko: string, en: string) => ({ ko, en });
const DOC = "docs/simple-model.html";
const FIELDS: (FieldDef & { parameter: keyof SimpleModelBlock })[] = [
  { parameter: "beta_ref", sym: "\\beta_{ref}", label: L("확산 전류비", "Diffusion ratio"), help: L("I_DIFF = I_seed/β. 같은 형상에서 β는 바이어스 상수입니다. β = βref·(500 nm/L)·(Nref/Nbody)로 환산합니다.", "I_DIFF = I_seed/β. At fixed geometry, β is bias-independent. It scales as βref·(500 nm/L)·(Nref/Nbody)."), min: 0.01, max: 1e4, unit: "", main: true },
  { parameter: "tau_body_s", sym: "\\tau_B", label: L("바디 수명", "Body lifetime"), help: L("재결합을 QB/τB의 1차 손실로 근사합니다. 기준 수명이며 Tsi와 표면 재결합 비율에 따라 환산됩니다.", "First-order recombination QB/τB. This reference lifetime scales with Tsi and the surface fraction."), min: 0.001, max: 1e10, scale: 1e9, unit: "ns", main: true },
  { parameter: "r_lrs_ref_ohm", sym: "R_{LRS}", label: L("바디 저항", "Body resistance"), help: L("고주입 상태의 유효 바디 전압 강하 ID·RLRS. 기준 형상의 저항입니다.", "Effective high-injection body drop ID·RLRS, specified at the reference geometry."), min: 0, max: 1e9, scale: 1e-3, unit: "kΩ", main: true },
  { parameter: "is_ref_A", sym: "I_S", label: L("주입 전류", "Injection current"), help: L("BJT 주입 전류의 기준 계수입니다. HRS 데이터로 보정할 수 있습니다.", "Reference BJT injection coefficient; fit it to HRS data."), min: 1e-20, max: 1e12, scale: 1e15, unit: "fA", main: true },
  { parameter: "cb_ref_F", sym: "C_B", label: L("바디 용량", "Body capacitance"), help: L("QB = CB·VB,curr의 기준 용량입니다. 동역학과 유효 재결합 저항 τB/CB를 정합니다.", "Reference capacitance in QB = CB·VB,curr; sets dynamics and the effective loss resistance τB/CB."), min: 1e-5, max: 1e9, scale: 1e15, unit: "fF" },
  { parameter: "vbr_ref_V", sym: "V_{BR}", label: L("증배 전압", "Multiplication voltage"), help: L("충돌 이온화 증배의 기준 전압 척도입니다.", "Reference voltage scale for avalanche multiplication."), min: 0.1, max: 100, unit: "V" },
  { parameter: "avalanche_eta", sym: "\\eta", label: L("증배 지수", "Multiplication exponent"), help: L("충돌 이온화 증배의 전압 의존성을 정합니다.", "Controls the voltage dependence of avalanche multiplication."), min: 1, max: 12, unit: "" },
  { parameter: "gamma_fg", sym: "\\gamma_{FG}", label: L("게이트 결합", "Gate coupling"), help: L("전면 게이트가 유효 바디 바이어스에 주는 결합 계수입니다.", "Front-gate coupling to effective body bias."), min: 0, max: 1, unit: "" },
  { parameter: "gamma_bg", sym: "\\gamma_{BG}", label: L("백게이트 결합", "Back-gate coupling"), help: L("백게이트가 유효 바디 바이어스에 주는 기준 결합 계수입니다.", "Reference back-gate coupling to effective body bias."), min: 0, max: 1, unit: "" },
  { parameter: "vfb_V", sym: "V_{FB}", label: L("평탄대 전압", "Flat-band voltage"), help: L("게이트 결합의 전압 기준입니다.", "Voltage reference for front-gate coupling."), min: -10, max: 10, unit: "V" },
  { parameter: "btbt_scale", sym: "s_{BTBT}", label: L("BTBT 배율", "BTBT scale"), help: L("터널링 정공 생성 전류의 배율입니다.", "Scale of tunneling-generated hole current."), min: 0, max: 1e6, unit: "" },
  { parameter: "surface_fraction", sym: "f_{surf}", label: L("표면 손실 비율", "Surface loss fraction"), help: L("기준 재결합 중 표면 손실의 가정 비율입니다. 독립적으로 추출된 물성은 아닙니다.", "Assumed surface contribution to reference recombination; not an independently extracted material property."), min: 0, max: 1, step: 0.01, unit: "" },
  { parameter: "gidl_volume_scale", sym: "s_{GIDL}", label: L("GIDL 체적 배율", "GIDL volume factor"), help: L("GIDL 생성 체적 W·L_ov(5 nm)·W_t에 곱하는 배율입니다. 논문 참고 스크립트의 값 100이 기본이며, 1이면 물리적 체적만 사용합니다.", "Multiplies the GIDL generation volume W·L_ov(5 nm)·W_t. The paper's reference script uses 100 (default); 1 keeps only the physical volume."), min: 0, max: 1e6, unit: "" },
].map((f) => ({ ...f, parameter: f.parameter as keyof SimpleModelBlock, key: `simple-${f.parameter}`, path: ["device", "simple", f.parameter], documentationPath: DOC }));

function CalibrationDialog({ onClose }: { onClose: () => void }) {
  const t = useT();
  const L = (ko: string, en: string) => t.lang === "ko" ? ko : en;
  const [source] = useState(() => clone(useStore.getState().params.device));
  const [text, setText] = useState("");
  const [fit, setFit] = useState<SimpleFit>("is");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<SimpleCalibrationResult | null>(null);
  const [error, setError] = useState("");
  const token = useRef(0);
  const online = useStore((s) => s.backend === "online");
  useEffect(() => () => { token.current++; }, []);
  let points: ReturnType<typeof parseHrsPoints> = [];
  let parseError = "";
  try { points = parseHrsPoints(text); } catch (e) {
    parseError = (e as Error).message === "limit" ? L("최대 200점을 입력해 주세요.", "Enter at most 200 points.") : L("각 행에 양수 VD(V), ID(A)를 입력해 주세요.", "Each row needs positive VD (V), ID (A).");
  }
  const enough = points.length >= (fit === "is_tau" ? 3 : 1);
  const calculate = async () => {
    if (!online || !enough || parseError || busy) return;
    const run = ++token.current;
    setBusy(true); setError(""); setResult(null);
    try {
      const response = await runJob<SimpleCalibrationResult>(getBackend(), "simple_calibrate", { device: source, points, fit }, { isAborted: () => token.current !== run });
      if (run === token.current) setResult(response);
    } catch (e) {
      if (run === token.current && !(e instanceof JobAborted)) setError(geometryError((e as Error).message, t.lang));
    } finally { if (run === token.current) setBusy(false); }
  };
  const apply = () => {
    if (!result?.identifiable || modelOf(result.device) !== "simple") return;
    const current = useStore.getState();
    if (!deepEqual(current.params.device, source)) { setError(L("소자 설정이 변경되었습니다. 닫고 다시 보정해 주세요.", "Device settings changed. Close and reopen calibration.")); return; }
    current.updateParams((p) => ({ ...p, device: clone(result.device) }));
    onClose();
  };
  return <Modal title={L("HRS 보정", "HRS calibration")} onClose={onClose} width={520} testId="simple-calibration-dialog" footer={<>
    <button className="btn sm ghost" onClick={onClose}>{L("닫기", "Close")}</button><span className="spacer" />
    <button className="btn sm" disabled={!online || !enough || !!parseError || busy} onClick={() => void calculate()} data-testid="simple-calibrate-run">{busy ? L("보정 중…", "Fitting…") : L("계산", "Fit")}</button>
    <button className="btn sm primary" disabled={!result?.identifiable || busy} onClick={apply} data-testid="simple-calibrate-apply">{L("적용", "Apply")}</button>
  </>}>
    <p className="simple-calibration-intro">{L("현재 Geometry·바이어스의 HRS 점을 붙여넣으세요. 래치 이후 점은 제외해 주세요.", "Paste HRS points at the current geometry and bias. Exclude post-latch points.")}</p>
    <label className="simple-calibration-label" htmlFor="simple-hrs-points"><Tex tex="V_D\;(\mathrm{V}),\ I_D\;(\mathrm{A})" /><span>CSV / TSV</span></label>
    <textarea id="simple-hrs-points" className="input text simple-calibration-data" rows={6} spellCheck={false} value={text} placeholder={'VD_V, ID_A\n'} onChange={(e) => { setText(e.target.value); setResult(null); setError(""); }} disabled={busy} data-testid="simple-hrs-points" />
    <div className="simple-calibration-fit"><span>{L("보정 항목", "Fit parameters")}</span><select className="select" aria-label={L("보정 항목", "Fit parameters")} value={fit} disabled={busy} onChange={(e) => {setFit(e.target.value as SimpleFit); setResult(null);}} data-testid="simple-fit-kind"><option value="is">IS</option><option value="tau">τB</option><option value="is_tau">IS + τB</option></select></div>
    <p className="model-note">{points.length} {L("점", "points")}{fit === "is_tau" ? L(" · 동시 보정은 3점 이상", " · joint fit needs 3+ points") : ""} · {L("β 고정", "fixed β")}</p>
    {parseError && <p className="model-error" role="alert">{parseError}</p>}
    {!online && <p className="model-note">{L("실시간 계산 서버 연결이 필요합니다.", "A live compute server is required.")}</p>}
    {error && <p className="model-error" role="alert">{error}</p>}
    {result && <div className="simple-calibration-result" role="status" data-testid="simple-calibration-result">
      <div><span><Tex tex="I_S" /></span><strong>{result.device.simple?.is_ref_A.toExponential(4)} A</strong></div>
      <div><span><Tex tex="\tau_B" /></span><strong>{((result.device.simple?.tau_body_s ?? 0) * 1e9).toPrecision(5)} ns</strong></div>
      <div><span>RMSE · log₁₀</span><strong>{Number.isFinite(result.rmse_log10) ? result.rmse_log10.toPrecision(4) : "—"}</strong></div>
      {!result.identifiable && <p className="model-error">{L("이 데이터만으로 보정값을 구분할 수 없습니다. 한 항목만 선택하거나 점을 추가해 주세요.", "These data cannot identify the selected parameters. Fit one parameter or add points.")}</p>}
      {!!result.warnings?.length && <details><summary>{L("보정 범위", "Fit scope")}</summary>{result.warnings.map((warning, i) => <p key={i}>{warning}</p>)}</details>}
    </div>}
  </Modal>;
}

export function ModelControls() {
  const t = useT();
  const params = useStore((s) => s.params);
  const mode = useStore((s) => s.mode);
  const tab = useStore((s) => s.tab);
  const [calibration, setCalibration] = useState(false);
  const [startBias, setStartBias] = useState(false);
  const initialized = useRef(false);
  if (!initialized.current) { try { initialized.current = localStorage.getItem("stl-simple-model-started:v1") === "1"; } catch { /* optional persistence */ } }
  const model = modelOf(params.device);
  const values = simpleOf(params.device.simple);
  const ctx: Ctx = { root: params, mode, tab };
  const choose = (next: DeviceModel) => {
    const initializeBias = next === "simple" && !initialized.current && params.device.vg === -2;
    if (next === "simple") {
      initialized.current = true;
      try { localStorage.setItem("stl-simple-model-started:v1", "1"); } catch { /* optional persistence */ }
      setStartBias(initializeBias);
    }
    useStore.setState((s) => ({ mode: next === "simple" ? "deterministic" : s.mode,
      params: { ...s.params, device: { ...s.params.device, ...(initializeBias ? { vg: -3 } : {}), model: next, simple: simpleOf(s.params.device.simple) } } }));
  };
  const field = (f: typeof FIELDS[number]) => <Field key={f.key} f={f} ctx={ctx} value={values[f.parameter]} def={SIMPLE_DEFAULTS[f.parameter]} slider={false} onChange={(v) => useStore.getState().updateParams((p) => ({ ...p, device: { ...p.device, simple: { ...simpleOf(p.device.simple), [f.parameter]: v } } }))} />;
  return <section className="model-controls" data-testid="model-controls" aria-label="Model">
    <div className="model-head"><h2>Model</h2><GuidePopover id="model" testId="tip-model" label="Model" ariaLabel={t.lang === "ko" ? "모델 가이드" : "Model guide"} description={t.lang === "ko" ? "Detailed는 SRH 수송 해석, Simple은 바디 전하의 1차 재결합 근사입니다. 두 모델의 보정값은 별도로 저장됩니다." : "Detailed solves SRH transport; Simple uses first-order body-charge decay. Their calibrations are stored separately."} onMore={() => window.open(`${import.meta.env.BASE_URL}${DOC}`, "_blank", "noopener,noreferrer")} /></div>
    <div className="seg full model-switch" role="radiogroup" aria-label="Model">{(["detailed", "simple"] as const).map((m) => <button type="button" role="radio" aria-checked={model === m} key={m} onClick={() => choose(m)} data-testid={`model-${m}`}>{m === "simple" ? "Simple Model" : "Detailed Model"}</button>)}</div>
    {model === "simple" && <>
      <div className="model-status"><span>{deepEqual(values, SIMPLE_DEFAULTS) ? (t.lang === "ko" ? "초기값 · HRS 보정 필요" : "Initial values · fit HRS") : (t.lang === "ko" ? "사용자값 · 결정론적" : "Custom · deterministic")}</span><button type="button" className="link-btn" onClick={() => setCalibration(true)} data-testid="simple-calibrate-open">{t.lang === "ko" ? "HRS 보정" : "Fit HRS"}</button></div>
      {startBias && <p className="model-note">{t.lang === "ko" ? "시작 바이어스 " : "Starting bias "}<Tex tex="V_G = -3\,\mathrm{V}" /></p>}
      <div className="simple-main-fields">{FIELDS.filter((f) => f.main).map(field)}</div>
      <details className="simple-advanced"><summary>{t.lang === "ko" ? "세부 파라미터" : "More parameters"}<span>8</span></summary>{FIELDS.filter((f) => !f.main).map(field)}</details>
    </>}
    {calibration && <CalibrationDialog onClose={() => setCalibration(false)} />}
  </section>;
}
