import { useEffect, useRef, useState } from "react";
import { JobAborted, runJob } from "../api/client";
import type { BranchesResult } from "../api/types";
import { useT } from "../i18n";
import { getBackend } from "../state/runner";
import { useStore } from "../state/store";
import { downloadText } from "../utils/csv";
import { clone } from "../utils/object";
import { Modal } from "./Modal";
import { buildLtspiceExport, buildVerilogAExport, ModelExportError, type ExportFormat, type ExportSelection, type LtspiceExport, type VerilogAExport } from "./modelExport";
import "./modelExport.css";

const COPY = {
  button: { ko: "상용 시뮬레이터로 내보내기", en: "Export to simulator" },
  title: { ko: "상용 시뮬레이터로 내보내기", en: "Export to simulator" },
  choose: { ko: "내보내기 형식", en: "Export format" },
  scope: { ko: "고정 조건의 ID–VD 모델", en: "Fixed-condition ID–VD model" },
  detail: { ko: "정적 히스테리시스용입니다. CSVM의 Vtop·Vbottom·주파수는 포함하지 않습니다.", en: "Static hysteresis only. CSVM Vtop, Vbottom and frequency are not included." },
  limits: { ko: "지원 범위", en: "Model scope" },
  limitsDetail: { ko: "게이트·광조건을 고정한 준정적 모델입니다. 바디 전하의 시간 변화와 잡음은 제외됩니다. 상용 프로그램에서의 실행 검증은 아직 수행하지 않았습니다.", en: "Quasi-static model at fixed gate and light. Body-charge dynamics and noise are excluded. Execution in commercial simulators has not yet been validated." },
  ltspice: { ko: "모델 + 실행 예제 · .cir", en: "Model + example · .cir" },
  verilog: { ko: "Verilog-A 모델 · .va", en: "Verilog-A model · .va" },
  sentaurus: { ko: "구조 정보 필요", en: "Structure required" },
  sentaurusTitle: { ko: "Sentaurus 내보내기는 아직 지원하지 않습니다.", en: "Sentaurus export is not yet supported." },
  sentaurusDetail: { ko: "물리 TCAD에는 격자·도핑·접촉 정보가 필요합니다. 현재 보정값만으로 실행 가능한 소자를 만들 수 없습니다.", en: "Physical TCAD requires mesh, doping and contact data. The current calibration alone cannot produce an executable device." },
  sentaurusMore: { ko: "Sentaurus의 compact model 연동도 별도 인터페이스 구현·검증이 필요합니다. 이 Verilog-A 파일을 Sentaurus에 직접 불러오는 방식은 검증되지 않았습니다.", en: "Sentaurus compact-model integration also needs a separate interface and validation. Direct loading of this Verilog-A source into Sentaurus is not verified." },
  downloadCir: { ko: ".cir 다운로드", en: "Download .cir" },
  downloadVa: { ko: ".va 다운로드", en: "Download .va" },
  unavailable: { ko: "내보내기 미지원", en: "Not available" },
  preparing: { ko: "모델 준비 중…", en: "Preparing model…" },
  guide: { ko: "사용 안내", en: "Usage guide" },
  library: { ko: ".lib 다운로드", en: "Download .lib" },
  live: { ko: "정확한 보정값을 내보내려면 계산 서버에 연결해 주세요.", en: "Connect to the compute server to export the exact calibration." },
  invalid: { ko: "이 보정 결과를 내보낼 수 없습니다. 파라미터를 확인해 주세요.", en: "This calibration cannot be exported. Check the parameters." },
  range: { ko: "VD 최대값을 래치업 전압보다 높게 설정하고, 유효한 ID–VD 범위에서 다시 시도해 주세요.", en: "Set the VD maximum above latch-up and within the valid ID–VD range, then try again." },
  noData: { ko: "이 범위에 계산되지 않은 ID–VD 구간이 있습니다. 스윕 범위를 줄여 주세요.", en: "Some ID–VD values are unavailable in this range. Reduce the sweep range." },
  error: { ko: "내보내기에 실패했습니다. 계산 서버 연결을 확인해 주세요.", en: "Export failed. Check the compute server connection." },
  simple: { ko: "Simple Model의 상용 시뮬레이터 내보내기는 아직 지원하지 않습니다. 소자 JSON 저장은 가능합니다.", en: "Commercial export of Simple Model is not yet supported. Device JSON remains available." },
  ready: { ko: "준정적 모델을 내려받았습니다.", en: "Quasi-static model downloaded." },
  close: { ko: "닫기", en: "Close" },
};
type PreparedExport = { format: "ltspice"; data: LtspiceExport } | { format: "verilog-a"; data: VerilogAExport };
const FORMATS = [
  { value: "ltspice", label: "LTspice", detail: COPY.ltspice },
  { value: "verilog-a", label: "Verilog-A", detail: COPY.verilog },
  { value: "sentaurus", label: "Sentaurus TCAD", detail: COPY.sentaurus },
] as const;

/** Export captures these props when opened, so edits during generation cannot change its provenance. */
export function ModelExportButton(props: ExportSelection) {
  const t = useT();
  const online = useStore((s) => s.backend === "online");
  const [selection, setSelection] = useState<ExportSelection | null>(null);
  const [format, setFormat] = useState<ExportFormat>("ltspice");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<keyof typeof COPY | null>(null);
  const [output, setOutput] = useState<PreparedExport | null>(null);
  const generation = useRef(0);
  useEffect(() => () => { generation.current++; }, []);
  const close = () => { generation.current++; setSelection(null); setBusy(false); };
  const generate = async () => {
    if (!selection || busy || format === "sentaurus" || selection.device.model === "simple") return;
    if (useStore.getState().backend !== "online" || getBackend().isMock) { setError("live"); return; }
    const token = ++generation.current;
    setBusy(true);
    setError(null);
    try {
      const result = await runJob<BranchesResult>(getBackend(), "branches", {
        device: selection.device, sweep: selection.sweep,
      }, { isAborted: () => generation.current !== token });
      if (generation.current !== token) return;
      if (format === "verilog-a") {
        const exported = buildVerilogAExport(selection, result);
        setOutput({ format, data: exported });
        downloadText(exported.filename, exported.source, "text/plain;charset=utf-8");
      } else {
        const exported = buildLtspiceExport(selection, result);
        setOutput({ format, data: exported });
        downloadText(exported.filename, exported.circuit, "text/plain;charset=utf-8");
      }
    } catch (e) {
      if (generation.current === token && !(e instanceof JobAborted)) setError(e instanceof ModelExportError ? e.code : "error");
    } finally {
      if (generation.current === token) setBusy(false);
    }
  };
  return <>
    <button type="button" className="btn sm ghost" data-testid="model-export-open" onClick={() => {
      setSelection(clone(props)); setFormat("ltspice"); setError(null); setOutput(null);
    }}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 3v12m-4-4 4 4 4-4M5 16v4h14v-4" />
      </svg>
      {t.l(COPY.button)}
    </button>
    {selection && <Modal title={t.l(COPY.title)} onClose={close} width={470} testId="model-export-dialog" footer={<>
      <button type="button" className="btn sm ghost" onClick={close}>{t.l(COPY.close)}</button>
      <span className="spacer" />
      <button type="button" className="btn sm primary" onClick={() => void generate()} disabled={busy || !online || format === "sentaurus" || selection.device.model === "simple"} data-testid="model-export-download">
        {t.l(busy ? COPY.preparing : format === "sentaurus" ? COPY.unavailable : format === "verilog-a" ? COPY.downloadVa : COPY.downloadCir)}
      </button>
    </>}>
      <fieldset className="model-export-formats" disabled={busy}>
        <legend className="sr-only">{t.l(COPY.choose)}</legend>
        {FORMATS.map((item) => <label key={item.value} className={`model-export-format${format === item.value ? " selected" : ""}`}>
          <input type="radio" name="model-export-format" value={item.value} checked={format === item.value} onChange={() => {
            setFormat(item.value); setError(null); setOutput(null);
          }} data-testid={`model-export-${item.value}`} />
          <span><strong>{item.label}</strong><small>{t.l(item.detail)}</small></span>
        </label>)}
      </fieldset>
      {selection.device.model === "simple" ? <p className="model-export-note" role="status" data-testid="model-export-simple-info">{t.l(COPY.simple)}</p> : format === "sentaurus" ? <div className="model-export-scope" data-testid="model-export-sentaurus-info">
        <strong>{t.l(COPY.sentaurusTitle)}</strong>
        <p>{t.l(COPY.sentaurusDetail)}</p>
        <details><summary>{t.l(COPY.limits)}</summary><p>{t.l(COPY.sentaurusMore)}</p></details>
      </div> : <>
        <div className="model-export-scope">
          <strong>{t.l(COPY.scope)} <span>· V<sub>G</sub> {selection.device.vg} V</span></strong>
          <p>{t.l(COPY.detail)}</p>
          <details><summary>{t.l(COPY.limits)}</summary><p>{t.l(COPY.limitsDetail)}</p></details>
        </div>
        {!online && <p className="model-export-note" role="status">{t.l(COPY.live)}</p>}
        {error && <p className="model-export-error" role="alert">{t.l(COPY[error])}</p>}
        {output && <div className="model-export-result">
          <p role="status">{t.l(COPY.ready)}</p>
          <div className="model-export-links">
            {output.format === "ltspice" && <button type="button" className="link-btn" onClick={() => downloadText(`${output.data.modelName}.lib`, output.data.subcircuit, "text/plain;charset=utf-8")}>{t.l(COPY.library)}</button>}
            <button type="button" className="link-btn" onClick={() => downloadText(`${output.data.modelName}-README.md`, output.data.readme, "text/markdown;charset=utf-8")}>{t.l(COPY.guide)}</button>
          </div>
        </div>}
      </>}
    </Modal>}
  </>;
}
