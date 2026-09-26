import { useState } from "react";
import { connectionError, normalizeEndpoint, useConnection, type ConnectionStatus } from "../api/connection";
import { Modal } from "../devices/Modal";
import { connectToServer } from "../state/runner";
import { useStore } from "../state/store";
import "./connection.css";

const STATUS: Record<ConnectionStatus, [string, string]> = {
  checking: ["연결 확인 중", "Connecting"], online: ["연결됨", "Connected"], offline: ["연결 안 됨", "Offline"],
  "auth-required": ["비밀번호 필요", "Password required"], snapshot: ["저장된 결과", "Recorded results"], demo: ["개발용 데모", "Developer demo"],
};

export function ConnectionButton() {
  const state = useConnection();
  const lang = useStore((s) => s.lang);
  const health = useStore((s) => s.health);
  const label = STATUS[state.status][lang === "ko" ? 0 : 1];
  return <>
    <button type="button" className={`connection-button ${state.status}`} onClick={state.open} aria-haspopup="dialog" aria-expanded={state.dialogOpen} title={label} data-testid="connection-button">
      <span className={`connection-dot ${state.status}`} aria-hidden="true" />
      <span>{lang === "ko" ? "연결" : "Connection"}</span>
      <span data-testid="backend-status" className="sr-only"><span className="status-text">{state.status === "online" ? `API · ${health?.workers ?? "?"}w` : state.status === "demo" ? "mock" : state.status}</span></span>
    </button>
    {state.dialogOpen && <ConnectionDialog />}
  </>;
}

export function ConnectionDialog() {
  const state = useConnection();
  const ko = useStore((s) => s.lang === "ko");
  const lang = ko ? "ko" : "en";
  const [mode, setMode] = useState<"local" | "lab">(state.endpoint ? "lab" : "local");
  const [url, setUrl] = useState(state.endpoint);
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async () => {
    if (busy) return;
    setError(null);
    if (mode === "lab" && !url.trim()) { setError("invalid-url"); return; }
    setBusy(true);
    try { await connectToServer(mode === "local" ? "" : url, password || undefined); }
    catch (e) { setError((e as Error).message); }
    finally { setPassword(""); setBusy(false); }
  };
  let edited = false;
  try { edited = (mode === "local" ? "" : normalizeEndpoint(url)) !== state.endpoint || (mode === "lab" && !url.trim()); }
  catch { edited = true; }
  const message = error ?? (edited ? null : state.error);
  const displayStatus = edited ? "offline" : state.status;
  return <Modal title={ko ? "계산 서버 연결" : "Compute connection"} onClose={state.close} width={430} testId="connection-dialog" footer={<>
    <a className="connection-help" href="./docs/setup.html" target="_blank" rel="noreferrer">{ko ? "설치 안내" : "Setup guide"} ↗</a>
    <span className="spacer" />
    <button type="button" className="btn sm primary" onClick={() => void submit()} disabled={busy} data-testid="connection-test">{busy ? (ko ? "확인 중…" : "Connecting…") : (ko ? "연결 확인" : "Connect")}</button>
  </>}>
    <form className="connection-form" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
      <div className="connection-modes" role="radiogroup" aria-label={ko ? "서버 위치" : "Server location"}>
        {(["local", "lab"] as const).map((item) => <button type="button" role="radio" aria-checked={mode === item} disabled={busy} key={item} className={mode === item ? "selected" : ""} data-testid={`connection-${item}`} onClick={() => { setMode(item); setError(null); }}>
          {item === "local" ? (ko ? "로컬" : "Local") : (ko ? "연구실 서버" : "Lab server")}
        </button>)}
      </div>
      {mode === "local" ? <p className="connection-local-note">{ko ? "이 페이지를 실행한 서버에서 계산합니다." : "Compute on the server that hosts this page."}</p> : <label className="connection-field">
        <span>{ko ? "서버 주소" : "Server URL"}</span>
        <input type="url" value={url} onChange={(e) => { setUrl(e.target.value); setError(null); }} placeholder="https://lab.example.edu/biristor" autoComplete="url" spellCheck={false} disabled={busy} data-testid="connection-url" />
      </label>}
      <label className="connection-field">
        <span>{ko ? "서버 비밀번호" : "Server password"} <small>{ko ? "설정된 경우" : "if configured"}</small></span>
        <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="off" disabled={busy} data-testid="connection-password" />
      </label>
      <div className={`connection-result ${displayStatus}`} role="status" data-testid="connection-status">
        <span className={`connection-dot ${displayStatus}`} aria-hidden="true" /><span>{edited ? (ko ? "연결 확인 전" : "Not connected yet") : STATUS[state.status][ko ? 0 : 1]}</span>
      </div>
      {message && <p className="connection-error" role="alert">{connectionError(message, lang)}</p>}
      <p className="connection-note">{ko ? "주소만 저장합니다. 비밀번호는 저장하지 않습니다." : "Only the address is saved. Your password is not stored."}</p>
      <button className="sr-only" type="submit" tabIndex={-1}>{ko ? "연결" : "Connect"}</button>
    </form>
  </Modal>;
}
