import { create } from "zustand";

export const CONNECTION_STORAGE_KEY = "biristor-studio:connection";
export type ConnectionStatus = "checking" | "online" | "offline" | "auth-required" | "snapshot" | "demo";

/** A server root, not an API route. Empty means the server hosting this page. */
export function normalizeEndpoint(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  let url: URL;
  try { url = new URL(trimmed); } catch { throw new Error("invalid-url"); }
  if (!/^https?:$/.test(url.protocol) || url.username || url.password || url.search || url.hash || /[\\\s]/.test(trimmed)) throw new Error("invalid-url");
  return url.origin + url.pathname.replace(/\/+$/, "");
}

export function readEndpoint(): string {
  try {
    const saved = JSON.parse(localStorage.getItem(CONNECTION_STORAGE_KEY) ?? "null") as { version?: number; endpoint?: unknown } | null;
    return saved?.version === 1 && typeof saved.endpoint === "string" ? normalizeEndpoint(saved.endpoint) : "";
  } catch { return ""; }
}

export function endpointOrigin(endpoint: string): string {
  return endpoint ? new URL(endpoint).origin : typeof location !== "undefined" ? location.origin : "";
}

export function isRemoteEndpoint(endpoint: string): boolean {
  return !!endpoint && (typeof location === "undefined" || endpointOrigin(endpoint) !== location.origin);
}

interface ConnectionState {
  endpoint: string;
  status: ConnectionStatus;
  error: string | null;
  dialogOpen: boolean;
  open: () => void;
  close: () => void;
}

/** Only the endpoint is persisted. Session credentials never enter browser storage. */
export const useConnection = create<ConnectionState>((set) => ({
  endpoint: readEndpoint(), status: "checking", error: null, dialogOpen: false,
  open: () => set({ dialogOpen: true }), close: () => set({ dialogOpen: false }),
}));

let sessionToken: string | null = null;
export function getSessionToken(): string | null { return sessionToken; }
export function setSessionToken(token: string | null): void { sessionToken = token; }

export function selectEndpoint(value: string): string {
  const endpoint = normalizeEndpoint(value);
  if (endpoint !== useConnection.getState().endpoint) sessionToken = null;
  useConnection.setState({ endpoint, error: null });
  try { localStorage.setItem(CONNECTION_STORAGE_KEY, JSON.stringify({ version: 1, endpoint })); } catch { /* private mode */ }
  return endpoint;
}

export function connectionError(message: string, lang: "ko" | "en"): string {
  if (message === "invalid-url") return lang === "ko" ? "http 또는 https 서버 주소를 입력해 주세요. 계정·쿼리·#은 넣지 않습니다." : "Enter an HTTP(S) server URL without credentials, a query or a fragment.";
  if (message === "connection-required" || message === "snapshot-missing") return lang === "ko" ? "이 조건은 계산 서버 연결이 필요합니다. 상단의 연결을 눌러 주세요." : "This condition needs a compute server. Open Connection above.";
  if (message === "authentication-required" || /login required/i.test(message)) return lang === "ko" ? "서버 비밀번호를 입력해 주세요." : "Enter the server password.";
  if (message === "incorrect password") return lang === "ko" ? "비밀번호가 올바르지 않습니다." : "Incorrect password.";
  if (message === "browser origin is not allowed") return lang === "ko" ? "서버의 허용 주소에 이 페이지 주소를 추가해 주세요." : "Add this page's origin to the server's allowed origins.";
  if (message === "server-misconfigured") return lang === "ko" ? "서버의 접근 설정을 확인해야 합니다." : "The server access configuration needs attention.";
  if (message === "request timed out" || /Failed to fetch|fetch failed|NetworkError|network error|Load failed/i.test(message)) return lang === "ko" ? "서버에 닿지 않습니다. 주소와 서버 실행 상태를 확인해 주세요." : "Cannot reach the server. Check the address and whether it is running.";
  return message;
}
