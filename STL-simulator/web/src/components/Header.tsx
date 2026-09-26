// App chrome: brand, navigation and connection controls above a persistent simulation-mode strip.
// The navigation takes a second row on smaller screens; the mode control remains easy to find.
import { useEffect, useRef, useState, useSyncExternalStore, type KeyboardEvent } from "react";
import type { Mode } from "../api/types";
import { useT, type T } from "../i18n";
import type { StrKey } from "../i18n/strings";
import type { Tab } from "../params/schema";
import { initBackend } from "../state/runner";
import { useStore } from "../state/store";
import { useSch } from "../schematic/store";
import { useForcing } from "../device/forcing";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { Credits } from "./Credits";
import { IconMenu, IconMoon, IconSun } from "./icons";
import { Logo } from "./Logo";
import { useSnapshotState } from "./SnapshotNotice";
import { ConnectionButton } from "./ConnectionDialog";

const TABS: { id: Tab; key: StrKey }[] = [
  { id: "device", key: "tab.device" },
  { id: "circuit", key: "tab.circuit" },
  { id: "validation", key: "tab.validation" },
  { id: "physics", key: "tab.physics" },
  { id: "performance", key: "tab.performance" },
];

/** Tabs whose results depend on the Deterministic | Stochastic mode (the toggle shows only there). */
export const hasMode = (tab: Tab) => tab === "device" || tab === "circuit";

/** Context-strip sentence: what the current tab shows in the selected mode, in plain words. */
export function modeHint(t: T, tab: Tab, mode: Mode, method: string): string {
  if (tab === "circuit") return t(mode === "deterministic" ? "mode.circuit.deterministic.hint" : "mode.circuit.stochastic.hint", { method });
  if (tab === "validation") return t("mode.validation.hint");
  if (tab === "physics") return t("mode.physics.hint");
  if (tab === "performance") return t.lang === "ko" ? "모델별 실행 시간과 계산 환경의 예상 시간" : "Model runtimes and compute-host estimates";
  return t(mode === "deterministic" ? "mode.deterministic.hint" : "mode.stochastic.hint");
}

/** The technical line behind the plain sentence (its tooltip). */
export function modeHintTech(t: T, tab: Tab, mode: Mode, method: string): string {
  if (tab === "circuit") return t(mode === "deterministic" ? "mode.circuit.deterministic.tech" : "mode.circuit.stochastic.tech", { method });
  if (tab === "validation") return t("mode.validation.tech");
  if (tab === "physics") return t("mode.physics.tech");
  if (tab === "performance") return t.lang === "ko" ? "결과 캐시·첫 준비·작업 대기를 분리한 계산 시간" : "Compute times with result cache, first-run setup, and queueing treated separately";
  return t(mode === "deterministic" ? "mode.deterministic.tech" : "mode.stochastic.tech");
}

// ---------------------------------------------------------------- two-row header breakpoint (same as app.css)
const PHONE = "(max-width: 960px)";
function subscribePhone(cb: () => void) {
  const mq = typeof window !== "undefined" ? window.matchMedia?.(PHONE) : undefined;
  mq?.addEventListener?.("change", cb);
  return () => mq?.removeEventListener?.("change", cb);
}
const isPhone = () => (typeof window !== "undefined" ? !!window.matchMedia?.(PHONE).matches : false);
/** True at ≤ 960 px (the mode toggle then lives in the context strip, never in both places). */
export const usePhone = () => useSyncExternalStore(subscribePhone, isPhone, () => false);

function arrowNav<V>(e: KeyboardEvent, items: V[], cur: V, set: (v: V) => void) {
  const i = items.indexOf(cur);
  let j = -1;
  if (e.key === "ArrowRight") j = (i + 1) % items.length;
  else if (e.key === "ArrowLeft") j = (i - 1 + items.length) % items.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = items.length - 1;
  if (j < 0) return;
  e.preventDefault();
  set(items[j]);
  const btns = (e.currentTarget as HTMLElement).querySelectorAll<HTMLButtonElement>("button");
  btns[j]?.focus();
}

/** A distinct trace icon keeps the active mode legible without relying on colour alone. */
function ModeTrace({ mode }: { mode: Mode }) {
  return (
    <svg className="mode-trace" viewBox="0 0 20 18" width="20" height="18" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {mode === "deterministic" ? <path d="M2 14 H7 V4 H18" /> : <>
        <path d="M2 14 H5 V6 H18 M2 12 H9 V3 H18" opacity=".4" strokeWidth="1.2" />
        <path d="M2 15 H7 V5 H18" />
      </>}
    </svg>
  );
}

/** [결정론적 | 확률적]: persistent mode selection with a plain tooltip explanation. */
export function ModeToggle({ compact }: { compact?: boolean }) {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const tab = useStore((s) => s.tab);
  const method = useStore((s) => s.params.circuit.solver.method);
  const simpleDevice = useStore((s) => s.params.device.model === "simple");
  const simpleCircuit = useSch((s) => s.doc.elements.some((e) => e.stl?.device.model === "simple"));
  const simple = tab === "circuit" ? simpleCircuit : simpleDevice;
  useEffect(() => { if (simple && mode === "stochastic") setMode("deterministic"); }, [simple, mode, setMode]);
  const modes: Mode[] = ["deterministic", "stochastic"];
  return (
    <div className={`mode-toggle${compact ? " compact" : ""}`} role="radiogroup" aria-label={t("mode.aria")} data-testid="mode-toggle" onKeyDown={(e) => arrowNav(e, simple ? ["deterministic"] : modes, mode, setMode)}>
      {modes.map((m) => (
        <button
          key={m}
          type="button"
          role="radio"
          aria-checked={mode === m}
          tabIndex={mode === m ? 0 : -1}
          className={`mode-btn ${m === "deterministic" ? "det" : "sto"}`}
          disabled={simple && m === "stochastic"}
          onClick={() => setMode(m)}
          title={simple && m === "stochastic" ? (t.lang === "ko" ? "Simple Model은 현재 결정론적 해석을 지원합니다." : "Simple Model currently supports deterministic analysis.") : modeHint(t, tab === "circuit" ? "circuit" : "device", m, method)}
          data-testid={`mode-${m}`}
        >
          <ModeTrace mode={m} />
          <span>{t(m === "deterministic" ? "mode.deterministic" : "mode.stochastic")}</span>
        </button>
      ))}
    </div>
  );
}

export function Header() {
  const t = useT();
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);
  const lang = useStore((s) => s.lang);
  const setLang = useStore((s) => s.setLang);
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  const sidebarOpen = useStore((s) => s.sidebarOpen);
  const setSidebar = useStore((s) => s.setSidebar);
  const hasSidebar = tab === "device" || tab === "circuit";
  return (
    <header className="header" role="banner">
      {hasSidebar && (
        <button type="button" className="icon-btn sidebar-btn" aria-label={t("sidebar.toggle")} aria-expanded={sidebarOpen} aria-controls="sidebar" onClick={() => setSidebar(!sidebarOpen)}>
          <IconMenu size={17} />
          <span className="sidebar-btn-label" aria-hidden>
            {t("sidebar.title")}
          </span>
        </button>
      )}
      <div className="brand" title={t("app.subtitle")}>
        <Logo size={35} className="brand-logo" />
        <span className="brand-title"><strong>STL</strong> simulator</span>
        <span className="tech-chip" title={t("brand.tech.title")} data-testid="tech-chip">
          {t("brand.tech")}
        </span>
      </div>
      <nav className="tabs" role="tablist" aria-label={t("tabs.aria")} onKeyDown={(e) => arrowNav(e, TABS.map((x) => x.id), tab, setTab)}>
        {TABS.map((x) => (
          <button key={x.id} type="button" role="tab" id={`tab-${x.id}`} aria-selected={tab === x.id} aria-controls="main" tabIndex={tab === x.id ? 0 : -1} className="tab" onClick={() => setTab(x.id)} data-testid={`tab-${x.id}`}>
            {x.id === "physics" ? (lang === "ko" ? "가이드" : "Guide") : t(x.key)}
          </button>
        ))}
      </nav>
      <span className="spacer" />
      <ConnectionButton />
      <button type="button" className="icon-btn lang-btn" onClick={() => setLang(lang === "ko" ? "en" : "ko")} aria-label={t("lang.toggle")} title={t("lang.toggle")} data-testid="lang-toggle">
        {lang === "ko" ? "EN" : "한"}
      </button>
      <button type="button" className="icon-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={theme === "dark" ? t("theme.toggle.light") : t("theme.toggle.dark")} title={theme === "dark" ? t("theme.toggle.light") : t("theme.toggle.dark")} data-testid="theme-toggle">
        {theme === "dark" ? <IconSun size={16} /> : <IconMoon size={16} />}
      </button>
    </header>
  );
}

/**
 * Backend status pill (right side of the context strip): shown only when the results on screen are not live
 * model results — demo data (mock), offline (with a retry button) or the static snapshot. Tap / click / Enter
 * opens a short explanation; Esc or a click outside closes it.
 */
function StatusPill() {
  const t = useT();
  const backend = useStore((s) => s.backend);
  const snap = useSnapshotState();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        setOpen(false);
        btn.current?.focus();
      }
    };
    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("pointerdown", onDown, true);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [open]);
  if (backend !== "mock" && backend !== "offline" && backend !== "snapshot") return null;
  const label = backend === "mock" ? t("pill.mock") : backend === "offline" ? t("pill.offline") : t("pill.snapshot");
  const short = backend === "mock" ? t("pill.mock.short") : backend === "offline" ? t("pill.offline.short") : t("pill.snapshot.short");
  const warn = backend === "snapshot" && snap.anyMiss;
  return (
    <div ref={wrap} className={`status-pill ${backend}${warn ? " warn" : ""}`} role="status" data-testid="offline-banner">
      <button
        ref={btn}
        type="button"
        className="status-pill-btn"
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={t("pill.more", { what: label })}
        title={backend === "mock" ? t("banner.mockForced") : backend === "offline" ? (t.lang === "ko" ? "계산 서버에 연결하면 시뮬레이션할 수 있습니다." : "Connect a compute server to simulate.") : (t.lang === "ko" ? "현재 조건과 일치하는 저장된 계산 결과입니다." : "Recorded computation matching the selected parameters.")}
        onClick={() => setOpen((o) => !o)}
      >
        <span className={`dot ${backend}`} aria-hidden />
        {/* phones show the short word */}
        <span data-testid={backend === "snapshot" ? "snapshot-banner" : undefined}>
          <span className="pill-long">{label}</span>
          <span className="pill-short" aria-hidden>
            {short}
          </span>
        </span>
      </button>
      {backend === "offline" && (
        <button type="button" className="status-pill-retry" onClick={() => void initBackend()}>
          {t("banner.retry")}
        </button>
      )}
      {open && (
        <div className="status-pop" role="dialog" aria-label={label}>
          <strong>{label}</strong>
          {backend === "snapshot" ? <p>{t.lang === "ko" ? "저장된 계산 결과입니다. 파라미터를 바꿔 계산하려면 서버에 연결하세요." : "These are recorded computations. Connect a server to calculate other parameter sets."}</p> : <p>{backend === "mock" ? t("banner.mockForced") : t.lang === "ko" ? "파라미터와 회로를 편집할 수 있습니다. 계산하려면 상단의 연결 버튼에서 로컬 또는 연구실 서버에 연결하세요." : "You can edit parameters and circuits. Use Connect above to select a local or laboratory compute server."}</p>}
        </div>
      )}
    </div>
  );
}

/** Context strip (`modebar`): mode control · quantity hint · connection status · credits. */
export function ContextStrip() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const tab = useStore((s) => s.tab);
  const method = useStore((s) => s.params.circuit.solver.method);
  const forcing = useForcing((s) => s.forcing);
  const csvm = tab === "device" && forcing === "csvm";
  const hint = csvm ? (t.lang === "ko" ? "전류 구동 · 드레인 전압 파형" : "Current forcing · drain-voltage transient") : modeHint(t, tab, mode, method);
  return (
    <div className={`modestrip${hasMode(tab) ? "" : " no-mode"}`} data-testid="modebar">
      {hasMode(tab) && <ModeToggle compact />}
      <span className="hint" data-testid="mode-hint" title={csvm ? hint : `${hint} · ${modeHintTech(t, tab, mode, method)}`}>
        <SubText text={subs(tab === "device" ? (csvm ? "FDSOI · VD(t)" : "FDSOI · ID–VD") : tab === "circuit" ? t("tab.circuit") : t(tab === "validation" ? "tab.validation" : tab === "performance" ? "tab.performance" : "tab.physics"))} />
      </span>
      <StatusPill />
      <Credits />
    </div>
  );
}
