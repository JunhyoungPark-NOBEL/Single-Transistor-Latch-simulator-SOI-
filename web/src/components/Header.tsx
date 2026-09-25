// App chrome: the 52 px header (brand, single-language tabs, one-word mode toggle on the Device and Circuit
// tabs, backend status dot, KO/EN, theme) and the 32 px context strip under it (one plain sentence about the
// current tab and mode, the backend status pill that replaces the old Demo / Offline / Snapshot banners, and
// the credits corner). On phones (≤ 760 px) the tabs take a second header row and the mode toggle moves into
// the context strip.
import { useEffect, useRef, useState, useSyncExternalStore, type KeyboardEvent } from "react";
import type { Mode } from "../api/types";
import { useT, type T } from "../i18n";
import type { StrKey } from "../i18n/strings";
import type { Tab } from "../params/schema";
import { initBackend } from "../state/runner";
import { useStore } from "../state/store";
import { Credits } from "./Credits";
import { IconMenu, IconMoon, IconSun } from "./icons";
import { Logo } from "./Logo";
import { SnapshotDetails, snapshotStatusText, useSnapshotState } from "./SnapshotNotice";

const TABS: { id: Tab; key: StrKey }[] = [
  { id: "device", key: "tab.device" },
  { id: "circuit", key: "tab.circuit" },
  { id: "validation", key: "tab.validation" },
  { id: "physics", key: "tab.physics" },
];

/** Tabs whose results depend on the Deterministic | Stochastic mode (the toggle shows only there). */
export const hasMode = (tab: Tab) => tab === "device" || tab === "circuit";

/** Context-strip sentence: what the current tab shows in the selected mode, in plain words. */
export function modeHint(t: T, tab: Tab, mode: Mode, method: string): string {
  if (tab === "circuit") return t(mode === "deterministic" ? "mode.circuit.deterministic.hint" : "mode.circuit.stochastic.hint", { method });
  if (tab === "validation") return t("mode.validation.hint");
  if (tab === "physics") return t("mode.physics.hint");
  return t(mode === "deterministic" ? "mode.deterministic.hint" : "mode.stochastic.hint");
}

/** The technical line behind the plain sentence (its tooltip). */
export function modeHintTech(t: T, tab: Tab, mode: Mode, method: string): string {
  if (tab === "circuit") return t(mode === "deterministic" ? "mode.circuit.deterministic.tech" : "mode.circuit.stochastic.tech", { method });
  if (tab === "validation") return t("mode.validation.tech");
  if (tab === "physics") return t("mode.physics.tech");
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

/** [결정론 | 확률]: one word per mode, the plain hint of each mode in its tooltip. */
export function ModeToggle({ compact }: { compact?: boolean }) {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const tab = useStore((s) => s.tab);
  const method = useStore((s) => s.params.circuit.solver.method);
  const modes: Mode[] = ["deterministic", "stochastic"];
  return (
    <div className={`mode-toggle${compact ? " compact" : ""}`} role="radiogroup" aria-label={t("mode.aria")} data-testid="mode-toggle" onKeyDown={(e) => arrowNav(e, modes, mode, setMode)}>
      {modes.map((m) => (
        <button
          key={m}
          type="button"
          role="radio"
          aria-checked={mode === m}
          tabIndex={mode === m ? 0 : -1}
          className={`mode-btn ${m === "deterministic" ? "det" : "sto"}`}
          onClick={() => setMode(m)}
          title={modeHint(t, tab === "circuit" ? "circuit" : "device", m, method)}
          data-testid={`mode-${m}`}
        >
          <span className="mode-dot" aria-hidden />
          <span>{t(m === "deterministic" ? "mode.deterministic" : "mode.stochastic")}</span>
        </button>
      ))}
    </div>
  );
}

/** Backend status as an 8 px dot; the text stays in the DOM for screen readers (and tests), the tooltip has the rest. */
function StatusDot() {
  const t = useT();
  const backend = useStore((s) => s.backend);
  const health = useStore((s) => s.health);
  const text =
    backend === "online" ? t("status.online", { n: health?.workers ?? "?" }) : backend === "mock" ? t("status.mock") : backend === "offline" ? t("status.offline") : backend === "snapshot" ? snapshotStatusText(t, health) : t("status.checking");
  const short = backend === "online" ? `API · ${health?.workers ?? "?"}w` : backend === "mock" ? "mock" : backend === "offline" ? "offline" : backend === "snapshot" ? t("snapshot.status.short") : "…";
  return (
    <button type="button" className="status" title={`${text}${health?.version ? ` · v${health.version}` : ""}`} aria-label={text} data-testid="backend-status" onClick={() => void initBackend()}>
      <span className={`dot ${backend}`} aria-hidden />
      <span className="sr-only status-text">{short}</span>
    </button>
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
  const phone = usePhone();
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
        <Logo size={30} className="brand-logo" />
        <span className="brand-title">{t("app.title")}</span>
        <span className="tech-chip" title={t("brand.tech.title")} data-testid="tech-chip">
          {t("brand.tech")}
        </span>
      </div>
      <nav className="tabs" role="tablist" aria-label={t("tabs.aria")} onKeyDown={(e) => arrowNav(e, TABS.map((x) => x.id), tab, setTab)}>
        {TABS.map((x) => (
          <button key={x.id} type="button" role="tab" id={`tab-${x.id}`} aria-selected={tab === x.id} aria-controls="main" tabIndex={tab === x.id ? 0 : -1} className="tab" onClick={() => setTab(x.id)} data-testid={`tab-${x.id}`}>
            {t(x.key)}
          </button>
        ))}
      </nav>
      <span className="spacer" />
      {hasMode(tab) && !phone && <ModeToggle />}
      <StatusDot />
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
        title={backend === "mock" ? t("banner.mockForced") : backend === "offline" ? t("banner.offline") : t("snapshot.banner")}
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
          {backend === "snapshot" ? <SnapshotDetails /> : <p>{backend === "mock" ? t("banner.mockForced") : t("banner.offline")}</p>}
        </div>
      )}
    </div>
  );
}

/** Context strip (`modebar`): plain hint · status pill · credits. At ≤ 960 px the mode toggle sits here instead. */
export function ContextStrip() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const tab = useStore((s) => s.tab);
  const method = useStore((s) => s.params.circuit.solver.method);
  const phone = usePhone();
  const hint = modeHint(t, tab, mode, method);
  return (
    <div className={`modestrip${hasMode(tab) ? "" : " no-mode"}`} data-testid="modebar">
      {phone && hasMode(tab) && <ModeToggle compact />}
      <span className="hint" data-testid="mode-hint" title={modeHintTech(t, tab, mode, method)}>
        {hint}
      </span>
      <StatusPill />
      <Credits />
    </div>
  );
}
