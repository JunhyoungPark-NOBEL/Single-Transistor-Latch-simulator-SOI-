// App header: brand, primary tabs, Deterministic | Stochastic toggle, backend status, KO/EN, theme.
import type { KeyboardEvent } from "react";
import type { Mode } from "../api/types";
import { useT, type T } from "../i18n";
import type { StrKey } from "../i18n/strings";
import type { Tab } from "../params/schema";
import { initBackend } from "../state/runner";
import { useStore } from "../state/store";
import { IconMenu, IconMoon, IconSun } from "./icons";

const TABS: { id: Tab; key: StrKey; en: string }[] = [
  { id: "device", key: "tab.device", en: "Device" },
  { id: "circuit", key: "tab.circuit", en: "Circuit" },
  { id: "validation", key: "tab.validation", en: "Validation" },
  { id: "physics", key: "tab.physics", en: "Physics" },
];

/** Mode-banner / toggle hint: what the selected mode means on the current tab. */
export function modeHint(t: T, tab: Tab, mode: Mode, method: string): string {
  if (tab === "circuit") return t(mode === "deterministic" ? "mode.circuit.deterministic.hint" : "mode.circuit.stochastic.hint", { method });
  if (tab === "validation") return t("mode.validation.hint");
  if (tab === "physics") return t("mode.physics.hint");
  return t(mode === "deterministic" ? "mode.deterministic.hint" : "mode.stochastic.hint");
}

function arrowNav<T>(e: KeyboardEvent, items: T[], cur: T, set: (v: T) => void) {
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

export function ModeToggle() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const tab = useStore((s) => s.tab);
  const method = useStore((s) => s.params.circuit.solver.method);
  const modes: Mode[] = ["deterministic", "stochastic"];
  return (
    <div className="mode-toggle" role="radiogroup" aria-label={t("mode.aria")} data-testid="mode-toggle" onKeyDown={(e) => arrowNav(e, modes, mode, setMode)}>
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
          <span className="mode-sub">{t(m === "deterministic" ? "mode.deterministic.sub" : "mode.stochastic.sub")}</span>
        </button>
      ))}
    </div>
  );
}

function StatusDot() {
  const t = useT();
  const backend = useStore((s) => s.backend);
  const health = useStore((s) => s.health);
  const text =
    backend === "online" ? t("status.online", { n: health?.workers ?? "?" }) : backend === "mock" ? t("status.mock") : backend === "offline" ? t("status.offline") : t("status.checking");
  return (
    <button type="button" className="status" title={`${text}${health?.version ? ` · v${health.version}` : ""}`} aria-label={text} data-testid="backend-status" onClick={() => void initBackend()}>
      <span className={`dot ${backend}`} aria-hidden />
      <span className="status-text">{backend === "online" ? `API · ${health?.workers ?? "?"}w` : backend === "mock" ? "mock" : backend === "offline" ? "offline" : "…"}</span>
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
  const hasSidebar = tab === "device" || tab === "circuit";
  return (
    <header className="header" role="banner">
      {hasSidebar && (
        <button type="button" className="icon-btn sidebar-btn" aria-label={t("sidebar.toggle")} aria-expanded={sidebarOpen} aria-controls="sidebar" onClick={() => setSidebar(!sidebarOpen)}>
          <IconMenu size={17} />
        </button>
      )}
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
            <path d="M3 18c4 0 5-1 7-4s2-5 4-7 4-3 7-3" opacity=".55" />
            <path d="M3 20h5l2-9 2 9h9" />
          </svg>
        </div>
        <div className="brand-text">
          <span className="brand-title">{t("app.title")}</span>
          <span className="brand-sub">{t("app.subtitle")}</span>
        </div>
      </div>
      <nav className="tabs" role="tablist" aria-label={t("tabs.aria")} onKeyDown={(e) => arrowNav(e, TABS.map((x) => x.id), tab, setTab)}>
        {TABS.map((x) => (
          <button key={x.id} type="button" role="tab" id={`tab-${x.id}`} aria-selected={tab === x.id} aria-controls="main" tabIndex={tab === x.id ? 0 : -1} className="tab" onClick={() => setTab(x.id)} data-testid={`tab-${x.id}`}>
            {t(x.key)}
            {lang === "ko" && <span className="tab-en">{x.en}</span>}
          </button>
        ))}
      </nav>
      <span className="spacer" />
      <ModeToggle />
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

export function ModeBar() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const tab = useStore((s) => s.tab);
  const method = useStore((s) => s.params.circuit.solver.method);
  const hint = modeHint(t, tab, mode, method);
  return (
    <div className="modestrip" data-testid="modebar">
      <span className="modestrip-tag">{t(mode === "deterministic" ? "mode.deterministic" : "mode.stochastic")}</span>
      <span className="sep" aria-hidden />
      <span className="hint" data-testid="mode-hint" title={hint}>{hint}</span>
    </div>
  );
}
