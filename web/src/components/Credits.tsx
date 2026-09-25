// Credits corner + About popover. The credits sit at the right end of the sticky context strip (never over
// plots or the Run bar): the full attribution at ≥ 1280 px, "KAIST · NOBEL Lab ⓘ" below (CSS). Clicking
// opens a small non-modal About card anchored under it: logo, description, device + model scope, lab /
// advisor / institution / developer, and the app/engine version from GET /api/health.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { IconX } from "./icons";
import { BiristorGlyph, Logo } from "./Logo";

function Sub({ text }: { text: string }) {
  // "L_g 500 nm · T_Si 50 nm" → subscripts for the X_y tokens
  const parts = text.split(/([A-Za-z]+_[A-Za-z]+)/g);
  return (
    <>
      {parts.map((p, i) => {
        const m = /^([A-Za-z]+)_([A-Za-z]+)$/.exec(p);
        return m ? (
          <span key={i}>
            {m[1]}
            <sub>{m[2]}</sub>
          </span>
        ) : (
          <span key={i}>{p}</span>
        );
      })}
    </>
  );
}

function VersionLine() {
  const t = useT();
  const backend = useStore((s) => s.backend);
  const health = useStore((s) => s.health) as (Record<string, unknown> & { version?: string }) | null;
  if (backend === "mock") return <>{t("brand.about.version.demo")}</>;
  if (!health || backend === "offline") return <>{t("brand.about.version.offline")}</>;
  const app = typeof health.app_version === "string" ? health.app_version : null;
  const eng = typeof health.engine_version === "string" ? health.engine_version : typeof health.version === "string" ? health.version : null;
  const items = [app && t("brand.about.version.app", { v: app }), eng && t("brand.about.version.engine", { v: eng.slice(0, 8) })].filter(Boolean);
  return <span className="mono">{items.join(" · ") || "—"}</span>;
}

function About({ anchor, onClose }: { anchor: HTMLElement | null; onClose: (refocus: boolean) => void }) {
  const t = useT();
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; right: number }>({ top: 100, right: 16 });

  useLayoutEffect(() => {
    const place = () => {
      const r = anchor?.getBoundingClientRect();
      if (r) setPos({ top: Math.round(r.bottom + 8), right: Math.max(8, Math.round(window.innerWidth - r.right)) });
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [anchor]);

  useEffect(() => {
    ref.current?.querySelector<HTMLElement>("h2")?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose(true);
      }
    };
    const onDown = (e: PointerEvent) => {
      const el = e.target as Node;
      if (ref.current?.contains(el) || anchor?.contains(el)) return;
      onClose(false);
    };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [anchor, onClose]);

  const rows: [string, ReactNode][] = [
    [t("brand.about.lab"), t("brand.about.lab.value")],
    [t("brand.about.advisor"), t("brand.about.advisor.value")],
    [t("brand.about.institution"), t("brand.about.institution.value")],
    [t("brand.about.developer"), t("brand.about.developer.value")],
    [t("brand.about.version"), <VersionLine key="v" />],
  ];

  return createPortal(
    <div ref={ref} className="about" role="dialog" aria-modal="false" aria-labelledby="about-title" style={{ top: pos.top, right: pos.right }} data-testid="about">
      <button type="button" className="icon-btn xs about-close" onClick={() => onClose(true)} aria-label={t("close")} title={t("close")}>
        <IconX size={14} />
      </button>
      <div className="about-head">
        <Logo size={56} className="about-logo" />
        <div className="about-titles">
          <h2 id="about-title" tabIndex={-1}>
            {t("app.title")}
          </h2>
          <div className="about-tagline">{t("brand.about.tagline")}</div>
        </div>
      </div>
      <p className="about-body">{t("brand.about.body")}</p>
      <div className="about-device">
        <div className="about-device-head">
          {/* the device's equivalent symbol (floating-base NPN), small and monochrome in the brand colour */}
          <BiristorGlyph size={20} title={t("brand.about.glyph")} className="about-glyph" style={{ color: "var(--logo-b)", flex: "none" }} />
          <span className="tech-chip">{t("brand.tech")}</span>
          <span className="about-device-geom">
            <Sub text={t("brand.about.device.value").replace(/^FDSOI · /, "")} />
          </span>
        </div>
        <div className="about-note">
          {t("brand.about.scope")} {t("brand.about.device.soon")}
        </div>
      </div>
      <dl className="about-credits">
        {rows.map(([k, v]) => (
          <div key={k} className="about-row">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>,
    document.body,
  );
}

/** Credits chip for the mode strip; opens the About card. */
export function Credits() {
  const t = useT();
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) btn.current?.focus();
  }, []);
  return (
    <>
      <button
        ref={btn}
        type="button"
        className="credits"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={t("brand.credits.aria")}
        title={t("brand.credits.full")}
        onClick={() => setOpen((o) => !o)}
        data-testid="credits-chip"
      >
        <span className="credits-text full">{t("brand.credits.full")}</span>
        <span className="credits-text short">{t("brand.credits.short")}</span>
        <svg className="credits-i" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden focusable="false">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5.5M12 7.6v.01" />
        </svg>
      </button>
      {open && <About anchor={btn.current} onClose={close} />}
    </>
  );
}
