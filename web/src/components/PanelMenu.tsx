// Panel ⋯ menu (role=menu: 보기 toggles, 내보내기 actions) and a small tool popover (e.g. the ⚙ V_G range of
// the V_G panels). Keyboard: Enter/Space opens, arrows/Home/End move, Esc closes (focus returns to the
// trigger), Tab or a click outside closes. The popup flips upwards / leftwards when it would leave the viewport.
import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useT } from "../i18n";
import { UX } from "../i18n/strings.ux";
import { IconCheck, IconMore, IconSliders } from "./icons";

/** Section of the ⋯ menu: 보기 (view toggles) or 내보내기 (export actions). */
export type PanelMenuSection = "view" | "export";

interface PanelMenuBase {
  /** Unique within the menu (React key). */
  id: string;
  /** Localised label. */
  label: string;
  /** Default: "view" for check/radio items, "export" for actions. */
  section?: PanelMenuSection;
  disabled?: boolean;
  title?: string;
  testId?: string;
}

/**
 * One item of a panel's ⋯ menu: a checkbox (e.g. ☑ 측정), a radio (consecutive items with the same `group`
 * form one set, `group` is its caption, e.g. 로그 | 선형) or an action (e.g. CSV). Check and radio items keep
 * the menu open; an action closes it.
 */
export type PanelMenuItem =
  | (PanelMenuBase & { kind: "check"; checked: boolean; onChange: (checked: boolean) => void })
  | (PanelMenuBase & { kind: "radio"; group: string; checked: boolean; onSelect: () => void })
  | (PanelMenuBase & { kind: "action"; onSelect: () => void; icon?: ReactNode });

const sectionOf = (it: PanelMenuItem): PanelMenuSection => it.section ?? (it.kind === "action" ? "export" : "view");

/**
 * Keeps a popup (absolutely positioned under its trigger) inside the viewport: it opens upwards when there is
 * no room below, and switches between right- and left-aligned when its preferred side would leave the page.
 */
function useFlip(open: boolean, pop: React.RefObject<HTMLElement | null>, prefer: "left" | "right" = "right") {
  const [flip, setFlip] = useState<{ up: boolean; left: boolean }>({ up: false, left: prefer === "left" });
  useLayoutEffect(() => {
    if (!open) {
      setFlip({ up: false, left: prefer === "left" });
      return;
    }
    const el = pop.current;
    const wrap = el?.parentElement;
    if (!el || !wrap) return;
    const r = el.getBoundingClientRect();
    const w = wrap.getBoundingClientRect();
    const up = r.bottom > window.innerHeight - 8 && w.top - r.height - 4 > 8;
    const left = prefer === "left" ? !(w.left + r.width > window.innerWidth - 8 && w.right - r.width > 8) : r.left < 8 && w.left + r.width < window.innerWidth - 8;
    if (up || left !== (prefer === "left")) setFlip({ up, left });
  }, [open, pop, prefer]);
  return `${flip.up ? " up" : ""}${flip.left ? " left" : ""}`;
}

/** Closes on a pointer-down outside `wrap` while open. */
function useOutside(open: boolean, wrap: React.RefObject<HTMLElement | null>, close: () => void) {
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) close();
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open, wrap, close]);
}

export function PanelMenu({ panelId, items }: { panelId: string; items: PanelMenuItem[] }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement | null>(null);
  const pop = useRef<HTMLDivElement | null>(null);
  const trigger = useRef<HTMLButtonElement | null>(null);
  const menuId = useId();
  const flip = useFlip(open, pop);
  const closeRef = useRef(() => setOpen(false));
  useOutside(open, wrap, closeRef.current);

  useEffect(() => {
    if (open) pop.current?.querySelector<HTMLButtonElement>(".pmenu-item:not(:disabled)")?.focus();
  }, [open]);

  const close = (refocus: boolean) => {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation(); // Esc closes the menu before any window around it
      close(true);
      return;
    }
    if (e.key === "Tab") {
      setOpen(false);
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const els = [...(pop.current?.querySelectorAll<HTMLButtonElement>(".pmenu-item:not(:disabled)") ?? [])];
    if (!els.length) return;
    const i = els.indexOf(document.activeElement as HTMLButtonElement);
    const n = els.length;
    const next = e.key === "Home" ? 0 : e.key === "End" ? n - 1 : e.key === "ArrowDown" ? (i + 1) % n : (i - 1 + n) % n;
    els[next].focus();
  };

  const sections: { id: PanelMenuSection; title: string; items: PanelMenuItem[] }[] = (
    [
      { id: "view", title: t.l(UX["menu.view"]), items: items.filter((it) => sectionOf(it) === "view") },
      { id: "export", title: t.l(UX["menu.export"]), items: items.filter((it) => sectionOf(it) === "export") },
    ] as const
  ).filter((sec) => sec.items.length > 0);

  const renderItem = (it: PanelMenuItem) => {
    const common = { type: "button" as const, className: "pmenu-item", disabled: it.disabled, title: it.title, "data-testid": it.testId };
    if (it.kind === "check")
      return (
        <button key={it.id} {...common} role="menuitemcheckbox" aria-checked={it.checked} onClick={() => it.onChange(!it.checked)}>
          <span className={`pmenu-mark pm-box${it.checked ? " is-on" : ""}`} aria-hidden>
            {it.checked && <IconCheck size={11} />}
          </span>
          <span className="pmenu-label">{it.label}</span>
        </button>
      );
    if (it.kind === "radio")
      return (
        <button key={it.id} {...common} role="menuitemradio" aria-checked={it.checked} onClick={() => it.onSelect()}>
          <span className={`pmenu-mark pm-radio${it.checked ? " is-on" : ""}`} aria-hidden />
          <span className="pmenu-label">{it.label}</span>
        </button>
      );
    return (
      <button
        key={it.id}
        {...common}
        role="menuitem"
        onClick={() => {
          it.onSelect();
          close(true);
        }}
      >
        <span className="pmenu-mark pm-icon" aria-hidden>
          {it.icon}
        </span>
        <span className="pmenu-label">{it.label}</span>
      </button>
    );
  };

  return (
    <div
      className="pmenu-wrap"
      ref={wrap}
      onBlur={(e) => {
        if (open && wrap.current && e.relatedTarget && !wrap.current.contains(e.relatedTarget as Node)) setOpen(false);
      }}
    >
      <button
        ref={trigger}
        type="button"
        className="icon-btn xs"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={t.l(UX["menu.open"])}
        title={t.l(UX["menu.open"])}
        data-testid={`panel-menu-${panelId}`}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" && !open) {
            e.preventDefault();
            setOpen(true);
          }
        }}
      >
        <IconMore size={14} />
      </button>
      {open && (
        <div ref={pop} className={`pmenu${flip}`} role="menu" id={menuId} aria-label={t.l(UX["menu.open"])} onKeyDown={onKey} data-testid={`panel-menu-${panelId}-list`}>
          {sections.map((sec) => (
            <div key={sec.id} className="pmenu-sec" role="group" aria-label={sec.title}>
              <div className="pmenu-title" aria-hidden>
                {sec.title}
              </div>
              {sec.items.map((it, i) => {
                const prev = sec.items[i - 1];
                const caption = it.kind === "radio" && !(prev?.kind === "radio" && prev.group === it.group) ? it.group : null;
                return (
                  <div key={it.id} role="none">
                    {caption && (
                      <div className="pmenu-group" aria-hidden>
                        {caption}
                      </div>
                    )}
                    {renderItem(it)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * A toolbar button that opens a small non-modal popover (role=dialog) with its own controls, e.g.
 * "⚙ 범위" → V_G min/max/points + 다시 계산. Esc or a click outside closes it; focus returns to the button.
 */
export function ToolPopover({
  label,
  title,
  testId,
  icon,
  children,
}: {
  label: string;
  /** Accessible name and heading of the popover. */
  title: string;
  testId?: string;
  icon?: ReactNode;
  children: ReactNode | ((close: () => void) => ReactNode);
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement | null>(null);
  const pop = useRef<HTMLDivElement | null>(null);
  const trigger = useRef<HTMLButtonElement | null>(null);
  const id = useId();
  const flip = useFlip(open, pop, "left");
  const closeRef = useRef(() => setOpen(false));
  useOutside(open, wrap, closeRef.current);
  useEffect(() => {
    if (open) pop.current?.querySelector<HTMLElement>("input, select, button")?.focus();
  }, [open]);
  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };
  return (
    <div className="pmenu-wrap" ref={wrap}>
        <button
          ref={trigger}
          type="button"
          className="btn sm tool-btn"
          aria-haspopup="dialog"
          aria-expanded={open}
          aria-controls={open ? id : undefined}
          data-testid={testId}
          onClick={() => setOpen((o) => !o)}
        >
          {icon ?? <IconSliders size={13} />}
          {label}
        </button>
        {open && (
          <div
            ref={pop}
            id={id}
            className={`pmenu tool-pop${flip}`}
            role="dialog"
            aria-label={title}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                e.stopPropagation();
                close();
              }
            }}
          >
            <div className="pmenu-title">{title}</div>
            {typeof children === "function" ? children(close) : children}
          </div>
        )}
    </div>
  );
}
