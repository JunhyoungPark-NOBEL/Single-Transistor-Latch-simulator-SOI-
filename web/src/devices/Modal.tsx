// Small modal dialog (focus trap-lite: focus moves to the first control, Esc closes, click on the backdrop
// closes, focus returns to the opener). The effect runs once per opening: a parent re-render that passes a
// new `onClose` must not steal focus back from a field the user is typing in.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { IconX } from "../components/icons";
import { useT } from "../i18n";
import "./devices.css";

/** First control that should take focus when the dialog opens (the close button only as a last resort). */
const FIRST = "[autofocus], input:not([type=hidden]):not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]):not([data-testid=modal-close])";

export function Modal({ title, onClose, children, footer, width = 560, testId }: { title: ReactNode; onClose: () => void; children: ReactNode; footer?: ReactNode; width?: number; testId?: string }) {
  const t = useT();
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  // the opener, read during the first render: a child's autoFocus moves focus before any effect runs
  const [opener] = useState(() => (typeof document !== "undefined" ? (document.activeElement as HTMLElement | null) : null));
  useEffect(() => {
    const prev = opener;
    const first = ref.current?.querySelector<HTMLElement>(FIRST);
    (first ?? ref.current)?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        closeRef.current();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      if (prev && document.contains(prev)) prev.focus?.();
    };
  }, [opener]);
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={typeof title === "string" ? title : undefined} ref={ref} tabIndex={-1} style={{ width: `min(${width}px, calc(100vw - 24px))` }} data-testid={testId}>
        <header className="modal-head">
          <h2>{title}</h2>
          <button type="button" className="icon-btn xs" onClick={onClose} aria-label={t("close")} title={t("close")} data-testid="modal-close">
            <IconX size={15} />
          </button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-foot">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}
