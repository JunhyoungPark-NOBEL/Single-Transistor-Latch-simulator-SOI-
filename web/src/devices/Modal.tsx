// Small modal dialog (focus trap-lite: focus the dialog, Esc closes, click on the backdrop closes).
import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { IconX } from "../components/icons";
import "./devices.css";

export function Modal({ title, onClose, children, footer, width = 560, testId }: { title: ReactNode; onClose: () => void; children: ReactNode; footer?: ReactNode; width?: number; testId?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      prev?.focus?.();
    };
  }, [onClose]);
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={typeof title === "string" ? title : undefined} ref={ref} tabIndex={-1} style={{ width: `min(${width}px, calc(100vw - 24px))` }} data-testid={testId}>
        <header className="modal-head">
          <h2>{title}</h2>
          <button type="button" className="icon-btn xs" onClick={onClose} aria-label="close" data-testid="modal-close">
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
