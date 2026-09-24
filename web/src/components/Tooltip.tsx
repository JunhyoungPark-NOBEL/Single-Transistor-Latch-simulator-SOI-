// Small hover/focus tooltip ("ⓘ") — short help next to parameter labels; distinct from the Details window.
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function InfoTip({ content, label, testId }: { content: ReactNode; label: string; testId?: string }) {
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const id = useId();
  const show = (el: HTMLElement) => setAnchor(el.getBoundingClientRect());
  const hide = () => {
    setAnchor(null);
    setPos(null);
  };
  useLayoutEffect(() => {
    if (!anchor || !tipRef.current) return;
    const r = tipRef.current.getBoundingClientRect();
    let left = anchor.left - 10;
    let top = anchor.bottom + 8;
    if (left + r.width > window.innerWidth - 8) left = window.innerWidth - r.width - 8;
    if (top + r.height > window.innerHeight - 8) top = anchor.top - r.height - 8;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [anchor]);
  return (
    <>
      <button
        type="button"
        className="info-dot"
        aria-label={label}
        aria-describedby={anchor ? id : undefined}
        data-testid={testId}
        onMouseEnter={(e) => show(e.currentTarget)}
        onMouseLeave={hide}
        onFocus={(e) => show(e.currentTarget)}
        onBlur={hide}
        onKeyDown={(e) => e.key === "Escape" && hide()}
        onClick={(e) => e.preventDefault()}
      >
        i
      </button>
      {anchor &&
        createPortal(
          <div ref={tipRef} role="tooltip" id={id} className="tip" style={pos ?? { left: -9999, top: -9999 }}>
            {content}
          </div>,
          document.body,
        )}
    </>
  );
}
