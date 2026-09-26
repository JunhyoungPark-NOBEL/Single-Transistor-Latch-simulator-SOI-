// Hover/focus tooltip for arbitrary content (table headers, row labels). Same look as the parameter ⓘ tips
// (`.tip`), but the trigger is the content itself (keyboard-focusable, dotted underline).
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function HoverTip({ tip, children, className }: { tip: ReactNode; children: ReactNode; className?: string }) {
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const id = useId();
  const hide = () => {
    setAnchor(null);
    setPos(null);
  };
  useLayoutEffect(() => {
    if (!anchor || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    let left = anchor.left + anchor.width / 2 - r.width / 2;
    let top = anchor.bottom + 6;
    if (left + r.width > window.innerWidth - 8) left = window.innerWidth - r.width - 8;
    if (top + r.height > window.innerHeight - 8) top = anchor.top - r.height - 6;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [anchor]);
  if (!tip) return <>{children}</>;
  return (
    <>
      <span
        className={`hovertip${className ? " " + className : ""}`}
        tabIndex={0}
        aria-describedby={anchor ? id : undefined}
        onMouseEnter={(e) => setAnchor(e.currentTarget.getBoundingClientRect())}
        onMouseLeave={hide}
        onFocus={(e) => setAnchor(e.currentTarget.getBoundingClientRect())}
        onBlur={hide}
        onKeyDown={(e) => e.key === "Escape" && hide()}
      >
        {children}
      </span>
      {anchor &&
        createPortal(
          <div ref={ref} role="tooltip" id={id} className="tip stats-tip" style={pos ?? { left: -9999, top: -9999 }}>
            {tip}
          </div>,
          document.body,
        )}
    </>
  );
}
