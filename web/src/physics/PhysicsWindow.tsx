// Details window: non-modal, compact floating window with the physics topic. Draggable by its header,
// resizable (CSS resize), Esc/× to close, focus moves in on open and returns to the trigger on close.
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { PHYSICS_TOPICS } from "../content/physics";
import { takeTrigger } from "../components/DetailsButton";
import { IconBack, IconBook, IconExternal, IconGrip, IconX } from "../components/icons";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { TopicBody } from "./TopicBody";

export function PhysicsWindow() {
  const t = useT();
  const pw = useStore((s) => s.physics);
  const close = useStore((s) => s.closePhysics);
  const back = useStore((s) => s.physicsBack);
  const nav = useStore((s) => s.navigatePhysics);
  const move = useStore((s) => s.movePhysics);
  const showInTab = useStore((s) => s.showInPhysicsTab);
  const winRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const trigger = useRef<HTMLElement | null>(null);
  const [active, setActive] = useState(0);
  const [dragging, setDragging] = useState(false);
  const topic = PHYSICS_TOPICS[pw.topic];

  const doClose = useCallback(() => {
    close();
    const el = trigger.current;
    trigger.current = null;
    if (el && document.contains(el)) setTimeout(() => el.focus(), 0);
  }, [close]);

  // focus into the window when it opens / the topic changes; remember the trigger for focus return
  useEffect(() => {
    if (!pw.open) return;
    const tr = takeTrigger();
    if (tr) trigger.current = tr;
    titleRef.current?.focus({ preventScroll: true });
    bodyRef.current?.scrollTo({ top: 0 });
    setActive(0);
  }, [pw.open, pw.topic]);

  // Esc closes (non-modal: the rest of the page stays interactive)
  useEffect(() => {
    if (!pw.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.defaultPrevented) {
        e.preventDefault();
        doClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [pw.open, doClose]);

  // write back user resizes (CSS resize handle) to the store
  useLayoutEffect(() => {
    const el = winRef.current;
    if (!el || !pw.open) return;
    const ro = new ResizeObserver(() => {
      const w = Math.round(el.offsetWidth);
      const h = Math.round(el.offsetHeight);
      const s = useStore.getState().physics;
      if (Math.abs(w - s.w) > 1 || Math.abs(h - s.h) > 1) move({ w, h });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [pw.open, move]);

  // keep inside the viewport on window resize
  useEffect(() => {
    if (!pw.open) return;
    const onResize = () => {
      const s = useStore.getState().physics;
      move({ x: Math.max(8, Math.min(s.x, window.innerWidth - Math.min(s.w, window.innerWidth - 16) - 8)), y: Math.max(8, Math.min(s.y, window.innerHeight - 80)) });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [pw.open, move]);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest("button, a, input")) return;
    const start = { x: e.clientX, y: e.clientY, ox: pw.x, oy: pw.y };
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    setDragging(true);
    const onMove = (ev: PointerEvent) => {
      const w = winRef.current?.offsetWidth ?? pw.w;
      const x = Math.max(8 - w + 120, Math.min(window.innerWidth - 120, start.ox + ev.clientX - start.x));
      const y = Math.max(0, Math.min(window.innerHeight - 48, start.oy + ev.clientY - start.y));
      move({ x, y });
    };
    const onUp = (ev: PointerEvent) => {
      el.releasePointerCapture(ev.pointerId);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      setDragging(false);
    };
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
  };

  // keyboard move with Alt+arrows while the header has focus
  const onHeadKey = (e: React.KeyboardEvent) => {
    if (!e.altKey) return;
    const d = e.shiftKey ? 40 : 12;
    const dx = e.key === "ArrowLeft" ? -d : e.key === "ArrowRight" ? d : 0;
    const dy = e.key === "ArrowUp" ? -d : e.key === "ArrowDown" ? d : 0;
    if (dx || dy) {
      e.preventDefault();
      move({ x: pw.x + dx, y: Math.max(0, pw.y + dy) });
    }
  };

  const onScroll = () => {
    const body = bodyRef.current;
    if (!body) return;
    const secs = Array.from(body.querySelectorAll<HTMLElement>("[data-section]"));
    let cur = 0;
    for (const s of secs) if (s.offsetTop - body.offsetTop <= body.scrollTop + 24) cur = Number(s.dataset.section);
    setActive(cur);
  };

  const goSection = (i: number) => {
    const body = bodyRef.current;
    const el = body?.querySelector<HTMLElement>(`#pw-sec-${i}`);
    if (body && el) body.scrollTo({ top: el.offsetTop - body.offsetTop - 6, behavior: "smooth" });
    setActive(i);
  };

  if (!pw.open || !topic) return null;

  return createPortal(
    <div
      ref={winRef}
      className="pw"
      role="dialog"
      aria-modal="false"
      aria-labelledby="pw-title"
      data-testid="physics-window"
      style={{ left: pw.x, top: pw.y, width: pw.w, height: pw.h }}
    >
      <div className={`pw-head${dragging ? " dragging" : ""}`} onPointerDown={onPointerDown} onKeyDown={onHeadKey} data-testid="physics-window-header" title={t("ph.drag")}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="pw-kicker">
            <IconBook size={12} /> {t("ph.window")}
          </div>
          <h2 className="pw-title" id="pw-title" ref={titleRef} tabIndex={-1}>
            {t.l(topic.title)}
          </h2>
          {topic.tags && topic.tags.length > 0 && (
            <div className="pw-tags">
              {topic.tags.map((g) => (
                <span key={g} className="badge">{g}</span>
              ))}
            </div>
          )}
        </div>
        <div className="pw-actions">
          {pw.history.length > 0 && (
            <button type="button" className="icon-btn" onClick={back} aria-label={t("back")} title={t("back")} data-testid="physics-back">
              <IconBack size={15} />
            </button>
          )}
          <button type="button" className="icon-btn" onClick={() => showInTab(pw.topic)} aria-label={t("ph.openTab")} title={t("ph.openTab")}>
            <IconExternal size={15} />
          </button>
          <button type="button" className="icon-btn" onClick={doClose} aria-label={t("close")} title={`${t("close")} (Esc)`} data-testid="physics-close">
            <IconX size={15} />
          </button>
        </div>
      </div>
      {topic.sections.length > 1 && (
        <nav className="pw-nav" aria-label={t("ph.sections")}>
          {topic.sections.map((s, i) => (
            <button key={i} type="button" className={`pill${active === i ? " active" : ""}`} aria-current={active === i ? "true" : undefined} onClick={() => goSection(i)}>
              {t.l(s.heading)}
            </button>
          ))}
        </nav>
      )}
      <div className="pw-body" ref={bodyRef} onScroll={onScroll} tabIndex={0} aria-label={t.l(topic.title)}>
        <TopicBody topic={topic} idPrefix="pw" onRelated={nav} />
      </div>
      <div className="pw-foot">
        <button type="button" className="link-btn" onClick={() => showInTab(pw.topic)}>
          {t("ph.openTab")} →
        </button>
        <span className="spacer" />
        <span className="small muted">Esc</span>
      </div>
      <IconGrip className="pw-grip" size={12} />
    </div>,
    document.body,
  );
}
