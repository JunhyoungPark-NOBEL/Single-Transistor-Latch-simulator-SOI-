// ⓘ parameter popover (replaces the hover-only InfoTip for sidebar fields). The guide comes first:
// 쉽게 말하면 (intuitive) → 키우면 (3 effect lines, V_LU blue / V_LD red) → ⚠ caveat → 근거, then a divider
// and today's technical help (meaning, code index, default, range), then "물리 자세히 보기 →".
//
// Behaviour: hover (300 ms) or keyboard focus shows a read-only preview (role="tooltip", same content);
// click / tap / Enter pins it (role="dialog", non-modal, ✕, Esc, click outside; focus returns to the ⓘ).
// One popover at a time (module store). It stacks above the Details window, and its Esc handler runs in
// the capture phase and marks the event handled, so Esc closes the popover before the window.
// At ≤ 760 px the pinned popover is a bottom sheet (max 70 vh, own scroll).
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode, type Ref } from "react";
import { createPortal } from "react-dom";
import { create } from "zustand";
import type { L10n } from "../content/physics/types";
import { useT, type T } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { effectChips, inlineLead, splitLead, type Effect, type GuideVerb, type ParamGuide } from "../params/guideUi";
import { IconAlert, IconX } from "./icons";
import "./guide.css";

// ---------------------------------------------------------------- text helpers (shared by every guide view)
const SYM = /(V_(?:LU|LD|GD|G|D,max|D)|I_(?:PH|p)|E_G|τ_[pn]|δφ_[GE]0|dφ_[GE]|dV_(?:LU|LD)|a_loc)/g;

/** Guide text with symbol subscripts (V_G → V<sub>G</sub>); V_LU / V_LD in their quantity colours. */
export function GuideText({ text, plain = false }: { text: string; plain?: boolean }) {
  const parts = text.split(SYM);
  return (
    <>
      {parts.map((p, i) => {
        if (i % 2 === 0) return p;
        const u = p.indexOf("_");
        const base = p.slice(0, u);
        const sub = p.slice(u + 1);
        const cls = plain ? undefined : p === "V_LU" ? "q-lu" : p === "V_LD" ? "q-ld" : undefined;
        return (
          <span key={i} className={cls ? `q ${cls}` : "q"}>
            {base}
            <sub>{sub}</sub>
          </span>
        );
      })}
    </>
  );
}

/** One effect line as written in the guide, with the leading V_LU / V_LD token coloured. */
export function EffectLine({ line }: { line: string }) {
  const { q, rest } = splitLead(line);
  return (
    <>
      {q && (
        <span className={`q ${q === "V_LU" ? "q-lu" : "q-ld"} lead`}>
          V<sub>{q.slice(2)}</sub>
        </span>
      )}
      <GuideText text={rest} />
    </>
  );
}

const ARROW: Record<Effect["dir"], string> = { up: "↑", down: "↓", flat: "→" };

/** Compact chip parsed from an effect line: "V_LU ↑ 80 mV", "V_LD → 그대로", "V_LU 평균 → 그대로 · σ ↑". */
export function EffectChip({ e }: { e: Effect }) {
  const t = useT();
  if (!e.q) return null;
  return (
    <span className={`gchip ${e.q === "V_LU" ? "lu" : "ld"}`} data-dir={e.dir}>
      <span className={`q ${e.q === "V_LU" ? "q-lu" : "q-ld"}`}>
        V<sub>{e.q.slice(2)}</sub>
      </span>{" "}
      {e.mean && (
        <>
          <span className="gchip-mean">{t.l(GUIDE["guide.mean"])}</span>{" "}
        </>
      )}
      <span className="gchip-arrow">{ARROW[e.dir]}</span>{" "}
      <span className="gchip-mag">{e.dir === "flat" ? t.l(GUIDE["guide.flat"]) : e.mag ?? ""}</span>
      {e.sigma && (
        <>
          {" "}
          <span className="gchip-sigma">· σ {e.sigma === "up" ? "↑" : "↓"}</span>
        </>
      )}
    </span>
  );
}

export const verbLabel = (t: T, v: GuideVerb): string => t.l(GUIDE[v === "on" ? "guide.on" : v === "change" ? "guide.change" : "guide.raise"]);

/** "키우면 (+0.1 V)  V_LU ↑ 80 mV  V_LD → 그대로": the step the guide measured with comes right after the verb,
 *  so a chip is never read as "any increase moves V_LU by 80 mV" (and, for a negative V_G, "+0.1 V" says
 *  which way "raise" goes). When neither voltage moves (V_D,max, cycle counts), the guide's third line says
 *  what does change ("래치업되는 사이클 비율만 ↑ …"), so the row never reads as "this knob does nothing". */
export function EffectChips({ guide, verb, refTag }: { guide: ParamGuide; verb: GuideVerb; refTag?: boolean }) {
  const t = useT();
  const [lu, ld] = effectChips(guide, t.lang);
  if (!lu?.q && !ld?.q) return null;
  const step = lu?.step ?? ld?.step;
  const still = (e: Effect | null) => !e || (e.dir === "flat" && !e.sigma);
  // (units glued to their numbers so "100 %" never breaks across lines)
  const why = still(lu) && still(ld) ? t.l(guide.effect[2]).split(" · ")[0].trim().replace(/(\d) (%|V|mV)/g, "$1\u00a0$2") : "";
  return (
    // the numbers are measured on the reference calibration (V_G = −2 V, dark): said in the tooltip
    <span className="gchips" title={legendNote(t)}>
      <span className="gchips-lead">{verbLabel(t, verb)}</span>
      {step && (
        <>
          {" "}
          <span className="gchips-step">({step})</span>
        </>
      )}{" "}
      {lu && <EffectChip e={lu} />}{" "}
      {ld && <EffectChip e={ld} />}
      {why && (
        <>
          {" "}
          <span className="gchips-why">
            · <GuideText text={why} plain />
          </span>
        </>
      )}
      {/* another preset is active: the arrows and numbers are still those of the reference calibration */}
      {refTag && (
        <>
          {" "}
          <span className="gchips-ref" data-testid="gchips-ref">
            {t.l(GUIDE["guide.refTag"])}
          </span>
        </>
      )}
    </span>
  );
}

/** Caveat callout: ⚠ + text clamped to 2 lines with a [더 보기] toggle (interactive only when pinned). */
export function Caveat({ text, interactive }: { text: string; interactive: boolean }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);
  const [long, setLong] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (el) setLong(el.scrollHeight > el.clientHeight + 2);
  }, [text]);
  return (
    <div className="gp-caveat" role="note">
      <IconAlert size={14} className="gp-caveat-icon" aria-hidden />
      <div className="gp-caveat-body">
        <span className="sr-only">{t.l(GUIDE["guide.caveat"])}: </span>
        <p ref={ref} className={`gp-caveat-text${open ? " open" : ""}`}>
          <GuideText text={text} plain />
        </p>
        {interactive && (long || open) && (
          <button type="button" className="link-btn gp-caveat-more" aria-expanded={open} onClick={() => setOpen(!open)}>
            {t.l(GUIDE[open ? "guide.caveat.less" : "guide.caveat.more"])}
          </button>
        )}
      </div>
    </div>
  );
}

/** How to read the arrows (a UI-owned short form of GUIDE_LEGEND.arrows): "화살표는 이 값을 키울 때의 변화, 괄호 안은
 *  키운 폭입니다 (기준 보정: V_G = −2 V, 암조건)." */
export const legendNote = (t: T) => t.l(GUIDE["guide.legend.short"]);

// ---------------------------------------------------------------- popover store (one open at a time)
type PopMode = "preview" | "pinned";
interface PopState {
  id: string | null;
  mode: PopMode | null;
  trigger: HTMLElement | null;
  show: (id: string, mode: PopMode, trigger: HTMLElement) => void;
  hide: (id?: string) => void;
}
export const useGuidePop = create<PopState>((set, get) => ({
  id: null,
  mode: null,
  trigger: null,
  show: (id, mode, trigger) => set({ id, mode, trigger }),
  hide: (id) => {
    if (id && get().id !== id) return;
    set({ id: null, mode: null, trigger: null });
  },
}));

/** Pin the popover of a trigger (the inline guide line uses this with the field's ⓘ button). */
export function pinGuide(id: string, trigger: HTMLElement | null | undefined) {
  if (trigger) useGuidePop.getState().show(id, "pinned", trigger);
}

/** Trigger that is getting focus back after its popover closed: that focus must not re-open a preview. */
let refocusing: HTMLElement | null = null;

const PHONE = "(max-width: 760px)";
const isPhone = () => typeof window !== "undefined" && !!window.matchMedia?.(PHONE).matches;

// ---------------------------------------------------------------- card content
export interface GuideCardProps {
  /** Symbol (KaTeX or text) shown in the header. */
  sym?: ReactNode;
  label: string;
  guide?: ParamGuide | null;
  verb?: GuideVerb;
  /** Today's technical help block (meaning, code index, default and range). */
  technical: ReactNode;
  /** "물리 자세히 보기 →" (pinned only). */
  onMore?: () => void;
}

function GuideCard({ sym, label, guide, verb = "raise", technical, onMore, pinned, headingId, onClose }: GuideCardProps & { pinned: boolean; headingId: string; onClose: () => void }) {
  const t = useT();
  const L = (x: L10n) => t.l(x);
  return (
    <>
      <div className="gp-head">
        {sym && <span className="gp-sym">{sym}</span>}
        <span className="gp-title" id={headingId}>
          {label}
        </span>
        {pinned && (
          <button type="button" className="icon-btn xs gp-close" onClick={onClose} aria-label={L(GUIDE["guide.close"])} title={`${L(GUIDE["guide.close"])} (Esc)`} data-testid="guide-pop-close">
            <IconX size={14} />
          </button>
        )}
      </div>
      {guide && (
        <>
          <div className="gp-block gp-easy">
            <div className="gp-h">{L(GUIDE["guide.easy"])}</div>
            <p className="gp-intuitive" data-testid="guide-pop-intuitive">
              <GuideText text={L(guide.intuitive)} />
            </p>
          </div>
          <div className="gp-block gp-eff">
            <div className="gp-h">
              {verbLabel(t, verb)}
              {verb === "raise" && <span aria-hidden> ↑</span>}
            </div>
            <ul className="gp-effects" data-testid="guide-pop-effects">
              {guide.effect.map((e, i) => (
                <li key={i} className={i === 2 ? "why" : undefined}>
                  <EffectLine line={L(e)} />
                </li>
              ))}
            </ul>
          </div>
          {guide.caveat && <Caveat text={L(guide.caveat)} interactive={pinned} />}
          {/* the guide's provenance note (English, for guide.test.ts) stays in the tooltip, not in the body */}
          <p className="gp-basis" title={guide.basis ? `${L(GUIDE["guide.basis"])}: ${guide.basis}` : undefined}>
            <GuideText text={legendNote(t)} plain />
          </p>
        </>
      )}
      <div className={`gp-tech${guide ? "" : " solo"}`}>
        {guide && <div className="gp-h">{L(GUIDE["guide.tech"])}</div>}
        {technical}
      </div>
      {pinned && onMore && (
        <div className="gp-foot">
          <button type="button" className="link-btn gp-more" onClick={onMore} data-testid="guide-pop-more">
            {L(GUIDE["guide.physics"])} →
          </button>
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------- trigger + popover
export interface GuidePopoverProps extends GuideCardProps {
  /** Unique id of this trigger (e.g. the field key). */
  id: string;
  /** Accessible name of the ⓘ button. */
  ariaLabel: string;
  testId?: string;
  /** Receives the ⓘ button (the inline guide line pins the popover through it). */
  triggerRef?: Ref<HTMLButtonElement>;
}

export function GuidePopover({ id, ariaLabel, testId, triggerRef, ...card }: GuidePopoverProps) {
  const open = useGuidePop((s) => s.id === id);
  const mode = useGuidePop((s) => (s.id === id ? s.mode : null));
  const btn = useRef<HTMLButtonElement | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const uid = useId();
  const popId = `gp-${uid}`;
  const headingId = `gp-h-${uid}`;

  const setBtn = useCallback(
    (el: HTMLButtonElement | null) => {
      btn.current = el;
      if (typeof triggerRef === "function") triggerRef(el);
      else if (triggerRef) (triggerRef as { current: HTMLButtonElement | null }).current = el;
    },
    [triggerRef],
  );

  useEffect(() => () => clearTimeout(timer.current), []);
  // unmount while open (group collapsed, field hidden): drop the popover
  useEffect(() => () => useGuidePop.getState().hide(id), [id]);

  const pinnedElsewhere = () => {
    const s = useGuidePop.getState();
    return s.mode === "pinned" && s.id !== null && s.id !== id;
  };
  const preview = () => {
    if (!btn.current || pinnedElsewhere()) return;
    const s = useGuidePop.getState();
    if (s.id === id && s.mode === "pinned") return;
    s.show(id, "preview", btn.current);
  };
  const unpreview = () => {
    clearTimeout(timer.current);
    const s = useGuidePop.getState();
    if (s.id === id && s.mode === "preview") s.hide(id);
  };

  return (
    <>
      <button
        ref={setBtn}
        type="button"
        className={`info-dot${open ? " on" : ""}`}
        aria-label={ariaLabel}
        aria-haspopup="dialog"
        aria-expanded={mode === "pinned"}
        aria-describedby={mode === "preview" ? popId : undefined}
        data-testid={testId}
        onMouseEnter={() => {
          clearTimeout(timer.current);
          timer.current = setTimeout(preview, 300);
        }}
        onMouseLeave={unpreview}
        onFocus={(e) => {
          if (refocusing === e.currentTarget) {
            refocusing = null;
            return;
          }
          // keyboard focus previews at once; a mouse/touch press is followed by a click that pins
          if (e.currentTarget.matches(":focus-visible")) preview();
        }}
        onBlur={unpreview}
        onKeyDown={(e) => {
          if (e.key === "Escape" && mode === "preview") {
            e.preventDefault();
            unpreview();
          }
        }}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          clearTimeout(timer.current);
          const s = useGuidePop.getState();
          if (s.id === id && s.mode === "pinned") s.hide(id);
          else s.show(id, "pinned", e.currentTarget);
        }}
      >
        i
      </button>
      {open && mode && <PopoverLayer id={id} card={card} mode={mode} popId={popId} headingId={headingId} />}
    </>
  );
}

function PopoverLayer({ id, card, mode, popId, headingId }: { id: string; card: GuideCardProps; mode: PopMode; popId: string; headingId: string }) {
  const { onMore } = card;
  const ref = useRef<HTMLDivElement>(null);
  const trigger = useGuidePop((s) => s.trigger);
  const [pos, setPos] = useState<{ left: number; top: number; maxHeight?: number } | null>(null);
  const [sheet, setSheet] = useState(() => isPhone());
  const pinned = mode === "pinned";

  const close = useCallback(
    (returnFocus: boolean) => {
      const s = useGuidePop.getState();
      const tr = s.trigger;
      s.hide(id);
      if (returnFocus && tr && document.contains(tr))
        setTimeout(() => {
          refocusing = tr;
          tr.focus({ preventScroll: true });
          if (document.activeElement !== tr || refocusing === tr) refocusing = null;
        }, 0);
    },
    [id],
  );

  // position next to the trigger (desktop) or as a bottom sheet (phone); follow the trigger on scroll
  useLayoutEffect(() => {
    const place = () => {
      const phone = isPhone();
      setSheet(phone);
      const el = ref.current;
      if (phone || !el || !trigger) return setPos(null);
      const a = trigger.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const w = el.offsetWidth;
      const h = el.scrollHeight;
      const left = Math.max(8, Math.min(a.left - 12, vw - w - 8));
      const below = vh - a.bottom - 16;
      const above = a.top - 16;
      if (h <= below) return setPos({ left, top: a.bottom + 8 });
      if (h <= above) return setPos({ left, top: a.top - 8 - h });
      // no room above or below: beside the trigger, as tall as the viewport allows (scrolls inside beyond that)
      const side = a.right + 10 + w <= vw - 8 ? a.right + 10 : Math.max(8, a.left - 10 - w);
      const maxHeight = vh - 16;
      setPos({ left: side, top: Math.max(8, Math.min(a.top - 24, vh - 8 - Math.min(h, maxHeight))), maxHeight });
    };
    place();
    let raf = 0;
    const onMove = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(place);
    };
    window.addEventListener("resize", onMove);
    window.addEventListener("scroll", onMove, true);
    const ro = typeof ResizeObserver !== "undefined" && ref.current ? new ResizeObserver(onMove) : null;
    if (ro && ref.current) ro.observe(ref.current);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onMove);
      window.removeEventListener("scroll", onMove, true);
      ro?.disconnect();
    };
  }, [trigger, mode]);

  // pinned: focus moves into the dialog; Esc (capture, before the Details window) and click outside close it
  useEffect(() => {
    if (!pinned) return;
    ref.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      close(true);
    };
    const onDown = (e: PointerEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      if (ref.current?.contains(target)) return;
      const tr = useGuidePop.getState().trigger;
      if (tr && tr.contains(target)) return; // the trigger's own click toggles
      if ((target as Element).closest?.("[data-guide-pin]")) return; // an inline guide line re-pins
      close(false);
    };
    window.addEventListener("keydown", onKey, true);
    document.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      document.removeEventListener("pointerdown", onDown, true);
    };
  }, [pinned, close]);

  // preview: Esc anywhere closes it too (keyboard users on the trigger)
  useEffect(() => {
    if (pinned) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        useGuidePop.getState().hide(id);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [pinned, id]);

  const style = sheet ? undefined : pos ? { left: pos.left, top: pos.top, maxHeight: pos.maxHeight } : { left: -9999, top: -9999 };
  const cls = `gp${pinned ? " pinned" : " preview"}${sheet ? " sheet" : ""}`;
  const more = onMore
    ? () => {
        close(false);
        onMore();
      }
    : undefined;
  return createPortal(
    <>
      {pinned && sheet && <div className="gp-scrim" aria-hidden onClick={() => close(true)} />}
      <div
        ref={ref}
        id={popId}
        className={cls}
        style={style}
        role={pinned ? "dialog" : "tooltip"}
        aria-modal={pinned ? false : undefined}
        aria-labelledby={pinned ? headingId : undefined}
        tabIndex={pinned ? -1 : undefined}
        onKeyDown={
          pinned
            ? (e) => {
                // Tab past either end leaves the popover: close it and continue from the ⓘ
                if (e.key !== "Tab" || !ref.current) return;
                const items = Array.from(ref.current.querySelectorAll<HTMLElement>("button, a[href], summary, [tabindex='0']")).filter((el) => !el.hasAttribute("disabled"));
                const first = items[0];
                const last = items[items.length - 1];
                const at = document.activeElement;
                if ((!e.shiftKey && (at === last || !items.length)) || (e.shiftKey && (at === first || at === ref.current))) {
                  e.preventDefault();
                  close(true);
                }
              }
            : undefined
        }
        data-testid="guide-pop"
        data-mode={mode}
      >
        {sheet && pinned && <div className="gp-grab" aria-hidden />}
        <GuideCard {...card} onMore={more} pinned={pinned} headingId={headingId} onClose={() => close(true)} />
      </div>
    </>,
    document.body,
  );
}

/** Inline line beside a main field: the whole intuitive picture with its glosses collapsed ("(GIDL)"),
 *  so the plain "raise it → …" half is always visible. The popover keeps the full text. */
export const intuitiveLead = (t: T, g: ParamGuide) => inlineLead(t.l(g.intuitive));
