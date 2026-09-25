// "상세 / Details" button: opens the floating PhysicsWindow for a topic next to the button. Ghost style
// (components/guide.css). From a parameter group it passes the group's visible field keys, so the window
// opens with the "한눈에" guide block first (physics/guideTarget.ts); from anywhere else the guide is cleared.
import { PHYSICS_TOPICS, type TopicId } from "../content/physics";
import { useT } from "../i18n";
import { useGuideTarget } from "../physics/guideTarget";
import { useStore } from "../state/store";
import { IconBook } from "./icons";
import "./guide.css";

let lastTrigger: HTMLElement | null = null;
/** Element that opened the Details window (focus returns there on close). */
export const takeTrigger = () => {
  const t = lastTrigger;
  lastTrigger = null;
  return t;
};

export interface DetailsGuide {
  /** Parameter keys for the "한눈에" block (in display order). */
  params?: string[];
  /** Key to expand and highlight. */
  focus?: string;
  /** Sidebar group id (lead sentence). */
  group?: string;
}

/**
 * Open the Details window on `topic` next to `from` (focus returns to `from` on close). With `guide.params`
 * the window starts with the "한눈에" block for those keys; without, any previous guide target is cleared.
 */
export function openDetails(topic: TopicId, from: HTMLElement | null, guide?: DetailsGuide) {
  lastTrigger = from;
  const gt = useGuideTarget.getState();
  const keys = guide?.params ?? (guide?.focus ? [guide.focus] : []);
  if (keys.length) gt.set({ topic, keys, focus: guide?.focus, group: guide?.group });
  else gt.clear();
  useStore.getState().openPhysics(topic, from?.getBoundingClientRect() ?? null);
}

export function DetailsButton({ topic, testId, params, focus, group, compact }: { topic: TopicId; testId?: string; compact?: boolean } & DetailsGuide) {
  const t = useT();
  const title = t.l(PHYSICS_TOPICS[topic]?.title) || topic;
  return (
    <button
      type="button"
      className={`details-btn${compact ? " icon" : ""}`}
      aria-haspopup="dialog"
      aria-label={t("details.aria", { topic: title })}
      title={compact ? `${t("details")} · ${title}` : title}
      data-testid={testId ?? `details-${topic}`}
      onClick={(e) => {
        e.stopPropagation();
        openDetails(topic, e.currentTarget, { params, focus, group });
      }}
    >
      <IconBook size={compact ? 14 : 13} />
      {!compact && t("details")}
    </button>
  );
}
