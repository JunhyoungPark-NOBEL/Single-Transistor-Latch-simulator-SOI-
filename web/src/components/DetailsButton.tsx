// "상세 / Details" button: opens the floating PhysicsWindow for a topic next to the button.
import { PHYSICS_TOPICS, type TopicId } from "../content/physics";
import { useT } from "../i18n";
import { useStore } from "../state/store";
import { IconBook } from "./icons";

let lastTrigger: HTMLElement | null = null;
/** Element that opened the Details window (focus returns there on close). */
export const takeTrigger = () => {
  const t = lastTrigger;
  lastTrigger = null;
  return t;
};

export function DetailsButton({ topic, testId }: { topic: TopicId; testId?: string }) {
  const t = useT();
  const open = useStore((s) => s.openPhysics);
  const title = t.l(PHYSICS_TOPICS[topic]?.title) || topic;
  return (
    <button
      type="button"
      className="details-btn"
      aria-haspopup="dialog"
      aria-label={t("details.aria", { topic: title })}
      title={title}
      data-testid={testId ?? `details-${topic}`}
      onClick={(e) => {
        e.stopPropagation();
        lastTrigger = e.currentTarget;
        open(topic, e.currentTarget.getBoundingClientRect());
      }}
    >
      <IconBook size={13} />
      {t("details")}
    </button>
  );
}
