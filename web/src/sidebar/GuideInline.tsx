// Inline guide under a main field ("옆에 적고"): the first sentence of the intuitive picture (2-line clamp)
// and the arrow chips "키우면  V_LU ↑ 80 mV  V_LD → 그대로". Clicking the line pins the field's ⓘ popover
// (full text, 3 effect lines, caveat). The line is linked to the input with aria-describedby (id).
import { EffectChips, GuideText, intuitiveLead, pinGuide } from "../components/GuidePopover";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import type { GuideVerb, ParamGuide } from "../params/guideUi";

export function GuideInline({ id, fieldKey, guide, verb, trigger }: { id: string; fieldKey: string; guide: ParamGuide; verb: GuideVerb; trigger: () => HTMLElement | null }) {
  const t = useT();
  return (
    <div
      className="guide-inline"
      id={id}
      data-testid={`guide-inline-${fieldKey}`}
      data-guide-pin=""
      title={t.l(GUIDE["guide.inline.title"])}
      onClick={() => pinGuide(fieldKey, trigger())}
    >
      <p className="gi-text">
        <GuideText text={intuitiveLead(t, guide)} />
      </p>
      <EffectChips guide={guide} verb={verb} />
    </div>
  );
}
