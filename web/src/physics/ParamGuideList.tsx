// Physics tab → "parameters" topic: the parameter guide for every group in sidebar order (main fields first),
// in the same row format as the Details window's "한눈에" block. The tab's search box filters the rows.
import { useMemo } from "react";
import { GuideText } from "../components/GuidePopover";
import "../components/guide.css";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { GUIDE_LEGEND, guideFor, isMainAnywhere } from "../params/guideUi";
import { GROUPS, LIGHT_FIELDS, type FieldDef, type GroupDef } from "../params/schema";
import { GuideRow, guideRowText } from "./GuideBlock";

interface ListGroup {
  group: GroupDef;
  keys: { key: string; main: boolean }[];
}

/** Groups in sidebar order: basic device groups, advanced device groups, then the circuit-only groups. */
const ORDER: ListGroup[] = (() => {
  const device = GROUPS.filter((g) => g.tabs.includes("device"));
  const circuitOnly = GROUPS.filter((g) => !g.tabs.includes("device"));
  const seq = [...device.filter((g) => !g.advanced), ...device.filter((g) => g.advanced), ...circuitOnly.filter((g) => !g.advanced), ...circuitOnly.filter((g) => g.advanced)];
  return seq
    .map((group) => {
      const fields: FieldDef[] = group.custom?.includes("light") ? [LIGHT_FIELDS.iph, LIGHT_FIELDS.power, LIGHT_FIELDS.resp] : group.fields;
      const withGuide = fields.filter((f) => !!guideFor(f.key));
      const keys = [...withGuide.filter(isMainAnywhere), ...withGuide.filter((f) => !isMainAnywhere(f))].map((f) => ({ key: f.key, main: isMainAnywhere(f) }));
      return { group, keys };
    })
    .filter((x) => x.keys.length > 0);
})();

const TEXT = new Map(ORDER.flatMap((g) => g.keys.map(({ key }) => [key, guideRowText(key)] as const)));

const terms = (q: string) => q.trim().toLowerCase().split(/\s+/).filter(Boolean);

/** Keys whose row matches every search term (all keys for an empty query). */
export function matchGuideKeys(q: string): Set<string> {
  const ws = terms(q);
  return new Set([...TEXT.entries()].filter(([, txt]) => ws.every((w) => txt.includes(w))).map(([k]) => k));
}

/** `labelledBy`: the id of an enclosing heading that already names the list (its own h3 is then left out). */
export function ParamGuideList({ query, labelledBy }: { query: string; labelledBy?: string }) {
  const t = useT();
  const hits = useMemo(() => matchGuideKeys(query), [query]);
  const groups = ORDER.map((g) => ({ ...g, keys: g.keys.filter((k) => hits.has(k.key)) })).filter((g) => g.keys.length > 0);
  return (
    <section className="guide-list" id="guide-list" data-testid="guide-list" aria-labelledby={labelledBy ?? "guide-list-title"}>
      {!labelledBy && <h3 id="guide-list-title">{t.l(GUIDE["guide.list.title"])}</h3>}
      <p className="lead">
        <GuideText text={t.l(GUIDE["guide.list.lead"])} plain />
      </p>
      {groups.length === 0 && <p className="small muted">{t.l(GUIDE["guide.list.none"])}</p>}
      {groups.map(({ group, keys }) => {
        const lead = guideFor(group.id);
        return (
          <div key={group.id} className="glist-group" data-testid={`guide-group-${group.id}`}>
            <h4>
              {t(group.title)}
              {group.stochasticOnly && <span className="badge sto">{t.l(GUIDE["guide.list.sto"])}</span>}
              {!group.tabs.includes("device") && <span className="badge">{t.l(GUIDE["guide.list.circuit"])}</span>}
            </h4>
            {lead && (
              <p className="glist-lead">
                <GuideText text={t.l(lead.intuitive)} />
              </p>
            )}
            {keys.map(({ key, main }) => (
              <GuideRow key={key} fieldKey={key} testId={`guide-row-${key}`} heading="h5" main={main} />
            ))}
          </div>
        );
      })}
      <p className="pwg-foot">
        <GuideText text={t.l(GUIDE_LEGEND.arrows)} plain />
      </p>
    </section>
  );
}
