// Physics tab: all topics (TOPIC_ORDER) as one long document with a sticky TOC and a search filter,
// rendered with the same TopicBody as the Details window. The "parameters" topic starts with the
// parameter guide (ParamGuideList, filtered by the same search); a chip at the top jumps to it.
import { useEffect, useMemo, useState } from "react";
import { PHYSICS_TOPICS, TOPIC_ORDER, type PhysicsTopic, type TopicId } from "../content/physics";
import { IconSearch } from "../components/icons";
import { RichText } from "../components/RichText";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { useStore } from "../state/store";
import { matchGuideKeys, ParamGuideList } from "./ParamGuideList";
import { TopicBody } from "./TopicBody";

const GUIDE_TOPIC: TopicId = "parameters";

function topicText(t: PhysicsTopic): string {
  const parts: string[] = [t.id, t.title.ko, t.title.en, t.summary.ko, t.summary.en, ...(t.tags ?? [])];
  for (const s of t.sections) {
    parts.push(s.heading.ko, s.heading.en, s.body?.ko ?? "", s.body?.en ?? "");
    for (const e of s.equations ?? []) parts.push(e.id, e.tex, e.label?.ko ?? "", e.label?.en ?? "", e.code ?? "");
    for (const v of s.variables ?? []) parts.push(v.symbol, v.name.ko, v.name.en, v.code ?? "");
    for (const n of s.notes ?? []) parts.push(n.ko, n.en);
  }
  return parts.join(" \n ").toLowerCase();
}

export function PhysicsTab() {
  const t = useT();
  const target = useStore((s) => s.physicsTarget);
  const [q, setQ] = useState("");
  const [active, setActive] = useState<TopicId>(TOPIC_ORDER[0]);
  const index = useMemo(() => new Map(TOPIC_ORDER.map((id) => [id, topicText(PHYSICS_TOPICS[id])])), []);
  const guideHits = useMemo(() => matchGuideKeys(q).size, [q]);
  const shown = useMemo(() => {
    const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return TOPIC_ORDER.filter((id) => terms.every((w) => index.get(id)?.includes(w)) || (id === GUIDE_TOPIC && guideHits > 0));
  }, [q, index, guideHits]);

  const jumpToGuide = () => {
    if (!shown.includes(GUIDE_TOPIC)) setQ("");
    setTimeout(() => {
      const el = document.getElementById("guide-list");
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActive(GUIDE_TOPIC);
      // the TOC observer fires while the smooth scroll passes earlier topics: settle on the target
      setTimeout(() => setActive(GUIDE_TOPIC), 900);
    }, 30);
  };

  const scrollTo = (id: TopicId, flash = false) => {
    const el = document.getElementById(`topic-${id}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    setActive(id);
    if (flash) {
      el.classList.remove("flash");
      void el.offsetWidth;
      el.classList.add("flash");
    }
  };

  useEffect(() => {
    if (!target) return;
    setQ("");
    const id = target.topic;
    const tm = setTimeout(() => scrollTo(id, true), 60);
    return () => clearTimeout(tm);
  }, [target]);

  useEffect(() => {
    const els = shown.map((id) => document.getElementById(`topic-${id}`)).filter((x): x is HTMLElement => !!x);
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id.replace(/^topic-/, "") as TopicId);
      },
      { rootMargin: "-100px 0px -60% 0px" },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, [shown]);

  return (
    <div className="physics-layout" data-testid="physics-tab">
      <aside className="toc" aria-label={t("ph.toc")}>
        <div className="search">
          <IconSearch size={15} />
          <input type="search" placeholder={t("ph.search")} aria-label={t("ph.search")} value={q} onChange={(e) => setQ(e.target.value)} data-testid="physics-search" />
        </div>
        <div className="sub-label" style={{ padding: "4px 10px 0" }}>{t("ph.toc")}</div>
        <ol className="toc-list">
          {shown.map((id) => {
            const tp = PHYSICS_TOPICS[id];
            return (
              <li key={id}>
                <button type="button" className={active === id ? "active" : ""} onClick={() => scrollTo(id)} aria-current={active === id ? "location" : undefined}>
                  <span className="toc-n">{TOPIC_ORDER.indexOf(id) + 1}</span>
                  <span>{t.l(tp.title)}</span>
                  {tp.sections.length === 0 && <span className="badge" style={{ marginLeft: "auto" }}>…</span>}
                </button>
              </li>
            );
          })}
        </ol>
      </aside>
      <div className="doc">
        <div className="guide-jump">
          <button type="button" className="chip" onClick={jumpToGuide} data-testid="guide-jump">
            {t.l(GUIDE["guide.list.jump"])} →
          </button>
          <span className="hint">{t.l(GUIDE["guide.list.jump.hint"])}</span>
        </div>
        {shown.length === 0 && <div className="empty" style={{ height: 200 }}>{t("ph.noresult")}</div>}
        {shown.map((id) => {
          const tp = PHYSICS_TOPICS[id];
          return (
            <article key={id} id={`topic-${id}`} className="doc-topic" data-testid={`topic-${id}`}>
              <h2>
                <span className="mono muted" style={{ fontSize: 13 }}>{TOPIC_ORDER.indexOf(id) + 1}</span>
                {t.l(tp.title)}
              </h2>
              {tp.tags && tp.tags.length > 0 && (
                <div className="pw-tags" style={{ marginBottom: 8 }}>
                  {tp.tags.map((g) => (
                    <span key={g} className="badge">{g}</span>
                  ))}
                </div>
              )}
              {id === GUIDE_TOPIC && <ParamGuideList query={q} />}
              <RichText text={t.l(tp.summary)} className="topic-summary" />
              <TopicBody topic={tp} idPrefix={`doc-${id}`} showSummary={false} onRelated={(r) => scrollTo(r, true)} />
            </article>
          );
        })}
      </div>
    </div>
  );
}
