// Physics tab: the parameter guide first ("what raising each value does to V_LU·V_LD", filtered by the same
// search), then all topics (TOPIC_ORDER) as one long document with a sticky TOC and a search filter, rendered
// with the same TopicBody as the Details window. The TOC starts with an unnumbered "파라미터 가이드" entry.
import { useEffect, useMemo, useState } from "react";
import { PHYSICS_TOPICS, TOPIC_ORDER, type PhysicsTopic, type TopicId } from "../content/physics";
import { IconSearch } from "../components/icons";
import { RichText } from "../components/RichText";
import { useT } from "../i18n";
import { TopicTag } from "./tags";
import { GUIDE } from "../i18n/strings.guide";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { useStore } from "../state/store";
import { matchGuideKeys, ParamGuideList } from "./ParamGuideList";
import { TopicBody } from "./TopicBody";
import "./studio-docs.css";

/** The guide article above the numbered topics (not a topic: the topic- count stays 18). */
const GUIDE_ID = "physics-guide";
type Active = TopicId | "guide";

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
  const [active, setActive] = useState<Active>("guide");
  const index = useMemo(() => new Map(TOPIC_ORDER.map((id) => [id, topicText(PHYSICS_TOPICS[id])])), []);
  const guideHits = useMemo(() => matchGuideKeys(q).size, [q]);
  const shown = useMemo(() => {
    const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return TOPIC_ORDER.filter((id) => terms.every((w) => index.get(id)?.includes(w)));
  }, [q, index]);
  const showGuide = !q.trim() || guideHits > 0;

  const jumpToGuide = () => {
    if (!showGuide) setQ("");
    setTimeout(() => {
      const el = document.getElementById(GUIDE_ID);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActive("guide");
      // the TOC observer fires while the smooth scroll passes other topics: settle on the target
      setTimeout(() => setActive("guide"), 900);
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
    const els = [GUIDE_ID, ...shown.map((id) => `topic-${id}`)].map((id) => document.getElementById(id)).filter((x): x is HTMLElement => !!x);
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id === GUIDE_ID ? "guide" : (vis[0].target.id.replace(/^topic-/, "") as TopicId));
      },
      { rootMargin: "-100px 0px -60% 0px" },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, [shown, showGuide]);

  return (
    <div className="physics-layout" data-testid="physics-tab">
      <aside className="toc" aria-label={t("ph.toc")}>
        <div className="search">
          <IconSearch size={15} />
          <input type="search" placeholder={t("ph.search")} aria-label={t("ph.search")} value={q} onChange={(e) => setQ(e.target.value)} data-testid="physics-search" />
        </div>
        <div className="sub-label" style={{ padding: "4px 10px 0" }}>{t("ph.toc")}</div>
        <ol className="toc-list">
          <li>
            <button type="button" className={`toc-guide${active === "guide" ? " active" : ""}`} onClick={jumpToGuide} aria-current={active === "guide" ? "location" : undefined} data-testid="guide-jump">
              <span className="toc-n" aria-hidden>
                ★
              </span>
              <span>{t.l(GUIDE["guide.list.jump"])}</span>
            </button>
          </li>
          {shown.map((id) => {
            const tp = PHYSICS_TOPICS[id];
            return (
              <li key={id}>
                <button type="button" className={active === id ? "active" : ""} onClick={() => scrollTo(id)} aria-current={active === id ? "location" : undefined}>
                  <span className="toc-n">{TOPIC_ORDER.indexOf(id) + 1}</span>
                  <span>
                    <SubText text={subs(t.l(tp.title))} />
                  </span>
                  {tp.sections.length === 0 && <span className="badge" style={{ marginLeft: "auto" }}>…</span>}
                </button>
              </li>
            );
          })}
        </ol>
      </aside>
      <div className="doc">
        <nav className="studio-doc-links" aria-label={t.lang === "ko" ? "빠른 안내" : "Quick guides"}>
          <a href={`${import.meta.env.BASE_URL}docs/setup.html`} target="_blank" rel="noopener noreferrer">{t.lang === "ko" ? "실행 · 서버 연결" : "Setup & connection"}<span aria-hidden>↗</span></a>
          <a href={`${import.meta.env.BASE_URL}docs/model-scope.html`} target="_blank" rel="noopener noreferrer">{t.lang === "ko" ? "모델 범위 · 가정" : "Model scope & assumptions"}<span aria-hidden>↗</span></a>
          <a href={`${import.meta.env.BASE_URL}docs/geometry-model.html`} target="_blank" rel="noopener noreferrer">Geometry<span aria-hidden>↗</span></a>
        </nav>
        {/* the parameter guide on top: plain picture + what raising each value does to V_LU·V_LD */}
        {showGuide && (
          <article id={GUIDE_ID} className="doc-topic doc-guide" data-testid="physics-guide" aria-labelledby={`${GUIDE_ID}-title`}>
            <h2 id={`${GUIDE_ID}-title`}>
              <span>
                <SubText text={subs(t.l(GUIDE["guide.list.heading"]))} />
              </span>
            </h2>
            <ParamGuideList query={q} labelledBy={`${GUIDE_ID}-title`} />
          </article>
        )}
        {shown.length === 0 && !showGuide && <div className="empty" style={{ height: 200 }}>{t("ph.noresult")}</div>}
        {shown.map((id) => {
          const tp = PHYSICS_TOPICS[id];
          return (
            <article key={id} id={`topic-${id}`} className="doc-topic" data-testid={`topic-${id}`}>
              <h2>
                <span className="mono muted" style={{ fontSize: 13 }}>{TOPIC_ORDER.indexOf(id) + 1}</span>
                <span>
                  <SubText text={subs(t.l(tp.title))} />
                </span>
              </h2>
              {tp.tags && tp.tags.length > 0 && (
                <div className="pw-tags" style={{ marginBottom: 8 }}>
                  {tp.tags.map((g) => (
                    <TopicTag key={g} tag={g} lang={t.lang} />
                  ))}
                </div>
              )}
              <RichText text={t.l(tp.summary)} className="topic-summary" />
              <TopicBody topic={tp} idPrefix={`doc-${id}`} showSummary={false} onRelated={(r) => scrollTo(r, true)} />
            </article>
          );
        })}
      </div>
    </div>
  );
}
