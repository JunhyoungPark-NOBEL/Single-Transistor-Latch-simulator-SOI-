// Shared physics-topic renderer (Details window + Physics tab): sections, numbered KaTeX display
// equations with label/note/code chip/copy button, variable tables, notes callouts, related chips.
import { useState } from "react";
import { PHYSICS_TOPICS, type Equation, type PhysicsTopic, type TopicId } from "../content/physics";
import { IconCheck, IconCopy } from "../components/icons";
import { RichText } from "../components/RichText";
import { FitTex, Tex } from "../components/Tex";
import { useT } from "../i18n";

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

function EquationBlock({ eq, n }: { eq: Equation; n: number }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  return (
    <figure className="eq" id={`eq-${eq.id}`} style={{ margin: "10px 0" }} data-testid="equation">
      <div className="eq-label">
        {eq.label ? <span>{t.l(eq.label)}</span> : <span className="muted">{t("ph.eq")} ({n})</span>}
        <span className="spacer" />
        <button
          type="button"
          className="btn sm ghost eq-copy"
          aria-label={`${t("ph.copyTex")} (${n})`}
          onClick={async () => {
            if (await copyText(eq.tex)) {
              setCopied(true);
              setTimeout(() => setCopied(false), 1300);
            }
          }}
        >
          {copied ? <IconCheck size={13} /> : <IconCopy size={13} />} {copied ? t("copied") : t("ph.copyTex")}
        </button>
      </div>
      <div className="eq-row">
        <FitTex tex={eq.tex} className="eq-tex" label={`${t("ph.eq")} (${n}) — ${t("ph.scroll")}`} />
        <span className="eq-num" aria-label={`${t("ph.eq")} ${n}`}>({n})</span>
      </div>
      {eq.note && <RichText text={t.l(eq.note)} className="eq-note" />}
      {eq.code && (
        <div className="eq-code">
          <span className="small muted">{t("ph.code")}</span>
          <code className="code-chip" title={eq.code}>{eq.code}</code>
        </div>
      )}
    </figure>
  );
}

export interface TopicBodyProps {
  topic: PhysicsTopic;
  idPrefix: string;
  onRelated?: (id: TopicId) => void;
  showSummary?: boolean;
}

export function TopicBody({ topic, idPrefix, onRelated, showSummary = true }: TopicBodyProps) {
  const t = useT();
  let n = 0;
  const pending = topic.sections.length === 0;
  return (
    <div className="topic">
      {showSummary && <RichText text={t.l(topic.summary)} className="topic-summary" />}
      {pending && <p className="muted small">{t("ph.pending")}</p>}
      {topic.sections.map((s, i) => (
        <section key={i} className="topic-section" id={`${idPrefix}-sec-${i}`} data-section={i}>
          <h3>
            <span className="sec-num">{i + 1}</span>
            {t.l(s.heading)}
          </h3>
          {s.body && <RichText text={t.l(s.body)} />}
          {(s.equations ?? []).map((eq) => (
            <EquationBlock key={eq.id} eq={eq} n={++n} />
          ))}
          {s.variables && s.variables.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table className="var-table" aria-label={t("ph.variables")}>
                <thead>
                  <tr>
                    <th>{t("ph.symbol")}</th>
                    <th>{t("ph.meaning")}</th>
                    <th>{t("ph.value")}</th>
                    <th>{t("ph.unit")}</th>
                    <th>{t("ph.code")}</th>
                  </tr>
                </thead>
                <tbody>
                  {s.variables.map((v, j) => (
                    <tr key={j}>
                      <td><Tex tex={v.symbol} /></td>
                      <td>{t.l(v.name)}</td>
                      <td className="v">{v.value ?? ""}</td>
                      <td className="v">{v.unit ?? ""}</td>
                      <td className="c">{v.code ? <code className="code-chip">{v.code}</code> : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {s.notes && s.notes.length > 0 && (
            <aside className="notes">
              <div className="notes-title">{t("ph.notes")}</div>
              <ul>
                {s.notes.map((note, j) => (
                  <li key={j}><RichText text={t.l(note)} /></li>
                ))}
              </ul>
            </aside>
          )}
        </section>
      ))}
      {topic.related && topic.related.length > 0 && (
        <div className="related">
          <span className="small muted">{t("ph.related")}:</span>
          {topic.related.map((id) => (
            <button key={id} type="button" className="chip" onClick={() => onRelated?.(id)} data-testid={`related-${id}`}>
              {t.l(PHYSICS_TOPICS[id]?.title) || id}
            </button>
          ))}
        </div>
      )}
      {topic.codeRefs && topic.codeRefs.length > 0 && (
        <div className="related" style={{ marginTop: 10 }}>
          <span className="small muted">{t("ph.sources")}:</span>
          {topic.codeRefs.map((c) => (
            <code key={c} className="code-chip">{c}</code>
          ))}
        </div>
      )}
    </div>
  );
}
