// Renderer for the physics RichText format (web/src/content/physics/types.ts): paragraphs separated by a
// blank line, inline math $…$, **bold**, *italic*, `code`, and bullet lines starting with "- ".
import { Fragment, memo, useMemo, type ReactNode } from "react";
import { renderTex } from "./Tex";

export type Block = { type: "p"; text: string } | { type: "ul"; items: string[] };

export function parseBlocks(text: string): Block[] {
  const out: Block[] = [];
  for (const para of text.replace(/\r\n/g, "\n").split(/\n[ \t]*\n/)) {
    let buf: string[] = [];
    let items: string[] | null = null;
    const flushP = () => {
      if (buf.length) out.push({ type: "p", text: buf.join(" ").trim() });
      buf = [];
    };
    const flushUl = () => {
      if (items && items.length) out.push({ type: "ul", items });
      items = null;
    };
    for (const raw of para.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      if (/^[-•]\s+/.test(line)) {
        flushP();
        items ??= [];
        items.push(line.replace(/^[-•]\s+/, ""));
      } else if (items && /^\s{2,}\S/.test(raw)) {
        items[items.length - 1] += " " + line; // continuation of a bullet
      } else {
        flushUl();
        buf.push(line);
      }
    }
    flushP();
    flushUl();
  }
  return out;
}

export type Token =
  | { t: "text"; v: string }
  | { t: "math"; v: string }
  | { t: "code"; v: string }
  | { t: "bold"; v: string }
  | { t: "italic"; v: string };

const INLINE = /(`[^`]+`)|(\$[^$\n]+\$)|(\*\*(?:.+?)\*\*)|(\*(?!\s)(?:[^*\n]+?)(?<!\s)\*)/g;

export function tokenize(s: string): Token[] {
  const out: Token[] = [];
  let last = 0;
  for (const m of s.matchAll(INLINE)) {
    const i = m.index ?? 0;
    if (i > last) out.push({ t: "text", v: s.slice(last, i) });
    const tok = m[0];
    if (m[1]) out.push({ t: "code", v: tok.slice(1, -1) });
    else if (m[2]) out.push({ t: "math", v: tok.slice(1, -1) });
    else if (m[3]) out.push({ t: "bold", v: tok.slice(2, -2) });
    else out.push({ t: "italic", v: tok.slice(1, -1) });
    last = i + tok.length;
  }
  if (last < s.length) out.push({ t: "text", v: s.slice(last) });
  return out;
}

export function Inline({ text }: { text: string }): ReactNode {
  const toks = tokenize(text);
  return (
    <>
      {toks.map((k, i) => {
        switch (k.t) {
          case "text": return <Fragment key={i}>{k.v}</Fragment>;
          case "code": return <code key={i}>{k.v}</code>;
          case "math": return <span key={i} className="m" dangerouslySetInnerHTML={{ __html: renderTex(k.v, false) }} />;
          case "bold": return <strong key={i}><Inline text={k.v} /></strong>;
          case "italic": return <em key={i}><Inline text={k.v} /></em>;
        }
        return null;
      })}
    </>
  );
}

export const RichText = memo(function RichText({ text, className }: { text: string | undefined; className?: string }) {
  const blocks = useMemo(() => parseBlocks(text ?? ""), [text]);
  if (!text) return null;
  const body = blocks.map((b, i) =>
    b.type === "p" ? (
      <p key={i}><Inline text={b.text} /></p>
    ) : (
      <ul key={i}>{b.items.map((it, j) => <li key={j}><Inline text={it} /></li>)}</ul>
    ),
  );
  return className ? <div className={className}>{body}</div> : <>{body}</>;
});

/** Plain-text version (for search indexing / aria labels). */
export function plainText(s: string): string {
  return s.replace(/\$([^$]+)\$/g, "$1").replace(/\*\*|`/g, "").replace(/\*/g, "");
}
