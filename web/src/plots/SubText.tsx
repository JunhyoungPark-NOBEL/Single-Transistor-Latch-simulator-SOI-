// React renderer for the small, trusted label vocabulary shared with Plotly.
// Unknown HTML is displayed literally; this never injects HTML into the page.
import { createElement, Fragment, type ReactNode } from "react";
import { mathMarkup } from "./labels";
import "../math/typography.css";

function elements(text: string): ReactNode[] {
  const result: ReactNode[] = [];
  const pattern = /<(i|sub|sup)>([\s\S]*?)<\/\1>/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index! > cursor) result.push(text.slice(cursor, match.index));
    result.push(createElement(match[1], { key: match.index, ...(match[1] === "i" ? { className: "math-var" } : {}) }, ...elements(match[2])));
    cursor = match.index! + match[0].length;
  }
  if (cursor < text.length) result.push(text.slice(cursor));
  return result;
}

export function SubText({ text }: { text: string }) {
  return <span className="math-label">{elements(mathMarkup(text)).map((part, i) => <Fragment key={i}>{part}</Fragment>)}</span>;
}
