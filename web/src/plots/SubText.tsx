// Renders a dictionary string that uses the Plotly label markup (<sub>…</sub>, <sup>…</sup>) as React
// elements, so panel footers read the same as the plot labels. Only these two tags are recognised; the
// input comes from the app's own dictionary, never from user data.
import { Fragment } from "react";

export function SubText({ text }: { text: string }) {
  const parts = text.split(/(<su[bp]>.*?<\/su[bp]>)/g);
  return (
    <>
      {parts.map((p, i) => {
        const m = /^<(su[bp])>(.*)<\/su[bp]>$/.exec(p);
        if (!m) return <Fragment key={i}>{p}</Fragment>;
        return m[1] === "sub" ? <sub key={i}>{m[2]}</sub> : <sup key={i}>{m[2]}</sup>;
      })}
    </>
  );
}
