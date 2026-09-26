import { describe, expect, it } from "vitest";
import { parseBlocks, plainText, tokenize } from "./RichText";
import { renderTex } from "./Tex";

describe("RichText parser", () => {
  it("paragraphs and bullets", () => {
    const b = parseBlocks("First line\ncontinues.\n\n- one\n- two $x$\n\nLast");
    expect(b).toEqual([
      { type: "p", text: "First line continues." },
      { type: "ul", items: ["one", "two $x$"] },
      { type: "p", text: "Last" },
    ]);
  });
  it("mixed paragraph then bullets without blank line", () => {
    expect(parseBlocks("Intro:\n- a\n- b")).toEqual([{ type: "p", text: "Intro:" }, { type: "ul", items: ["a", "b"] }]);
  });
  it("inline tokens", () => {
    expect(tokenize("a $V_{LU}$ **bold $x$** *it* `p[9]` end")).toEqual([
      { t: "text", v: "a " },
      { t: "math", v: "V_{LU}" },
      { t: "text", v: " " },
      { t: "bold", v: "bold $x$" },
      { t: "text", v: " " },
      { t: "italic", v: "it" },
      { t: "text", v: " " },
      { t: "code", v: "p[9]" },
      { t: "text", v: " end" },
    ]);
    // code protects dollars and stars
    expect(tokenize("`a*$b$*`")).toEqual([{ t: "code", v: "a*$b$*" }]);
    // math containing * is not italic
    expect(tokenize("$x^*$ and *y*")).toEqual([{ t: "math", v: "x^*" }, { t: "text", v: " and " }, { t: "italic", v: "y" }]);
  });
  it("plainText", () => {
    expect(plainText("**a** $b$ `c`")).toBe("a b c");
  });
  it("KaTeX fallback never throws", () => {
    expect(renderTex("\\frac{a}{b}")).toContain("katex");
    expect(renderTex("\\frac{a")).toContain("tex-error");
  });
});
