import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { uprightTexSubscripts } from "./typography";
import { mathMarkup, mathPlotLabels } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { renderTex } from "../components/Tex";
import { PHYSICS_TOPICS } from "../content/physics";
import { GROUPS } from "../params/schema";

const upright = (s: string) => `_{\\htmlClass{math-upright-sub}{\\mathrm{${s}}}}`;

describe("scientific typography", () => {
  it("formats only variable letters, preserving prose, units and Plotly placeholders", () => {
    expect(mathMarkup("Drain current I<sub>D</sub> (A)")).toBe("Drain current <i>I</i><sub>D</sub> (A)");
    expect(mathMarkup("Tsi (nm) · Nbody (cm⁻³)")).toBe("<i>T</i><sub>Si</sub> (nm) · <i>N</i><sub>body</sub> (cm⁻³)");
    expect(mathMarkup("t = %{x:.3f} ms<br>V_D = %{y:.3f} V")).toBe("<i>t</i> = %{x:.3f} ms<br><i>V</i><sub>D</sub> = %{y:.3f} V");
    expect(mathMarkup("I am ready · 3 V · 2 W · 10 nm · FDSOI")).toBe("I am ready · 3 V · 2 W · 10 nm · FDSOI");
    expect(mathMarkup("VD(t) / I(t) (V)")).toBe("<i>V</i><sub>D</sub>(<i>t</i>) / <i>I</i>(<i>t</i>) (V)");
    expect(mathMarkup("V<sub>D</sub>(t)")).toBe("<i>V</i><sub>D</sub>(<i>t</i>)");
    expect(mathMarkup("L 500 nm · W 200 nm")).toBe("<i>L</i> 500 nm · <i>W</i> 200 nm");
  });
  it("is idempotent and safely renders React markup", () => {
    const text = mathMarkup("V_LU − V_LD · Iin");
    expect(mathMarkup(text)).toBe(text);
    expect(renderToStaticMarkup(<SubText text="T_Si" />)).toContain('<i class="math-var">T</i><sub>Si</sub>');
    expect(renderToStaticMarkup(<SubText text='<img src=x onerror="alert(1)">' />)).toContain("&lt;img");
  });
  it("styles balanced TeX indices without changing bases, powers, escaped underscores or prose", () => {
    expect(uprightTexSubscripts("V_G + T_{Si}^{2}")).toBe(`V${upright("G")} + T${upright("Si")}^{2}`);
    expect(uprightTexSubscripts(String.raw`\tau_\phi + \text{file_name} + x\_y`)).toBe(`\\tau${upright("\\phi")} + \\text{file_name} + x\\_y`);
    const nested = String.raw`\sigma_{V_{LU}} + I_{\frac{a}{b}}`;
    const value = uprightTexSubscripts(nested);
    expect(value).toContain(`V${upright("LU")}`);
    expect(value).toContain(String.raw`\frac{a}{b}`);
    expect(uprightTexSubscripts(value)).toBe(value);
    expect(renderTex(nested)).not.toContain("tex-error");
  });
  it("keeps valid unbraced macro indices attached to their arguments", () => {
    for (const tex of [String.raw`V_\mathrm{eff}`, String.raw`I_\frac{1}{2}`, String.raw`I_\frac12`, String.raw`x_\sqrt[3]{a}`]) {
      expect(renderTex(tex), tex).not.toContain("tex-error");
      expect(uprightTexSubscripts(tex)).toContain("math-upright-sub");
    }
    expect(uprightTexSubscripts(String.raw`x_\unknown{value}`)).toBe(String.raw`x_\unknown{value}`);
  });
  it("applies upright Greek subscripts while keeping the base math-normal", () => {
    const html = renderTex(String.raw`\tau_\phi`);
    expect(html).toContain('class="mord mathnormal"');
    expect(html).toContain("math-upright-sub");
    expect(html).not.toContain("tex-error");
  });
  it("does not alter numeric arrays, signal identities or caller-owned layout", () => {
    const x = [0, 1]; const y = [2, 3];
    const source = { name: "I_D", x, y, hovertemplate: "I_D = %{y} A", title: { text: "V_D (V)" } };
    const result = mathPlotLabels(source);
    expect(result.x).toBe(x); expect(result.y).toBe(y);
    expect(source.name).toBe("I_D"); expect(source.title.text).toBe("V_D (V)");
    expect(result.name).toBe("<i>I</i><sub>D</sub>");
    expect(result.title.text).toBe("<i>V</i><sub>D</sub> (V)");
  });
  it("keeps every shipped equation and parameter symbol renderable", () => {
    const walk = (value: unknown, key = "") => {
      if (typeof value === "string") {
        const formulas = key === "tex" || key === "symbol" || key === "sym"
          ? [value] : [...value.matchAll(/\$([^$]+)\$/g)].map((m) => m[1]);
        for (const tex of formulas) expect(renderTex(tex), tex).not.toContain("tex-error");
      } else if (Array.isArray(value)) value.forEach((v) => walk(v));
      else if (value && typeof value === "object") Object.entries(value).forEach(([k, v]) => walk(v, k));
    };
    walk(PHYSICS_TOPICS); walk(GROUPS);
  });
});
