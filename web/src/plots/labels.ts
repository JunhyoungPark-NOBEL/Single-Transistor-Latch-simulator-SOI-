// Plot-label helpers shared by the result panels. Axis titles, legend names and annotations come from the
// axis.* block of the i18n dictionary (web/src/i18n/strings.brand.ts); these helpers only format values and
// turn server-provided labels ("V_LU (drain node)", "ΔQ_B") into Plotly HTML subscripts.

/** "V_LU" → "V<sub>LU</sub>", "t_stop" → "t<sub>stop</sub>" (Plotly HTML); other text is left as is. */
export function subs(text: string): string {
  return text.replace(/([A-Za-zΔσφδ]+)_(\{[^}]+\}|[A-Za-z0-9]+)/g, (_, a: string, b: string) => `${a}<sub>${b.replace(/^\{|\}$/g, "")}</sub>`);
}

/** Number with a typographic minus sign (−1.8), for labels. */
export function signed(v: number, digits: number): string {
  return v.toFixed(digits).replace("-", "−");
}

/** "Name (unit)" for a server label + unit; dimensionless ("1" or "") gets no parentheses. */
export function withUnit(label: string, unit: string | undefined): string {
  return unit && unit !== "1" ? `${label} (${unit})` : label;
}
