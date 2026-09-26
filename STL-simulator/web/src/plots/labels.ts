// Shared typography for React labels and Plotly's supported HTML subset.

const aliases: Record<string, string> = {
  Tsi: "T_Si", TSi: "T_Si", Tox: "T_ox", Tbox: "T_box", Nbody: "N_body",
  VLU: "V_LU", VLD: "V_LD", Vtop: "V_top", Vbottom: "V_bottom",
  Vth: "V_th", VBG: "V_BG", VG: "V_G", VD: "V_D", VB: "V_B", ID: "I_D",
  Iin: "I_in", Cdrain: "C_drain",
};
const compactSymbols = new RegExp(`(^|[^A-Za-z0-9_])(${Object.keys(aliases).join("|")})(?![A-Za-z0-9_])`, "g");

function plainSymbols(text: string, standalone: boolean): string {
  let value = text.replace(compactSymbols, (_, prefix: string, sym: string) => prefix + aliases[sym]);
  value = value.replace(/(\b(?:[VIQRCfh]|[VI]_[A-Za-z]+))\(([tur])\)/g, "$1(<i>$2</i>)");
  // Unsubscripted variables only in unambiguous symbol/equation positions.
  // Unit strings and English words (e.g. "I am", "2 W") are never inferred as math.
  value = value.replace(/(^|[^A-Za-z0-9_>])([VICQRLTWNfhutσβτφ]|Δ[Vt])(?=\s*(?:=|[<>≤≥±]|\())/g, "$1<i>$2</i>");
  value = value.replace(/(^|[·\s])([LW])(?=\s+[−+\-]?\d)/g, "$1<i>$2</i>");
  if (standalone && /^(?:[LWfhutNPσβτφ]|Δ[Vt])$/.test(value.trim())) value = value.replace(/\S+/, "<i>$&</i>");
  value = value.replace(/(^|[^A-Za-z0-9_])(Δ?[A-Za-zσφδτβεηκλμψρ]|[dδ][Vφ])_(\{[^}]+\}|[A-Za-z0-9φτ]+(?:,max|,min)?)/g,
    (_, pre: string, base: string, sub: string) => `${pre}<i>${base}</i><sub>${sub.replace(/^\{|\}$/g, "")}</sub>`);
  return value;
}

/**
 * Math base letters italic, descriptive subscripts upright. Existing <i>/<sub>
 * markup is respected so repeated formatting does not nest tags. HTML tags and
 * Plotly placeholders are preserved, never parsed as physics variables.
 */
export function mathMarkup(text: string): string {
  const tokens = text.split(/(<[^>]*>|%\{[^}]*\})/g);
  let protectedDepth = 0;
  return tokens.map((token, i) => {
    if (/^<\/(?:i|sub|sup|code)>$/.test(token)) { protectedDepth = Math.max(0, protectedDepth - 1); return token; }
    if (/^<(?:i|sub|sup|code)>$/.test(token)) { protectedDepth++; return token; }
    if (token.startsWith("<") || token.startsWith("%{") || protectedDepth) return token;
    let value = plainSymbols(token, tokens.length === 1);
    if (/^<\/(?:i|sub)>$/.test(tokens[i - 1] ?? "")) value = value.replace(/^\(([tur])\)/, "(<i>$1</i>)");
    // A legacy "V<sub>D</sub>" already has the index but still needs its base italic.
    if (tokens[i + 1] === "<sub>") value = value.replace(/(^|[^A-Za-z0-9_])(Δ?[A-Za-zσφδτβεηκλμψρ]|[dδ][Vφ])$/, "$1<i>$2</i>");
    return value;
  }).join("");
}

/** Server-style V_LU, existing markup and compact ID/VD all share one formatter. */
export function subs(text: string): string { return mathMarkup(text); }

/** Number with a typographic minus sign (−1.8), for labels. */
export function signed(v: number, digits: number): string { return v.toFixed(digits).replace("-", "−"); }

/** "Name (unit)" for a server label + unit; dimensionless ("1" or "") gets no parentheses. */
export function withUnit(label: string, unit: string | undefined): string { return unit && unit !== "1" ? `${label} (${unit})` : label; }

/** Clone only presentation fields. Numeric arrays and simulation results retain their identities. */
export function mathPlotLabels<T>(obj: T): T {
  if (!obj || typeof obj !== "object") return obj;
  const out: Record<string, unknown> = { ...(obj as Record<string, unknown>) };
  const labels = new Set(["name", "text", "hovertext", "hovertemplate", "ticktext"]);
  for (const [key, value] of Object.entries(out)) {
    if (labels.has(key)) {
      if (typeof value === "string") out[key] = mathMarkup(value);
      else if (Array.isArray(value)) out[key] = value.map((v: unknown) => typeof v === "string" ? mathMarkup(v) : v);
    } else if (key === "title") {
      out[key] = typeof value === "string" ? mathMarkup(value) : mathPlotLabels(value);
    } else if (/^(?:[xyz]axis\d*|colorbar|legend)$/.test(key)) out[key] = mathPlotLabels(value);
    else if (key === "annotations" && Array.isArray(value)) out[key] = value.map(mathPlotLabels);
  }
  return out as T;
}
