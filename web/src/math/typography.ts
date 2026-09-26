/** Typography only: no equation tokens, numerical inputs or model definitions are changed. */

/** End of a TeX atom (balanced group, control sequence, or one character). */
function atomEnd(tex: string, start: number): number {
  if (tex[start] === "{") {
    let depth = 1;
    for (let i = start + 1; i < tex.length; i++) {
      if (tex[i] === "\\") { i = atomEnd(tex, i) - 1; continue; }
      if (tex[i] === "{") depth++;
      if (tex[i] === "}" && --depth === 0) return i + 1;
    }
    return start; // malformed: let KaTeX report the original error
  }
  if (tex[start] === "\\") {
    const cmd = /^\\(?:[A-Za-z]+|.)/.exec(tex.slice(start));
    return start + (cmd?.[0].length ?? 1);
  }
  return Math.min(start + 1, tex.length);
}


/** Complete a command used as an unbraced index, including its mandatory atoms. */
function subscriptEnd(tex: string, start: number): number {
  let end = atomEnd(tex, start);
  if (tex[start] !== "\\") return end;
  const command = tex.slice(start, end);
  const argumentsNeeded = /^(?:\\frac|\\dfrac|\\tfrac|\\binom|\\overset|\\underset)$/.test(command) ? 2
    : /^(?:\\mathrm|\\mathit|\\mathbf|\\mathsf|\\mathtt|\\mathcal|\\mathbb|\\mathfrak|\\mathnormal|\\text|\\textrm|\\operatorname|\\overline|\\underline|\\bar|\\hat|\\widehat|\\tilde|\\widetilde|\\vec|\\dot|\\ddot|\\sqrt)$/.test(command) ? 1 : 0;
  // Unknown macros with a braced argument retain their original TeX. Treating
  // only the command as an atom could detach its argument and break the formula.
  if (!argumentsNeeded && /^\s*\{/.test(tex.slice(end))) return start;
  if (command === "\\sqrt") {
    const optional = /^\s*\[[^\]]*\]/.exec(tex.slice(end));
    if (optional) end += optional[0].length;
  }
  for (let n = 0; n < argumentsNeeded; n++) {
    while (end < tex.length && /\s/.test(tex[end])) end++;
    const next = atomEnd(tex, end);
    if (next <= end || end >= tex.length) return start;
    end = next;
  }
  return end;
}

/**
 * Make subscript arguments upright while keeping math above the baseline unchanged.
 * A balanced scanner handles nested indices, \frac and escaped braces; prose in
 * \text / \operatorname / URLs is copied as-is, including literal underscores.
 */
export function uprightTexSubscripts(tex: string): string {
  let out = "";
  for (let i = 0; i < tex.length;) {
    if (tex[i] === "\\") {
      const end = atomEnd(tex, i);
      const cmd = tex.slice(i, end);
      out += cmd;
      i = end;
      if (/^\\(?:text|textrm|textsf|texttt|textnormal|operatorname|url|href)$/.test(cmd)) {
        while (/\s/.test(tex[i] ?? "") && i < tex.length) out += tex[i++];
        const end = atomEnd(tex, i);
        if (tex[i] === "{" && end > i) { out += tex.slice(i, end); i = end; }
      }
      continue;
    }
    if (tex[i] !== "_") { out += tex[i++]; continue; }
    let start = i + 1;
    while (start < tex.length && /\s/.test(tex[start])) start++;
    const end = subscriptEnd(tex, start);
    if (end <= start || !tex[start]) { out += tex[i++]; continue; }
    const raw = tex.slice(start, end);
    const inner = raw[0] === "{" ? raw.slice(1, -1) : raw;
    // Idempotent when a label has already passed through the formatter.
    out += inner.startsWith("\\htmlClass{math-upright-sub}")
      ? `_{${inner}}`
      : `_{\\htmlClass{math-upright-sub}{\\mathrm{${uprightTexSubscripts(inner)}}}}`;
    i = end;
  }
  return out;
}
