// Physics-content interface shared by the content modules (src/content/physics/*) and the UI.
// Owned by the physics-content work package; the UI only imports these types and PHYSICS_TOPICS.

/** Bilingual string. `ko` is the default UI language. */
export interface L10n {
  ko: string;
  en: string;
}

/**
 * Rich text: paragraphs separated by a blank line, inline math `$...$` (KaTeX, no `$$`),
 * `**bold**`, `*italic*`, inline code with backticks, and bullet lines starting with "- ".
 * Nothing else (no HTML, no headings, no links).
 */
export type RichText = L10n;

export interface Equation {
  /** Stable id, unique across all topics, e.g. "eq-charge-balance". Used for cross references. */
  id: string;
  /** Short caption shown above/beside the equation. */
  label?: L10n;
  /** KaTeX display-mode source (no surrounding $$). Must render with katex.renderToString(..., {displayMode:true, throwOnError:true}). */
  tex: string;
  /** Optional one-paragraph explanation under the equation (RichText). */
  note?: RichText;
  /** Where it is implemented, e.g. "photo_mean.py · components()". */
  code?: string;
}

export interface VariableRow {
  /** KaTeX inline source for the symbol, e.g. "\\tau_{\\mathrm{bulk}}". */
  symbol: string;
  name: L10n;
  /** Calibrated/default value as display text (already formatted), e.g. "0.927 µs". */
  value?: string;
  unit?: string;
  /** Code reference, e.g. "p[1]" or "COX_F". */
  code?: string;
}

export interface TopicSection {
  heading: L10n;
  body?: RichText;
  equations?: Equation[];
  variables?: VariableRow[];
  /** Assumptions / caveats / open questions shown as a callout list. */
  notes?: RichText[];
}

export type TopicId =
  | "overview"
  | "electrostatics"
  | "impact-ionization"
  | "btbt-gidl"
  | "channel"
  | "bjt-transport"
  | "charge-balance"
  | "photo"
  | "stochastic-events"
  | "first-passage"
  | "local-states"
  | "sweep-mc"
  | "circuit-element"
  | "parameters"
  | "design-map"
  | "open-problems"
  | "validation"
  | "numerics";

export interface PhysicsTopic {
  id: TopicId;
  title: L10n;
  /** 1–2 sentence summary shown at the top of the compact window. */
  summary: RichText;
  /** Emoji-free short tag list, e.g. ["Eq. 1", "deterministic"]. */
  tags?: string[];
  sections: TopicSection[];
  related?: TopicId[];
  /** Source files the topic is derived from (relative to engine/ or server/). */
  codeRefs?: string[];
}
