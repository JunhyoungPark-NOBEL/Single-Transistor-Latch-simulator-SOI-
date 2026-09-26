import { useCallback } from "react";
import type { L10n } from "../content/physics/types";
import { useStore } from "../state/store";
import { STRINGS, type StrKey } from "./strings";

export type Lang = "ko" | "en";
export type { StrKey };

export function translate(lang: Lang, key: StrKey, vars?: Record<string, string | number>): string {
  let s: string = STRINGS[key]?.[lang] ?? String(key);
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
  return s;
}

/** Hook: `const t = useT(); t("run")`, plus `t.l(l10n)` for bilingual content objects. */
export function useT() {
  const lang = useStore((s) => s.lang);
  const t = useCallback((key: StrKey, vars?: Record<string, string | number>) => translate(lang, key, vars), [lang]);
  return Object.assign(t, { lang, l: (x: L10n | undefined | null) => (x ? x[lang] || x.en || x.ko : "") });
}
export type T = ReturnType<typeof useT>;
