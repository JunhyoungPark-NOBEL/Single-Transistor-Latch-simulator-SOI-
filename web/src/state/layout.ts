// Global results layout: "simple" (간단히: one hero plot + a tabbed analysis card, secondary controls behind
// one click) or "all" (모두 보기: today's full grids, tables and sidebar). Kept apart from the app store so
// it has no effect on payloads, persistence versions or the URL hash.
//
// Initial value: the query `?view=all|simple` wins (and is saved), then localStorage, then "simple".
// Never touches location.hash — the app store rewrites the hash (#tab=…&mode=…) on every tab/mode change.
import { create } from "zustand";

export type LayoutMode = "simple" | "all";

export const LAYOUT_STORAGE_KEY = "stl-websim:layout";
/** Query parameter that forces a layout (e2e specs and shared links use `?view=all`). */
export const LAYOUT_QUERY = "view";

const isLayout = (v: unknown): v is LayoutMode => v === "simple" || v === "all";

/** Minimal storage surface (lets the unit test pass a fake or throwing storage). */
type Store = Pick<Storage, "getItem" | "setItem">;

function defaultStorage(): Store | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage : null;
  } catch {
    return null; // access itself can throw (blocked site data)
  }
}

function queryLayout(search: string): LayoutMode | null {
  try {
    const v = new URLSearchParams(search).get(LAYOUT_QUERY);
    return isLayout(v) ? v : null;
  } catch {
    return null;
  }
}

function saveLayout(storage: Store | null, layout: LayoutMode) {
  try {
    storage?.setItem(LAYOUT_STORAGE_KEY, layout);
  } catch {
    /* storage unavailable (private mode, blocked) — ignore */
  }
}

/**
 * Layout at page load (pure apart from saving a query value). `search` is `location.search`; a valid
 * `view` value is written to `storage` so it survives navigation to a URL without the query.
 */
export function initialLayout(search: string, storage: Store | null = defaultStorage()): LayoutMode {
  const q = queryLayout(search);
  if (q) {
    saveLayout(storage, q);
    return q;
  }
  try {
    const v = storage?.getItem(LAYOUT_STORAGE_KEY);
    if (isLayout(v)) return v;
  } catch {
    /* ignore */
  }
  return "simple";
}

/** Keeps an explicit `?view=` in the address bar in step with the toggle (the hash is carried over as is). */
function syncQuery(layout: LayoutMode) {
  try {
    const url = new URL(window.location.href);
    if (!url.searchParams.has(LAYOUT_QUERY) || url.searchParams.get(LAYOUT_QUERY) === layout) return;
    url.searchParams.set(LAYOUT_QUERY, layout);
    window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${window.location.hash}`);
  } catch {
    /* ignore */
  }
}

/** `data-layout` on <html>, for CSS that differs between the two layouts (`:root[data-layout="all"] …`). */
function markRoot(layout: LayoutMode) {
  try {
    if (typeof document !== "undefined") document.documentElement.dataset.layout = layout;
  } catch {
    /* ignore */
  }
}

export interface LayoutState {
  layout: LayoutMode;
  setLayout: (layout: LayoutMode) => void;
}

const start: LayoutMode = typeof window !== "undefined" ? initialLayout(window.location.search) : "simple";
markRoot(start);

/**
 * Hook + store: `const layout = useLayout((s) => s.layout)`, `useLayout.getState().setLayout("all")`.
 * `setLayout` saves to localStorage and updates `?view=` only when the URL already carries it.
 */
export const useLayout = create<LayoutState>((set) => ({
  layout: start,
  setLayout: (layout) => {
    if (!isLayout(layout)) return;
    set({ layout });
    saveLayout(defaultStorage(), layout);
    if (typeof window !== "undefined") syncQuery(layout);
    markRoot(layout);
  },
}));

/** True in the "모두 보기" layout (today's full grids, expanded tables, every sidebar field). */
export const useIsAll = (): boolean => useLayout((s) => s.layout === "all");
