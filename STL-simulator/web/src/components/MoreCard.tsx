// Focus layout for result screens: one hero panel plus one tabbed analysis card ("MoreCard") in the
// 간단히 layout; today's `.grid` with every panel in the 모두 보기 layout (state/layout.ts).
//
// MoreCard renders only the active tab's panel, inside MoreCardContext { embedded: true } (the Panel then
// drops its card border and hides its h3 visually: the tab label is the title). Tabs carry status dots
// (running | error | stale) so a failure in a hidden tab stays visible. The active tab is kept per scope in
// localStorage "stl-websim:more:<scope>"; at ≤ 760 px the tab strip becomes a <select>.
import { createContext, Fragment, useEffect, useRef, useState, useSyncExternalStore, type KeyboardEvent, type ReactNode } from "react";
import { create } from "zustand";
import { useT } from "../i18n";
import { UX } from "../i18n/strings.ux";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { useIsAll } from "../state/layout";
import type { ResultEntry } from "../state/store";
import "./more.css";

// ---------------------------------------------------------------- context read by Panel
export interface MoreCardCtx {
  /** True inside a MoreCard tab: the panel renders borderless with a visually hidden h3. */
  embedded: boolean;
}
export const MoreCardContext = createContext<MoreCardCtx>({ embedded: false });
const EMBEDDED: MoreCardCtx = { embedded: true };
/**
 * Right end of the MoreCard tab row: an embedded Panel portals its status badge, 📖 and ⋯ there, so the tab
 * row doubles as the panel's title row (null outside a MoreCard, or before the row has mounted).
 */
export const MoreSlotContext = createContext<HTMLElement | null>(null);

// ---------------------------------------------------------------- tab status
export type TabStatus = "running" | "error" | "stale";

/** The fields of a result entry that decide its status (ResultEntry, or Panel's looser entry prop). */
export type StatusEntry = Pick<ResultEntry, "status"> & { data?: unknown; dataKey?: string };

/**
 * Status dot of a tab whose panel shows `entry`, with the same rules as Panel's badges and overlay:
 * queued/running → "running"; error → "error"; data computed from another payload than `currentKey`
 * (and not re-running) → "stale"; otherwise null (idle, done and current, cancelled).
 */
export function entryStatus(entry: StatusEntry | null | undefined, currentKey?: string | null): TabStatus | null {
  if (!entry) return null;
  if (entry.status === "running" || entry.status === "queued") return "running";
  if (entry.status === "error") return "error";
  if (currentKey && entry.data !== undefined && entry.dataKey && entry.dataKey !== currentKey) return "stale";
  return null;
}

const RANK: Record<TabStatus, number> = { stale: 1, error: 2, running: 3 };
/** Most urgent of several statuses (running > error > stale), for a tab that depends on more than one entry. */
export function mergeStatus(...s: (TabStatus | null | undefined)[]): TabStatus | null {
  let best: TabStatus | null = null;
  for (const v of s) if (v && (!best || RANK[v] > RANK[best])) best = v;
  return best;
}

// ---------------------------------------------------------------- active tab per scope (persisted)
const tabKey = (scope: string) => `stl-websim:more:${scope}`;

function readTab(scope: string): string | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage.getItem(tabKey(scope)) : null;
  } catch {
    return null;
  }
}

/** Chosen tab per scope (null: nothing chosen yet → defaultTab). Loaded from localStorage once per scope. */
const useChosen = create<Record<string, string | null>>(() => ({}));
const loaded = new Map<string, string | null>();
function chosenOnce(scope: string): string | null {
  if (!loaded.has(scope)) loaded.set(scope, readTab(scope));
  return loaded.get(scope) ?? null;
}

/** Selects (and persists) a tab of a MoreCard, e.g. "show the charge balance at this V_D" from another panel. */
export function selectMoreTab(scope: string, id: string) {
  loaded.set(scope, id);
  useChosen.setState({ [scope]: id });
  try {
    localStorage.setItem(tabKey(scope), id);
  } catch {
    /* storage unavailable (private mode, blocked) — ignore */
  }
}

// ---------------------------------------------------------------- phone breakpoint (same as app.css)
const NARROW = "(max-width: 760px)";
function subscribeNarrow(cb: () => void) {
  const mq = typeof window !== "undefined" ? window.matchMedia?.(NARROW) : undefined;
  mq?.addEventListener?.("change", cb);
  return () => mq?.removeEventListener?.("change", cb);
}
const isNarrow = () => (typeof window !== "undefined" ? !!window.matchMedia?.(NARROW).matches : false);
const useNarrow = () => useSyncExternalStore(subscribeNarrow, isNarrow, () => false);

// ---------------------------------------------------------------- components
export interface MoreTab {
  /** Panel id without the "panel-" prefix (e.g. "components"); the tab's testid is `more-tab-<id>`. */
  id: string;
  /** Tab label (already localised). */
  label: string;
  /** The panel, e.g. <ComponentsPanel />. Rendered only while its tab is active (simple layout). */
  panel: ReactNode;
  /** Status dot, usually `entryStatus(entry, currentKey)`. */
  status?: TabStatus | null;
  /** Leave the tab out (e.g. a stochastic-only panel in deterministic mode) in both layouts. */
  hidden?: boolean;
  /** Tooltip of the tab label. */
  title?: string;
  /** Label for the phone <select>, whose options cannot show subscripts (else `label` with "_" dropped). */
  plainLabel?: string;
}

export interface MoreCardProps {
  /** Unique per screen: testid `more-<scope>`, select `more-select-<scope>`, storage "stl-websim:more:<scope>". */
  scope: string;
  tabs: MoreTab[];
  /** Tab shown until the user picks one (also when the stored tab is hidden now). May change with the data. */
  defaultTab: string;
  className?: string;
}

export function MoreCard({ scope, tabs, defaultTab, className }: MoreCardProps) {
  const t = useT();
  const narrow = useNarrow();
  const stored = useChosen((s) => s[scope]);
  const chosen = stored !== undefined ? stored : chosenOnce(scope);
  const visible = tabs.filter((x) => !x.hidden);
  const active = visible.find((x) => x.id === chosen) ?? visible.find((x) => x.id === defaultTab) ?? visible[0];
  const activeId = active?.id;
  const btns = useRef(new Map<string, HTMLButtonElement>());
  const [slot, setSlot] = useState<HTMLElement | null>(null);

  // a newly shown tab mounts its Plotly chart; a window resize lets every chart re-lay out to its box
  const prev = useRef(activeId);
  useEffect(() => {
    if (prev.current === activeId) return;
    prev.current = activeId;
    const fire = () => window.dispatchEvent(new Event("resize"));
    if (typeof window.requestAnimationFrame === "function") window.requestAnimationFrame(fire);
    else setTimeout(fire, 16);
  }, [activeId]);

  if (!active) return null;
  const statusText = (s: TabStatus) => t.l(UX[`status.${s}` as const]);
  const domId = (id: string) => `more-${scope}-${id}`;

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = visible.findIndex((x) => x.id === activeId);
    const n = visible.length;
    const next = e.key === "ArrowRight" ? (i + 1) % n : e.key === "ArrowLeft" ? (i - 1 + n) % n : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    const id = visible[next].id;
    selectMoreTab(scope, id);
    btns.current.get(id)?.focus();
  };

  return (
    <section className={`more-card${className ? ` ${className}` : ""}`} data-testid={`more-${scope}`}>
      <div className="more-head">
        {narrow ? (
          <select className="select more-select" value={active.id} onChange={(e) => selectMoreTab(scope, e.target.value)} aria-label={t.l(UX["more.select"])} data-testid={`more-select-${scope}`}>
            {visible.map((x) => (
              <option key={x.id} value={x.id}>
                {(() => {
                  const l = x.plainLabel ?? x.label.replace(/([A-Za-z])_([A-Za-z0-9]+)/g, "$1$2");
                  return x.status ? `${l} · ${statusText(x.status)}` : l;
                })()}
              </option>
            ))}
          </select>
        ) : (
          <div className="more-tabs" role="tablist" aria-label={t.l(UX["more.tabs"])} onKeyDown={onKey}>
            {visible.map((x) => {
              const on = x.id === active.id;
              return (
                <button
                  key={x.id}
                  ref={(el) => {
                    if (el) btns.current.set(x.id, el);
                    else btns.current.delete(x.id);
                  }}
                  type="button"
                  role="tab"
                  id={`${domId(x.id)}-tab`}
                  className="more-tab"
                  aria-selected={on}
                  aria-controls={`${domId(x.id)}-panel`}
                  tabIndex={on ? 0 : -1}
                  title={x.title}
                  data-status={x.status ?? undefined}
                  data-testid={`more-tab-${x.id}`}
                  onClick={() => selectMoreTab(scope, x.id)}
                >
                  <span className="more-tab-label">
                    {/* "V_G 의존성" → V<sub>G</sub> 의존성 (the phone <select> keeps the plain label) */}
                    <SubText text={subs(x.label)} />
                  </span>
                  {x.status && (
                    <>
                      <span className={`more-dot ${x.status}`} aria-hidden title={statusText(x.status)} />
                      <span className="sr-only"> · {statusText(x.status)}</span>
                    </>
                  )}
                </button>
              );
            })}
          </div>
        )}
        <div className="more-actions" ref={setSlot} />
      </div>
      <div className="more-body" {...(narrow ? {} : { role: "tabpanel", id: `${domId(active.id)}-panel`, "aria-labelledby": `${domId(active.id)}-tab` })}>
        <MoreCardContext.Provider value={EMBEDDED}>
          <MoreSlotContext.Provider value={slot}>{active.panel}</MoreSlotContext.Provider>
        </MoreCardContext.Provider>
      </div>
    </section>
  );
}

export interface FocusLayoutProps {
  /** Testid of the outer element in both layouts (e.g. "panels-deterministic"). */
  testId: string;
  /** MoreCard scope (see MoreCardProps.scope). */
  scope: string;
  /** Hero panel (pass `primary` to its Panel). Omit for a card-only screen (Validation). */
  hero?: ReactNode;
  tabs: MoreTab[];
  defaultTab: string;
  /** Hero and card side by side on wide screens (≥ 1280 px); otherwise stacked, hero first. */
  side?: boolean;
  /**
   * 모두 보기 only: grid order as tab ids, with "hero" for the hero (e.g. ["hero", "components",
   * "charge-balance", "vg"] to keep today's order). Unlisted visible tabs follow in tab order; an unlisted
   * hero goes first. For anything else, branch on `useIsAll()` in the caller.
   */
  allOrder?: string[];
}

/** 간단히: `<div class="focus-grid">{hero}<MoreCard/></div>`; 모두 보기: `<div class="grid">` with every panel. */
export function FocusLayout({ testId, scope, hero, tabs, defaultTab, side, allOrder }: FocusLayoutProps) {
  const all = useIsAll();
  if (all) {
    const visible = tabs.filter((x) => !x.hidden);
    const byId = new Map(visible.map((x) => [x.id, x.panel] as const));
    const order = [...(allOrder ?? [])].filter((id) => (id === "hero" ? hero != null : byId.has(id)));
    if (hero != null && !order.includes("hero")) order.unshift("hero");
    for (const x of visible) if (!order.includes(x.id)) order.push(x.id);
    return (
      <div className="grid" data-testid={testId}>
        {order.map((id) => (
          <Fragment key={id}>{id === "hero" ? hero : byId.get(id)}</Fragment>
        ))}
      </div>
    );
  }
  const cls = `focus-grid${side ? " side" : ""}${hero == null ? " no-hero" : ""}`;
  return (
    <div className={cls} data-testid={testId}>
      {hero}
      <MoreCard scope={scope} tabs={tabs} defaultTab={defaultTab} />
    </div>
  );
}
