// What the Details window's "한눈에" guide block shows: set when the window is opened from a parameter
// group (📖) or a field's popover ("물리 자세히 보기 →"), cleared when it is opened from anywhere else
// (a result panel). The block renders only while the window shows `topic`, so a related-topic link
// hides it and Back restores it. Kept apart from the app store (state/store.ts).
import { create } from "zustand";
import type { TopicId } from "../content/physics/types";

export interface GuideTarget {
  topic: TopicId;
  /** Parameter keys to list, in display order (the group's visible fields). */
  keys: string[];
  /** Key to expand, highlight and scroll into view. */
  focus?: string;
  /** Sidebar group id (lead sentence: PARAM_GUIDE[group] or the bucket picture). */
  group?: string;
  /** Changes on every open, so the same target re-highlights. */
  nonce: number;
}

interface GuideTargetState {
  target: GuideTarget | null;
  set: (t: Omit<GuideTarget, "nonce">) => void;
  clear: () => void;
}

export const useGuideTarget = create<GuideTargetState>((set, get) => ({
  target: null,
  set: (t) => set({ target: { ...t, keys: [...t.keys], nonce: (get().target?.nonce ?? 0) + 1 } }),
  clear: () => {
    if (get().target) set({ target: null });
  },
}));
