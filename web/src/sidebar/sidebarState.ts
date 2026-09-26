// Sidebar disclosure state (kept out of the app store): the per-group open state ("stl-websim:groups",
// unchanged format), the "고급 설정" disclosure ("stl-websim:adv-open") and the per-group "고급 항목 n개"
// disclosures ("stl-websim:field-adv"). Every storage access is wrapped: private mode or blocked storage
// just means nothing is remembered.
import { create } from "zustand";
import { parseOpenState } from "../state/persist";

export const GROUP_STATE_KEY = "stl-websim:groups";
export const ADV_OPEN_KEY = "stl-websim:adv-open";
export const FIELD_ADV_KEY = "stl-websim:field-adv";

function read(key: string): string | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage.getItem(key) : null;
  } catch {
    return null;
  }
}
function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

export interface SidebarUi {
  /** Saved open state per group id (absent = the layout's default). */
  groups: Record<string, boolean>;
  /** "고급 설정" disclosure in the 간단히 layout (persisted). */
  advOpen: boolean;
  /** Same disclosure in the 모두 보기 layout (open by default, not persisted). */
  advOpenAll: boolean;
  /** "고급 항목 n개" per group id (persisted). */
  fieldAdv: Record<string, boolean>;
  setGroupOpen: (id: string, open: boolean) => void;
  setAdvOpen: (open: boolean, all: boolean) => void;
  setFieldAdv: (id: string, open: boolean) => void;
}

const bools = (raw: string | null): Record<string, boolean> => parseOpenState(raw);

export const useSidebarUi = create<SidebarUi>((set, get) => ({
  groups: bools(read(GROUP_STATE_KEY)),
  advOpen: read(ADV_OPEN_KEY) === "1",
  advOpenAll: true,
  fieldAdv: bools(read(FIELD_ADV_KEY)),
  setGroupOpen: (id, open) => {
    const groups = { ...get().groups, [id]: open };
    set({ groups });
    write(GROUP_STATE_KEY, JSON.stringify(groups));
  },
  setAdvOpen: (open, all) => {
    if (all) return set({ advOpenAll: open });
    set({ advOpen: open });
    write(ADV_OPEN_KEY, open ? "1" : "0");
  },
  setFieldAdv: (id, open) => {
    const fieldAdv = { ...get().fieldAdv, [id]: open };
    set({ fieldAdv });
    write(FIELD_ADV_KEY, JSON.stringify(fieldAdv));
  },
}));
