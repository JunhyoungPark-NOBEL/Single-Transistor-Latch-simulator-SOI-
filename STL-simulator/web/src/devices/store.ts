// User device library (Zustand), persisted in localStorage["stl-websim:devices"] and validated on load.
import { create } from "zustand";
import type { DeviceBlock } from "../api/types";
import { BUILTIN_META } from "../state/presets";
import { clone } from "../utils/object";
import { resolveBackGate } from "../params/geometry";
import { LIBRARY_VERSION, geometryFromDevice, isSupportedTechnology, newDeviceId, stochOf, uniqueName, validateDevice, type DeviceStochastic, type LibDevice } from "./library";

export const MAX_USER_DEVICES = 5;

export const DEVICES_KEY = "stl-websim:devices";

/** Validation base: every stored field is type-checked against the reference device block. */
export function validationBase(): { device: DeviceBlock; stochastic: DeviceStochastic } {
  const pr = BUILTIN_META.presets.paper;
  return { device: clone(pr.device), stochastic: stochOf(pr.stochastic) };
}

export function parseStoredDevices(raw: string | null | undefined): LibDevice[] {
  if (!raw) return [];
  try {
    const v = JSON.parse(raw) as unknown;
    const list = Array.isArray(v) ? v : v && typeof v === "object" && Array.isArray((v as { devices?: unknown }).devices) ? (v as { devices: unknown[] }).devices : [];
    const base = validationBase();
    const out: LibDevice[] = [];
    const ids = new Set<string>();
    for (const x of list) {
      const d = validateDevice(x, base);
      if (!d) continue;
      if (ids.has(d.id)) d.id = newDeviceId();
      ids.add(d.id);
      out.push(d);
    }
    return out;
  } catch {
    return [];
  }
}

function load(): LibDevice[] {
  try {
    return parseStoredDevices(typeof localStorage !== "undefined" ? localStorage.getItem(DEVICES_KEY) : null);
  } catch {
    return [];
  }
}
function save(devices: LibDevice[]) {
  try {
    localStorage.setItem(DEVICES_KEY, JSON.stringify({ v: LIBRARY_VERSION, devices }));
  } catch {
    /* storage unavailable — keep in memory */
  }
}

/** Keep displayed library geometry and the submitted simulation geometry in sync. */
function canonicalDevice(d: LibDevice): LibDevice {
  const geometry = geometryFromDevice(d.device);
  return { ...d, geometry, device: { ...d.device, geometry: clone(geometry), vbg: resolveBackGate(d.device.vbg) } };
}

export interface DeviceLibState {
  devices: LibDevice[];
  add: (d: Omit<LibDevice, "id" | "created"> & Partial<Pick<LibDevice, "id" | "created">>) => LibDevice;
  update: (id: string, patch: Partial<Omit<LibDevice, "id" | "builtin">>) => void;
  rename: (id: string, name: string) => void;
  duplicate: (d: LibDevice, suffix: string) => LibDevice;
  remove: (id: string) => void;
  importMany: (ds: LibDevice[]) => number;
}

export const useDeviceLib = create<DeviceLibState>((set, get) => ({
  devices: load(),
  add: (d) => {
    if (get().devices.length >= MAX_USER_DEVICES) throw new Error("Device limit reached (5)");
    const names = get().devices.map((x) => x.name);
    const dev = canonicalDevice({ ...clone(d), id: d.id ?? newDeviceId(), created: d.created ?? new Date().toISOString(), name: uniqueName(d.name.trim() || "Device", names), builtin: undefined, label: undefined });
    set((s) => ({ devices: [...s.devices, dev] }));
    return dev;
  },
  update: (id, patch) => set((s) => ({ devices: s.devices.map((d) => {
    if (d.id !== id) return d;
    // A saved unsupported record is an archive, never an FDSOI calibration target.
    if (!isSupportedTechnology(d.technology)) return d;
    return canonicalDevice({ ...d, ...clone(patch) });
  }) })),
  rename: (id, name) =>
    set((s) => {
      const n = name.trim();
      if (!n) return {};
      const others = s.devices.filter((d) => d.id !== id).map((d) => d.name);
      return { devices: s.devices.map((d) => (d.id === id ? { ...d, name: uniqueName(n, others) } : d)) };
    }),
  duplicate: (d, suffix) => {
    if (get().devices.length >= MAX_USER_DEVICES) throw new Error("Device limit reached (5)");
    const names = get().devices.map((x) => x.name);
    const copy = canonicalDevice({ ...clone(d), id: newDeviceId(), created: new Date().toISOString(), name: uniqueName(`${d.name} ${suffix}`.trim(), names), builtin: undefined, label: undefined });
    set((s) => ({ devices: [...s.devices, copy] }));
    return copy;
  },
  remove: (id) => set((s) => ({ devices: s.devices.filter((d) => d.id !== id) })),
  importMany: (ds) => {
    const cur = get().devices;
    const names = cur.map((x) => x.name);
    const ids = new Set(cur.map((x) => x.id));
    const add = ds.slice(0, Math.max(0, MAX_USER_DEVICES - cur.length)).map((d) => {
      const name = uniqueName(d.name, names);
      names.push(name);
      const id = ids.has(d.id) ? newDeviceId() : d.id;
      ids.add(id);
      return canonicalDevice({ ...clone(d), id, name });
    });
    set({ devices: [...cur, ...add] });
    return add.length;
  },
}));

if (typeof window !== "undefined") {
  useDeviceLib.subscribe((s, prev) => {
    if (s.devices !== prev.devices) save(s.devices);
  });
}
