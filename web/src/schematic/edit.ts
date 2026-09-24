// Pure editing operations on a schematic document: add, move (wires attached to moved pins follow and
// are re-routed orthogonally), rotate, mirror, delete, duplicate and orthogonal wire paths.
import { clone } from "../utils/object";
import { newId, nextName, pinPositions, ptKey, type Pt, type Rot, type SchematicDoc, type SElement, type Wire } from "./model";

/** Orthogonal L path from a to b (horizontal first unless `verticalFirst`). */
export function lPath(a: Pt, b: Pt, verticalFirst = false): Wire[] {
  if (a.x === b.x && a.y === b.y) return [];
  if (a.x === b.x || a.y === b.y) return [{ id: newId("w"), x1: a.x, y1: a.y, x2: b.x, y2: b.y }];
  const c = verticalFirst ? { x: a.x, y: b.y } : { x: b.x, y: a.y };
  return [
    { id: newId("w"), x1: a.x, y1: a.y, x2: c.x, y2: c.y },
    { id: newId("w"), x1: c.x, y1: c.y, x2: b.x, y2: b.y },
  ];
}

/** Drop zero-length wires and exact duplicates (either direction). */
export function cleanWires(wires: Wire[]): Wire[] {
  const seen = new Set<string>();
  const out: Wire[] = [];
  for (const w of wires) {
    if (w.x1 === w.x2 && w.y1 === w.y2) continue;
    const a = ptKey(w.x1, w.y1);
    const b = ptKey(w.x2, w.y2);
    const k = a < b ? `${a}|${b}` : `${b}|${a}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(w);
  }
  return out;
}

/** Re-attach wire end points that sat on moved points; non-orthogonal results become an L. */
function reattach(wires: Wire[], map: Map<string, Pt>, skip: Set<string>): Wire[] {
  const out: Wire[] = [];
  for (const w of wires) {
    if (skip.has(w.id)) {
      out.push(w);
      continue;
    }
    const a = map.get(ptKey(w.x1, w.y1));
    const b = map.get(ptKey(w.x2, w.y2));
    if (!a && !b) {
      out.push(w);
      continue;
    }
    const na = a ?? { x: w.x1, y: w.y1 };
    const nb = b ?? { x: w.x2, y: w.y2 };
    if (na.x === nb.x || na.y === nb.y) {
      out.push({ ...w, x1: na.x, y1: na.y, x2: nb.x, y2: nb.y });
      continue;
    }
    // keep the fixed end's direction, then turn towards the moved end
    const horiz = w.y1 === w.y2;
    const fixed = a && !b ? nb : na;
    const moved = a && !b ? na : nb;
    const corner = horiz ? { x: moved.x, y: fixed.y } : { x: fixed.x, y: moved.y };
    out.push({ ...w, x1: fixed.x, y1: fixed.y, x2: corner.x, y2: corner.y });
    out.push({ id: newId("w"), x1: corner.x, y1: corner.y, x2: moved.x, y2: moved.y });
  }
  return out;
}

function pinMap(before: SElement[], after: SElement[]): Map<string, Pt> {
  const m = new Map<string, Pt>();
  const byId = new Map(after.map((e) => [e.id, e]));
  for (const e of before) {
    const n = byId.get(e.id);
    if (!n) continue;
    const pa = pinPositions(e);
    const pb = pinPositions(n);
    pa.forEach((p, i) => {
      if (p.x !== pb[i].x || p.y !== pb[i].y) m.set(ptKey(p.x, p.y), { x: pb[i].x, y: pb[i].y });
    });
  }
  return m;
}

/** Move the selected elements and wires by (dx, dy); attached wires follow. */
export function moveItems(doc: SchematicDoc, ids: string[], dx: number, dy: number): SchematicDoc {
  if (!dx && !dy) return doc;
  const sel = new Set(ids);
  const moved = doc.elements.filter((e) => sel.has(e.id));
  const elements = doc.elements.map((e) => (sel.has(e.id) ? { ...e, x: e.x + dx, y: e.y + dy } : e));
  const map = pinMap(moved, elements);
  const selWires = new Set(doc.wires.filter((w) => sel.has(w.id)).map((w) => w.id));
  for (const w of doc.wires)
    if (selWires.has(w.id)) {
      map.set(ptKey(w.x1, w.y1), { x: w.x1 + dx, y: w.y1 + dy });
      map.set(ptKey(w.x2, w.y2), { x: w.x2 + dx, y: w.y2 + dy });
    }
  const wires = reattach(
    doc.wires.map((w) => (selWires.has(w.id) ? { ...w, x1: w.x1 + dx, y1: w.y1 + dy, x2: w.x2 + dx, y2: w.y2 + dy } : w)),
    map,
    selWires,
  );
  return { ...doc, elements, wires: cleanWires(wires) };
}

function transformItems(doc: SchematicDoc, ids: string[], fn: (e: SElement) => SElement): SchematicDoc {
  const sel = new Set(ids);
  const targets = doc.elements.filter((e) => sel.has(e.id) && e.kind !== "GND");
  if (!targets.length) return doc;
  const elements = doc.elements.map((e) => (sel.has(e.id) && e.kind !== "GND" ? fn(e) : e));
  const map = pinMap(targets, elements);
  return { ...doc, elements, wires: cleanWires(reattach(doc.wires, map, new Set())) };
}

export const rotateItems = (doc: SchematicDoc, ids: string[]) => transformItems(doc, ids, (e) => ({ ...e, rot: (((e.rot + 1) % 4) as Rot) }));
export const mirrorItems = (doc: SchematicDoc, ids: string[]) => transformItems(doc, ids, (e) => ({ ...e, mirror: !e.mirror }));

export function deleteItems(doc: SchematicDoc, ids: string[]): SchematicDoc {
  const sel = new Set(ids);
  if (!doc.elements.some((e) => sel.has(e.id)) && !doc.wires.some((w) => sel.has(w.id))) return doc;
  return { ...doc, elements: doc.elements.filter((e) => !sel.has(e.id)), wires: doc.wires.filter((w) => !sel.has(w.id)) };
}

/** Copy the selection (elements, and wires whose both ends touch the selection or that are selected). */
export function duplicateItems(doc: SchematicDoc, ids: string[], dx = 40, dy = 40): { doc: SchematicDoc; ids: string[] } {
  const sel = new Set(ids);
  const els = doc.elements.filter((e) => sel.has(e.id));
  const pts = new Set(els.flatMap(pinPositions).map((p) => ptKey(p.x, p.y)));
  const wires = doc.wires.filter((w) => sel.has(w.id) || (pts.has(ptKey(w.x1, w.y1)) && pts.has(ptKey(w.x2, w.y2))));
  if (!els.length && !wires.length) return { doc, ids };
  const all = [...doc.elements];
  const newEls: SElement[] = [];
  for (const e of els) {
    const c: SElement = { ...clone(e), id: newId(), x: e.x + dx, y: e.y + dy };
    if (c.kind !== "GND" && c.kind !== "LABEL") c.name = nextName(all, c.kind);
    all.push(c);
    newEls.push(c);
  }
  const newWires = wires.map((w) => ({ id: newId("w"), x1: w.x1 + dx, y1: w.y1 + dy, x2: w.x2 + dx, y2: w.y2 + dy }));
  return { doc: { ...doc, elements: all, wires: [...doc.wires, ...newWires] }, ids: [...newEls.map((e) => e.id), ...newWires.map((w) => w.id)] };
}

export function addWires(doc: SchematicDoc, segs: Wire[]): SchematicDoc {
  if (!segs.length) return doc;
  return { ...doc, wires: cleanWires([...doc.wires, ...segs]) };
}

/** Bounding box of everything (world units), or null for an empty document. */
export function docBounds(doc: SchematicDoc, box: (e: SElement) => [number, number, number, number]): [number, number, number, number] | null {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const e of doc.elements) {
    const b = box(e);
    x0 = Math.min(x0, b[0]);
    y0 = Math.min(y0, b[1]);
    x1 = Math.max(x1, b[2]);
    y1 = Math.max(y1, b[3]);
  }
  for (const w of doc.wires) {
    x0 = Math.min(x0, w.x1, w.x2);
    y0 = Math.min(y0, w.y1, w.y2);
    x1 = Math.max(x1, w.x1, w.x2);
    y1 = Math.max(y1, w.y1, w.y2);
  }
  return Number.isFinite(x0) ? [x0, y0, x1, y1] : null;
}
