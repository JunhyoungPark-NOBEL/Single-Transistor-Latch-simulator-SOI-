// Interactive SVG canvas: grid with snap, pan (drag empty space / middle mouse / Space+drag), wheel
// zoom about the cursor, place parts, orthogonal wires with junction dots, select / move (attached wires
// follow) / box-select, probe mode (click a node → V(node), a part → its current), ERC highlights and
// time-cursor annotations (node voltages, element currents with direction arrows).
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as RPE, type ReactNode } from "react";
import { useT } from "../i18n";
import { iKey, vKey } from "../api/circuitCustom";
import { useIsAll } from "../state/layout";
import type { ErcItem } from "./erc";
import { addWires, docBounds, lPath, moveItems } from "./edit";
import { DEFAULT_CMP, DEFAULT_MOS, DEFAULT_DIODE, DEFAULT_BJT, distToSeg, elementBox, GRID, newId, nextName, onWireInterior, pinPositions, rotatePt, snap, type Pt, type SchematicDoc, type SElement, type Wire } from "./model";
import type { Connectivity, NetInfo } from "./nets";
import { pinId } from "./nets";
import { fmtSI } from "./si";
import { useSch, stlRefFor } from "./store";
import { ElementView } from "./Symbols";
import { defaultWave } from "./waves";

export interface Annotations {
  /** net name → volts */
  v: Map<string, number>;
  /** signal key → value (I(R1), I(X1.d), X1.u …) */
  sig: Map<string, number>;
  t: number;
}

type Hover = { kind: "el"; el: SElement; pin?: string } | { kind: "wire"; w: Wire } | null;

type Drag =
  | { kind: "pan"; sx: number; sy: number; vx: number; vy: number; moved: boolean; clearOnClick: boolean }
  | { kind: "move"; start: Pt; ids: string[]; dx: number; dy: number }
  | { kind: "marquee"; start: Pt; cur: Pt; additive: boolean };

const TOL = 6;

function defaultsFor(kind: SElement["kind"]): Partial<SElement> {
  switch (kind) {
    case "R":
      return { value: 1e3 };
    case "C":
      return { value: 1e-15 };
    case "V":
      return { wave: defaultWave("dc", 1) };
    case "I":
      return { wave: defaultWave("dc", 1e-9) };
    case "LABEL":
      return { label: "" };
    case "MOS": return { mos: { ...DEFAULT_MOS } };
    case "D": return { diode: { ...DEFAULT_DIODE } };
    case "BJT": return { bjt: { ...DEFAULT_BJT } };
    case "CMP":
      return { cmp: { ...DEFAULT_CMP } };
    default:
      return {};
  }
}

export function makeElement(doc: SchematicDoc, kind: SElement["kind"], x: number, y: number, rot: SElement["rot"], mirror: boolean, stlChoice: string): SElement {
  const el: SElement = { id: newId(), kind, name: kind === "GND" || kind === "LABEL" ? "" : nextName(doc.elements, kind), x, y, rot, ...defaultsFor(kind) };
  if (mirror) el.mirror = true;
  if (kind === "STL") {
    el.stl = stlRefFor(stlChoice);
    el.light = null;
  }
  if (kind === "LABEL") {
    const used = new Set(doc.elements.filter((e) => e.kind === "LABEL").map((e) => e.label));
    let k = 1;
    while (used.has(`n${k}`)) k++;
    el.label = `n${k}`;
  }
  return el;
}

function hitElement(doc: SchematicDoc, p: Pt, tol: number): SElement | null {
  for (let i = doc.elements.length - 1; i >= 0; i--) {
    const e = doc.elements[i];
    const b = elementBox(e);
    if (p.x >= b[0] - tol && p.x <= b[2] + tol && p.y >= b[1] - tol && p.y <= b[3] + tol) return e;
  }
  return null;
}
function hitWire(doc: SchematicDoc, p: Pt, tol: number): Wire | null {
  let best: Wire | null = null;
  let bd = tol;
  for (const w of doc.wires) {
    const d = distToSeg(p.x, p.y, w);
    if (d <= bd) {
      bd = d;
      best = w;
    }
  }
  return best;
}
function nearestPin(el: SElement, p: Pt): string | undefined {
  let best: string | undefined;
  let bd = Infinity;
  for (const q of pinPositions(el)) {
    const d = Math.hypot(q.x - p.x, q.y - p.y);
    if (d < bd) {
      bd = d;
      best = q.pin;
    }
  }
  return best;
}

export function Canvas({ conn, ann, height }: { conn: Connectivity; erc: ErcItem[]; ann: Annotations | null; height: number }) {
  const t = useT();
  const all = useIsAll();
  const doc = useSch((s) => s.doc);
  const selection = useSch((s) => s.selection);
  const tool = useSch((s) => s.tool);
  const view = useSch((s) => s.view);
  const fitNonce = useSch((s) => s.fitNonce);
  const highlight = useSch((s) => s.highlight);
  const stlChoice = useSch((s) => s.stlChoice);
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const space = useRef(false);
  const [ghost, setGhost] = useState<Pt | null>(null);
  const [hover, setHover] = useState<Hover>(null);
  const [preview, setPreview] = useState<SchematicDoc | null>(null);
  const [marquee, setMarquee] = useState<[Pt, Pt] | null>(null);
  const [wireFrom, setWireFrom] = useState<{ p: Pt; vFirst: boolean | null } | null>(null);
  const [mouse, setMouse] = useState<{ x: number; y: number } | null>(null);
  const [panning, setPanning] = useState(false);
  const [flashIds, setFlashIds] = useState<string[]>([]);

  const shown = preview ?? doc;
  const sel = useMemo(() => new Set(selection), [selection]);

  // ---- viewport helpers
  const toWorld = useCallback(
    (clientX: number, clientY: number): Pt => {
      const r = svgRef.current!.getBoundingClientRect();
      const v = useSch.getState().view;
      return { x: (clientX - r.left - v.x) / v.k, y: (clientY - r.top - v.y) / v.k };
    },
    [],
  );
  const fit = useCallback(() => {
    const el = svgRef.current;
    if (!el) return;
    const d = useSch.getState().doc;
    const b = docBounds(d, elementBox);
    const W = el.clientWidth || 800;
    const H = el.clientHeight || 500;
    if (!b) {
      useSch.getState().setView({ x: W / 2, y: H / 2, k: 1 });
      return;
    }
    const pad = 72; // room for the time-cursor annotations around the parts
    const k = Math.max(0.3, Math.min(1.6, Math.min((W - 2 * pad) / Math.max(40, b[2] - b[0]), (H - 2 * pad) / Math.max(40, b[3] - b[1]))));
    useSch.getState().setView({ k, x: W / 2 - ((b[0] + b[2]) / 2) * k, y: H / 2 - ((b[1] + b[3]) / 2) * k });
  }, []);
  useLayoutEffect(() => {
    fit();
  }, [fitNonce, fit]);

  // wheel zoom (non-passive so the page does not scroll)
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const v = useSch.getState().view;
      const k = Math.max(0.2, Math.min(4, v.k * Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015))));
      const cx = e.clientX - r.left;
      const cy = e.clientY - r.top;
      useSch.getState().setView({ k, x: cx - ((cx - v.x) * k) / v.k, y: cy - ((cy - v.y) * k) / v.k });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Space held → pan
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && document.activeElement === wrapRef.current) {
        space.current = true;
        e.preventDefault();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") space.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // ERC highlight flash
  useEffect(() => {
    if (!highlight) return;
    setFlashIds(highlight.ids);
    const id = setTimeout(() => setFlashIds([]), 1800);
    return () => clearTimeout(id);
  }, [highlight]);

  // wire tool reset when the tool changes
  useEffect(() => {
    if (tool.kind !== "wire") setWireFrom(null);
  }, [tool]);
  useEffect(() => {
    const onCancel = () => setWireFrom(null);
    window.addEventListener("sch-cancel-wire", onCancel);
    return () => window.removeEventListener("sch-cancel-wire", onCancel);
  }, []);

  const isConnectionPoint = useCallback(
    (p: Pt): boolean => {
      const d = useSch.getState().doc;
      for (const e of d.elements) for (const q of pinPositions(e)) if (q.x === p.x && q.y === p.y) return true;
      for (const w of d.wires) if ((w.x1 === p.x && w.y1 === p.y) || (w.x2 === p.x && w.y2 === p.y) || onWireInterior(w, p.x, p.y)) return true;
      return false;
    },
    [],
  );

  // ---- pointer handling
  const onPointerDown = (e: RPE<SVGSVGElement>) => {
    wrapRef.current?.focus({ preventScroll: true });
    const st = useSch.getState();
    const w = toWorld(e.clientX, e.clientY);
    const sp = { x: snap(w.x), y: snap(w.y) };
    const tol = TOL / st.view.k;
    if (e.button === 1 || (e.button === 0 && space.current)) {
      e.preventDefault();
      drag.current = { kind: "pan", sx: e.clientX, sy: e.clientY, vx: st.view.x, vy: st.view.y, moved: false, clearOnClick: false };
      setPanning(true);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button === 2) {
      // right click: finish a wire / leave a placing tool
      if (st.tool.kind === "wire" && wireFrom) setWireFrom(null);
      else if (st.tool.kind !== "select") st.setTool({ kind: "select" });
      return;
    }
    if (e.button !== 0) return;
    switch (st.tool.kind) {
      case "place": {
        const el = makeElement(st.doc, st.tool.el, sp.x, sp.y, st.tool.rot, st.tool.mirror, st.stlChoice);
        st.commit((d) => ({ ...d, elements: [...d.elements, el] }));
        st.select([el.id]);
        return;
      }
      case "wire": {
        if (!wireFrom) {
          setWireFrom({ p: sp, vFirst: null });
          return;
        }
        const vFirst = wireFrom.vFirst ?? Math.abs(sp.y - wireFrom.p.y) > Math.abs(sp.x - wireFrom.p.x);
        const segs = lPath(wireFrom.p, sp, vFirst);
        if (segs.length) st.commit((d) => addWires(d, segs));
        if (e.detail >= 2 || (segs.length && isConnectionPoint(sp) && (sp.x !== wireFrom.p.x || sp.y !== wireFrom.p.y))) setWireFrom(null);
        else setWireFrom({ p: sp, vFirst: null });
        return;
      }
      case "probe": {
        const hit = probeTarget(st.doc, conn, w, tol);
        if (hit) st.addTrace(hit);
        return;
      }
      case "select": {
        const el = hitElement(st.doc, w, tol);
        const wi = el ? null : hitWire(st.doc, w, tol);
        const id = el?.id ?? wi?.id;
        if (id) {
          let ids = st.selection;
          if (e.shiftKey || e.ctrlKey || e.metaKey) ids = ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id];
          else if (!ids.includes(id)) ids = [id];
          st.select(ids);
          if (ids.includes(id)) {
            drag.current = { kind: "move", start: sp, ids, dx: 0, dy: 0 };
            e.currentTarget.setPointerCapture(e.pointerId);
          }
          return;
        }
        if (e.shiftKey) {
          drag.current = { kind: "marquee", start: w, cur: w, additive: true };
          e.currentTarget.setPointerCapture(e.pointerId);
          return;
        }
        drag.current = { kind: "pan", sx: e.clientX, sy: e.clientY, vx: st.view.x, vy: st.view.y, moved: false, clearOnClick: true };
        e.currentTarget.setPointerCapture(e.pointerId);
        return;
      }
    }
  };

  const onPointerMove = (e: RPE<SVGSVGElement>) => {
    const st = useSch.getState();
    const w = toWorld(e.clientX, e.clientY);
    const sp = { x: snap(w.x), y: snap(w.y) };
    const r = svgRef.current!.getBoundingClientRect();
    setMouse({ x: e.clientX - r.left, y: e.clientY - r.top });
    const d = drag.current;
    if (d?.kind === "pan") {
      const dx = e.clientX - d.sx;
      const dy = e.clientY - d.sy;
      if (Math.abs(dx) + Math.abs(dy) > 3) {
        d.moved = true;
        setPanning(true);
      }
      if (d.moved) st.setView({ ...st.view, x: d.vx + dx, y: d.vy + dy });
      return;
    }
    if (d?.kind === "move") {
      const dx = sp.x - d.start.x;
      const dy = sp.y - d.start.y;
      if (dx !== d.dx || dy !== d.dy) {
        d.dx = dx;
        d.dy = dy;
        setPreview(dx || dy ? moveItems(st.doc, d.ids, dx, dy) : null);
      }
      return;
    }
    if (d?.kind === "marquee") {
      d.cur = w;
      setMarquee([d.start, w]);
      return;
    }
    if (st.tool.kind === "place" || st.tool.kind === "wire") setGhost(sp);
    else setGhost(null);
    if (wireFrom && wireFrom.vFirst === null && (sp.x !== wireFrom.p.x || sp.y !== wireFrom.p.y)) setWireFrom({ ...wireFrom, vFirst: Math.abs(sp.y - wireFrom.p.y) > Math.abs(sp.x - wireFrom.p.x) });
    // hover
    const tol = TOL / st.view.k;
    const el = hitElement(st.doc, w, tol);
    if (el) setHover({ kind: "el", el, pin: el.kind === "STL" ? nearestPin(el, w) : undefined });
    else {
      const wi = hitWire(st.doc, w, tol);
      setHover(wi ? { kind: "wire", w: wi } : null);
    }
  };

  const onPointerUp = (e: RPE<SVGSVGElement>) => {
    const st = useSch.getState();
    const d = drag.current;
    drag.current = null;
    setPanning(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* not captured */
    }
    if (!d) return;
    if (d.kind === "pan" && !d.moved && d.clearOnClick) st.select([]);
    if (d.kind === "move" && (d.dx || d.dy)) st.commit((doc0) => moveItems(doc0, d.ids, d.dx, d.dy));
    if (d.kind === "marquee") {
      const x0 = Math.min(d.start.x, d.cur.x);
      const x1 = Math.max(d.start.x, d.cur.x);
      const y0 = Math.min(d.start.y, d.cur.y);
      const y1 = Math.max(d.start.y, d.cur.y);
      const inside = (x: number, y: number) => x >= x0 && x <= x1 && y >= y0 && y <= y1;
      const ids = [
        ...st.doc.elements.filter((el) => {
          const b = elementBox(el);
          return inside(b[0], b[1]) && inside(b[2], b[3]);
        }).map((el) => el.id),
        ...st.doc.wires.filter((w) => inside(w.x1, w.y1) && inside(w.x2, w.y2)).map((w) => w.id),
      ];
      st.select(d.additive ? [...new Set([...st.selection, ...ids])] : ids);
      setMarquee(null);
    }
    setPreview(null);
  };

  const onDoubleClick = () => {
    const st = useSch.getState();
    if (st.tool.kind === "wire") setWireFrom(null);
    else if (st.tool.kind === "select" && st.selection.length === 1) window.dispatchEvent(new CustomEvent("sch-focus-inspector"));
  };

  // ---- derived drawing data
  const hoverNet: NetInfo | undefined = useMemo(() => {
    if (!hover) return undefined;
    if (hover.kind === "wire") return conn.wireNet.get(hover.w.id);
    if (hover.el.kind === "LABEL" || hover.el.kind === "GND") return conn.pinNet.get(pinId(hover.el.id, "o"));
    return undefined;
  }, [hover, conn]);
  const hoverNetWires = useMemo(() => new Set(hoverNet?.wires.map((w) => w.id) ?? []), [hoverNet]);
  const unconnected = conn.unconnected.filter((p) => p.el.kind !== "GND" && p.el.kind !== "LABEL");

  const ghostEl: SElement | null =
    tool.kind === "place" && ghost ? { id: "ghost", kind: tool.el, name: tool.el === "GND" || tool.el === "LABEL" ? "" : nextName(doc.elements, tool.el), x: ghost.x, y: ghost.y, rot: tool.rot, mirror: tool.mirror, ...defaultsFor(tool.el), ...(tool.el === "LABEL" ? { label: "…" } : {}), ...(tool.el === "STL" ? { stl: stlRefFor(stlChoice) } : {}) } : null;
  const wirePreview = tool.kind === "wire" && wireFrom && ghost ? lPath(wireFrom.p, ghost, wireFrom.vFirst ?? Math.abs(ghost.y - wireFrom.p.y) > Math.abs(ghost.x - wireFrom.p.x)) : [];

  const cursor = panning ? "grabbing" : tool.kind === "place" || tool.kind === "wire" ? "crosshair" : tool.kind === "probe" ? "copy" : hover ? "pointer" : "grab";
  const g = GRID * view.k;
  const gridStep = g < 7 ? g * 5 : g;
  const major = GRID * 10 * view.k;
  const hint = tool.kind === "place" ? t("schematic.hint.place") : tool.kind === "wire" ? t("schematic.hint.wire") : tool.kind === "probe" ? t("schematic.hint.probe") : t("schematic.hint.select");

  return (
    <div ref={wrapRef} className={`sch-canvas-wrap tool-${tool.kind}`} tabIndex={0} data-testid="sch-canvas" aria-label={t("schematic.canvas.aria")} style={{ height }}>
      <svg
        ref={svgRef}
        className="sch-svg"
        style={{
          cursor,
          backgroundSize: `${gridStep}px ${gridStep}px, ${major}px ${major}px, ${major}px ${major}px`,
          backgroundPosition: `${view.x}px ${view.y}px, ${view.x}px ${view.y}px, ${view.x}px ${view.y}px`,
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          setGhost(null);
          setHover(null);
          setMouse(null);
        }}
        onDoubleClick={onDoubleClick}
        onContextMenu={(e) => e.preventDefault()}
        data-testid="sch-svg"
        data-k={view.k.toFixed(3)}
      >
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {/* wires */}
          <g className="sch-wires">
            {shown.wires.map((w) => {
              const net = conn.wireNet.get(w.id)?.name;
              const cls = ["sch-wire", sel.has(w.id) && "sel", hoverNetWires.has(w.id) && "hov", flashIds.includes(w.id) && "flag"].filter(Boolean).join(" ");
              return (
                <g key={w.id} data-wire={w.id} data-net={net}>
                  <line x1={w.x1} y1={w.y1} x2={w.x2} y2={w.y2} className="sch-wire-hit" />
                  <line x1={w.x1} y1={w.y1} x2={w.x2} y2={w.y2} className={cls} />
                </g>
              );
            })}
          </g>
          {/* junctions */}
          {!preview &&
            conn.junctions.map((p) => <circle key={`j${p.x},${p.y}`} cx={p.x} cy={p.y} r={3.6} className="sch-junction" />)}
          {/* elements */}
          {shown.elements.map((el) => (
            <ElementView
              key={el.id}
              el={el}
              selected={sel.has(el.id)}
              hovered={hover?.kind === "el" && hover.el.id === el.id}
              flagged={flashIds.includes(el.id)}
              probing={tool.kind === "probe" && hover?.kind === "el" && hover.el.id === el.id}
            />
          ))}
          {/* unconnected pins */}
          {!preview &&
            unconnected.map((p) => <rect key={`u${p.el.id}${p.pin}`} x={p.x - 3} y={p.y - 3} width={6} height={6} className="sch-open-pin" />)}
          {/* selection outlines */}
          {shown.elements
            .filter((el) => sel.has(el.id))
            .map((el) => {
              const b = elementBox(el);
              return <rect key={`s${el.id}`} x={b[0] - 6} y={b[1] - 6} width={b[2] - b[0] + 12} height={b[3] - b[1] + 12} rx={6} className="sch-sel-box" />;
            })}
          {/* annotations at the time cursor */}
          {ann && !preview && <AnnotationLayer doc={doc} conn={conn} ann={ann} />}
          {/* ghosts */}
          {ghostEl && <ElementView el={ghostEl} ghost />}
          {wirePreview.map((w) => <line key={w.id} x1={w.x1} y1={w.y1} x2={w.x2} y2={w.y2} className="sch-wire ghost" />)}
          {tool.kind === "wire" && ghost && <circle cx={ghost.x} cy={ghost.y} r={3} className="sch-cross" />}
          {wireFrom && <circle cx={wireFrom.p.x} cy={wireFrom.p.y} r={3.4} className="sch-junction ghost" />}
          {marquee && (
            <rect x={Math.min(marquee[0].x, marquee[1].x)} y={Math.min(marquee[0].y, marquee[1].y)} width={Math.abs(marquee[1].x - marquee[0].x)} height={Math.abs(marquee[1].y - marquee[0].y)} className="sch-marquee" />
          )}
        </g>
      </svg>
      {/* empty canvas: examples to start from (hidden while a tool is active, so the centre stays placeable) */}
      {doc.elements.length === 0 && tool.kind === "select" && (
        <div className="sch-empty">
          <strong>{t("schematic.empty.title")}</strong>
          <span>{t("schematic.empty.drawHint")}</span>
          <div className="sch-empty-keys"><kbd>W</kbd> {t("schematic.tool.wire")} <kbd>R</kbd> {t("schematic.tool.R")} <kbd>Esc</kbd> {t("schematic.tool.select")}</div>
        </div>
      )}
      <HoverTip hover={hover} net={hoverNet} conn={conn} ann={ann} mouse={mouse} />
      {/* 간단히: the generic select-tool line only until the first part is placed; tool help stays */}
      {(all || tool.kind !== "select" || doc.elements.length === 0) && (
        <div className="sch-hint" aria-live="polite">
          {hint}
        </div>
      )}
    </div>
  );
}

/** Signal key the probe tool adds for a click at world point p (node voltage or element current). */
function probeTarget(doc: SchematicDoc, conn: Connectivity, p: Pt, tol: number): string | null {
  const el = hitElement(doc, p, tol);
  if (el) {
    if (el.kind === "GND") return null;
    if (el.kind === "LABEL") {
      const n = conn.pinNet.get(pinId(el.id, "o"));
      return n ? vKey(n.name) : null;
    }
    // near a pin (outside the body) → that node's voltage
    for (const q of pinPositions(el)) {
      if (Math.hypot(q.x - p.x, q.y - p.y) <= tol * 0.9) {
        const n = conn.pinNet.get(pinId(el.id, q.pin));
        if (n) return vKey(n.name);
      }
    }
    if (el.kind === "STL" || el.kind === "MOS") {
      const pin = nearestPin(el, p);
      return iKey(el.name, pin === "s" ? "s" : pin === "g" ? "g" : "d");
    }
    if (el.kind === "CMP") return `${el.name}.bit`;
    if (el.kind === "BJT") { const pin = nearestPin(el, p); return iKey(el.name, pin === "b" ? "b" : pin === "e" ? "e" : "c"); }
    return iKey(el.name);
  }
  const w = hitWire(doc, p, tol);
  if (w) {
    const n = conn.wireNet.get(w.id);
    return n ? vKey(n.name) : null;
  }
  return null;
}

const fmtV = (v: number) => (Math.abs(v) >= 0.1 || v === 0 ? `${v.toFixed(3)} V` : fmtSI(v, "V", 3));
const fmtA = (v: number) => fmtSI(Math.abs(v) < 1e-21 ? 0 : v, "A", 3);

function AnnotationLayer({ doc, conn, ann }: { doc: SchematicDoc; conn: Connectivity; ann: Annotations }) {
  const out: ReactNode[] = [];
  for (const n of conn.nets) {
    if (n.ground || !n.pins.length) continue;
    const v = ann.v.get(n.name);
    if (v === undefined) continue;
    const onLabel = n.allPins.some((p) => p.el.kind === "LABEL" && p.x === n.anchor.x && p.y === n.anchor.y);
    const txt = fmtV(v);
    const w = 10 + txt.length * 6.6;
    // on a label: below-right of the label point (keeps the wires visible); else centred above the wire
    const x = onLabel ? n.anchor.x + 6 : n.anchor.x - w / 2;
    const y = onLabel ? n.anchor.y + 6 : n.anchor.y - 22;
    out.push(
      <g key={`v${n.id}`} className="sch-ann-v" data-ann-node={n.name}>
        <rect x={x} y={y} width={w} height={16} rx={8} />
        <text x={x + w / 2} y={y + 11.5} textAnchor="middle">
          {txt}
        </text>
      </g>,
    );
  }
  for (const el of doc.elements) {
    if (el.kind === "GND" || el.kind === "LABEL" || el.kind === "CMP") continue;
    const pins = pinPositions(el);
    let key = iKey(el.name);
    let a = pins[0];
    let b = pins[1];
    let len = 12;
    let offset = 20;
    if (["STL", "MOS", "BJT"].includes(el.kind)) {
      key = iKey(el.name, el.kind === "BJT" ? "c" : "d");
      // along the drain lead, pointing into the drain for positive current
      const top = rotatePt(0, -40, el.rot, el.mirror);
      const inner = rotatePt(0, -18, el.rot, el.mirror);
      a = { ...pins[0], x: el.x + top.x, y: el.y + top.y };
      b = { ...pins[0], x: el.x + inner.x, y: el.y + inner.y };
      len = 7;
      offset = 12;
    }
    const I = ann.sig.get(key);
    if (I === undefined) continue;
    const L = Math.hypot(b.x - a.x, b.y - a.y) || 1;
    const u = { x: (b.x - a.x) / L, y: (b.y - a.y) / L };
    // the value sits on the side away from the part's name/value text
    const td = rotatePt(1, 0, el.rot, el.mirror);
    const nrm = el.kind === "STL" ? { x: -td.x, y: -td.y } : { x: -td.x, y: -td.y };
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const c = { x: mid.x + nrm.x * offset, y: mid.y + nrm.y * offset };
    const s = Math.abs(I) < 1e-21 ? 0 : Math.sign(I);
    const p0 = { x: c.x - u.x * len * s, y: c.y - u.y * len * s };
    const p1 = { x: c.x + u.x * len * s, y: c.y + u.y * len * s };
    const head = s !== 0 ? `M${p1.x - u.x * s * 6 + nrm.x * 4} ${p1.y - u.y * s * 6 + nrm.y * 4} L${p1.x} ${p1.y} L${p1.x - u.x * s * 6 - nrm.x * 4} ${p1.y - u.y * s * 6 - nrm.y * 4}` : "";
    const tx = c.x + nrm.x * 8;
    const ty = c.y + nrm.y * 8;
    const anchor = Math.abs(nrm.x) > 0.5 ? (nrm.x < 0 ? "end" : "start") : "middle";
    out.push(
      <g key={`i${el.id}`} className="sch-ann-i" data-ann-el={el.name}>
        {s !== 0 && <path d={`M${p0.x} ${p0.y} L${p1.x} ${p1.y} ${head}`} />}
        <text x={tx} y={ty + (Math.abs(nrm.y) > 0.5 ? (nrm.y > 0 ? 10 : -4) : 4)} textAnchor={anchor}>
          {fmtA(Math.abs(I))}
        </text>
      </g>,
    );
  }
  return <g className="sch-ann">{out}</g>;
}

function HoverTip({ hover, net, conn, ann, mouse }: { hover: Hover; net?: NetInfo; conn: Connectivity; ann: Annotations | null; mouse: { x: number; y: number } | null }) {
  const t = useT();
  if (!hover || !mouse) return null;
  const rows: [string, string][] = [];
  let title = "";
  if (hover.kind === "wire" || (hover.kind === "el" && (hover.el.kind === "LABEL" || hover.el.kind === "GND"))) {
    const n = hover.kind === "wire" ? conn.wireNet.get(hover.w.id) : net;
    if (!n) return null;
    title = n.ground ? "0 (GND)" : n.name;
    const v = ann?.v.get(n.name);
    if (v !== undefined) rows.push([vKey(n.name), fmtV(v)]);
  } else if (hover.kind === "el") {
    const el = hover.el;
    title = el.name;
    const nets = pinPositions(el).map((p) => `${p.pin === "p" ? "+" : p.pin === "n" ? "−" : p.pin.toUpperCase()}: ${conn.pinNet.get(pinId(el.id, p.pin))?.name ?? "—"}`);
    rows.push([t("schematic.insp.nodes"), nets.join(" · ")]);
    if (ann) {
      const keys = el.kind === "MOS" ? [iKey(el.name, "d"), iKey(el.name, "g"), iKey(el.name, "s")] : el.kind === "BJT" ? [iKey(el.name, "c"), iKey(el.name, "b"), iKey(el.name, "e")] : el.kind === "STL" ? [iKey(el.name, "d"), iKey(el.name, "g"), iKey(el.name, "s"), `${el.name}.u`, `${el.name}.r`, `${el.name}.q_b`] : el.kind === "CMP" ? [`${el.name}.bit`, iKey(el.name)] : [iKey(el.name)];
      for (const k of keys) {
        const v = ann.sig.get(k);
        if (v === undefined) continue;
        rows.push([k, k.startsWith("I(") ? fmtA(v) : k.endsWith("q_b") ? fmtSI(v, "C", 3) : k.endsWith(".bit") ? String(v) : fmtV(v)]);
      }
    }
  }
  if (!rows.length && !title) return null;
  return (
    <div className="sch-tip" style={{ left: mouse.x + 14, top: mouse.y + 14 }} role="tooltip">
      <div className="sch-tip-title">
        {title}
        {ann && <span className="muted"> · {t("schematic.res.at", { t: fmtSI(ann.t, "s", 4) })}</span>}
      </div>
      {rows.map(([k, v]) => (
        <div key={k} className="sch-tip-row">
          <span>{k}</span>
          <span className="mono">{v}</span>
        </div>
      ))}
    </div>
  );
}
