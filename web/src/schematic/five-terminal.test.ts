import { describe, expect, it } from "vitest";
import { iKey, parseProbe } from "../api/circuitCustom";
import { BUILTIN_META } from "../state/presets";
import { clone } from "../utils/object";
import { hasErrors, runErc } from "./erc";
import { pinPositions, type SElement, type SchematicDoc, type PinName } from "./model";
import { extractNets, optionalPinConnected } from "./nets";
import { buildRequest, nodeList, validProbeKeys } from "./netlist";
import { defaultDoc, exportDocJson, parseDoc } from "./persist";
import { buildTemplate, TEMPLATE_ORDER } from "./templates";

const part = (id: string, kind: SElement["kind"], x: number, y: number, extra: Partial<SElement> = {}): SElement => ({ id, name: id, kind, x, y, rot: 0, ...extra });
const label = (id: string, name: string, x: number, y: number) => part(id, "LABEL", x, y, { label: name });
const ground = (id: string, x: number, y: number) => part(id, "GND", x, y);
function circuit(): SchematicDoc {
  const doc = defaultDoc("five terminals");
  doc.elements = [
    part("X1", "STL", 0, 0, { stl: { libId: "builtin:paper", name: "Device 1", device: clone(BUILTIN_META.presets.paper.device) } }),
    label("LD", "drain", 0, -40), label("LG", "gate", -40, 0), ground("G0", 0, 40),
    part("VD", "V", 100, 0, { wave: { kind: "dc", value: 1 } }), label("LVD", "drain", 100, -40), ground("GD", 100, 40),
    part("VG", "V", 200, 0, { wave: { kind: "pulse", v1: -3.6, v2: -3, td: 0, tr: 1e-6, tf: 1e-6, pw: 1e-3, per: 2e-3, ncycles: 0 } }), label("LVG", "gate", 200, -40), ground("GG", 200, 40),
  ];
  return doc;
}
function attachBody(doc: SchematicDoc, withR = true) {
  doc.elements.push(label("LB", "body", 50, 10), part("CB", "C", 400, 0, { value: 2e-15 }), label("LCB", "body", 400, -40), ground("GCB", 400, 40));
  if (withR) doc.elements.push(part("RB", "R", 500, 0, { value: 1e9 }), label("LRB", "body", 500, -40), ground("GRB", 500, 40));
}
function compiled(doc: SchematicDoc) {
  const conn = extractNets(doc);
  return { conn, req: buildRequest(doc, conn, "deterministic", null) };
}

describe("five-terminal STL circuit topology", () => {
  it("provides five distinct rotatable pins while preserving D/G/S positions", () => {
    const x = circuit().elements[0];
    expect(pinPositions(x).map(({ pin, x, y }) => [pin, x, y])).toEqual([
      ["d", 0, -40], ["g", -40, 0], ["s", 0, 40], ["bg", -50, -30], ["b", 50, 10],
    ]);
    expect(pinPositions({ ...x, rot: 1, mirror: true }).find((p) => p.pin === "b")).toMatchObject({ x: -10, y: -50 });
  });

  it("omits unconnected BG/B so saved back-gate bias and floating body survive", () => {
    const doc = circuit();
    doc.elements[0].stl!.device.vbg = -0.3;
    const { conn, req } = compiled(doc);
    const stl = req.netlist.elements.find((e) => e.type === "STL")!;
    expect(stl.nodes).toEqual({ d: "drain", g: "gate", s: "0" });
    if (stl.type === "STL") expect(stl.device.vbg).toBe(-0.3);
    expect(nodeList(conn)).toEqual(["0", "drain", "gate"]);
    expect(optionalPinConnected("X1", "b", conn)).toBe(false);
    expect(hasErrors(runErc(doc, conn))).toBe(false);
  });

  it("compiles gate pulse, back-gate sine and external body RC onto distinct nodes", () => {
    const doc = circuit();
    doc.elements.push(label("LBG", "backgate", -50, -30), part("VBG", "V", 300, 0, { wave: { kind: "sine", vo: 0, va: 0.2, freq: 1000, td: 0, theta: 0 } }), label("LVBG", "backgate", 300, -40), ground("GBG", 300, 40));
    attachBody(doc);
    const { conn, req } = compiled(doc);
    expect(req.netlist.elements.find((e) => e.type === "STL")?.nodes).toEqual({ d: "drain", g: "gate", s: "0", bg: "backgate", b: "body" });
    expect(req.netlist.elements.find((e) => e.name === "VG")).toMatchObject({ wave: { kind: "pulse" }, nodes: ["gate", "0"] });
    expect(req.netlist.elements.find((e) => e.name === "VBG")).toMatchObject({ wave: { kind: "sine", freq: 1000 }, nodes: ["backgate", "0"] });
    expect(req.netlist.elements.filter((e) => e.name === "RB" || e.name === "CB").map((e) => e.nodes)).toEqual([["body", "0"], ["body", "0"]]);
    expect(hasErrors(runErc(doc, conn))).toBe(false);
    const kept = buildRequest({ ...doc, save_all: false }, conn, "deterministic", ["I(X1.bg)", "I(X1.b)", "X1.vb", "X1.vbody", "V(body)"]);
    expect(kept.probes).toEqual(["I(X1.bg)", "I(X1.b)", "X1.vb", "X1.vbody", "V(body)"]);
    expect(compiled(parseDoc(JSON.parse(exportDocJson(doc)))!).req).toEqual(req);
  });

  it("accepts a body capacitor alone, but still rejects an undriven insulated back gate", () => {
    const doc = circuit();
    attachBody(doc, false);
    expect(hasErrors(runErc(doc, extractNets(doc)))).toBe(false);
    doc.elements.push(label("LBG", "backgate", -50, -30), part("CBG", "C", 600, 0, { value: 1e-15 }), label("LCBG", "backgate", 600, -40), ground("GCBG", 600, 40));
    expect(runErc(doc, extractNets(doc)).some((e) => e.code === "noDcPath" && e.vars?.node === "backgate")).toBe(true);
  });

  it("never attaches legacy wires to new pin positions during import or re-save", () => {
    const old = circuit();
    old.v = 1;
    old.wires.push({ id: "oldWire", x1: 50, y1: -10, x2: 50, y2: 30 });
    const migrated = parseDoc(old)!;
    expect(migrated.v).toBe(2);
    expect(migrated.elements[0].stlTerminalMode).toBe("legacy3");
    expect(pinPositions(migrated.elements[0]).map((p) => p.pin)).toEqual(["d", "g", "s"]);
    expect(compiled(migrated).req.netlist.elements[0].nodes).toEqual({ d: "drain", g: "gate", s: "0" });
    const again = parseDoc(JSON.parse(exportDocJson(migrated)))!;
    expect(again.elements[0].stlTerminalMode).toBe("legacy3");
    expect(again.wires).toEqual(old.wires);
  });

  it("keeps existing example circuits on their intended G/D/S topology", () => {
    for (const id of TEMPLATE_ORDER) {
      const doc = buildTemplate(id, circuit().elements[0].stl!, id);
      for (const e of compiled(doc).req.netlist.elements) if (e.type === "STL") {
        expect(e.nodes.bg, `${id}: BG must not attach to old wires`).toBeUndefined();
        expect(e.nodes.b, `${id}: B must not attach to old wires`).toBeUndefined();
      }
    }
  });

  it("recognizes back-gate current and separates electrostatic and contact-voltage probes", () => {
    const keys = validProbeKeys(compiled(circuit()).req.netlist.elements);
    for (const pin of ["d", "g", "s", "bg", "b"] as PinName[]) expect(keys.has(`I(X1.${pin})`)).toBe(true);
    expect(parseProbe(iKey("X1", "bg"))).toEqual({ type: "I", el: "X1", terminal: "bg" });
    expect(parseProbe("X1.vb")).toEqual({ type: "state", el: "X1", q: "vb" });
    expect(parseProbe("X1.vbody")).toEqual({ type: "state", el: "X1", q: "vbody" });
  });
});
