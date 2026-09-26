import { describe, expect, it } from "vitest";
import { fmtDuration, fmtmV, fmtSI, fmtSig, fmtV, nearlyEqual, parseNumber, siPrefix, toInputString } from "./format";

describe("format", () => {
  it("SI prefixes", () => {
    expect(siPrefix(3.2e-12)).toEqual([1e-12, "p"]);
    expect(siPrefix(4.5e-9)[1]).toBe("n");
    expect(siPrefix(2.5e-5)[1]).toBe("µ");
    expect(siPrefix(0.012)[1]).toBe("m");
    expect(siPrefix(1)[1]).toBe("");
    expect(siPrefix(0)[1]).toBe("");
  });
  it("fmtSI", () => {
    expect(fmtSI(3.2e-12, "A")).toBe("3.20 pA");
    expect(fmtSI(1.488987834334872e-11, "A")).toBe("14.9 pA");
    expect(fmtSI(2.5e-5, "A")).toBe("25.0 µA");
    expect(fmtSI(0.0123, "V")).toBe("12.3 mV");
    expect(fmtSI(null, "A")).toBe("—");
    expect(fmtSI(NaN, "A")).toBe("—");
    expect(fmtSI(0.5, "1")).toBe("0.500");
  });
  it("fmtSig / fmtV / fmtmV / fmtDuration", () => {
    expect(fmtSig(3.70372, 4)).toBe("3.704");
    expect(fmtSig(0, 3)).toBe("0");
    expect(fmtSig(1.2e-7, 3)).toBe("1.20e-7");
    expect(fmtV(3.7036889854)).toBe("3.704 V");
    expect(fmtmV(0.11895650192675203)).toBe("119.0 mV");
    expect(fmtDuration(0.4622)).toBe("462 ms");
    expect(fmtDuration(14.3)).toBe("14.3 s");
    expect(fmtDuration(75)).toBe("1 min 15 s");
  });
  it("input strings round-trip", () => {
    expect(toInputString(153.39035678526572)).toBe("153.39");
    expect(toInputString(9.266283807625294e-7 * 1e6)).toBe("0.926628");
    expect(toInputString(1e-12)).toBe("1e-12");
    expect(toInputString(2026092920)).toBe("2.02609e9");
    expect(toInputString(0)).toBe("0");
    expect(parseNumber(" 1,5 ")).toBe(1.5);
    expect(parseNumber("−2")).toBe(-2);
    expect(parseNumber("1e-3")).toBe(0.001);
    expect(parseNumber("abc")).toBeNull();
    expect(parseNumber("")).toBeNull();
    expect(parseNumber("-")).toBeNull();
  });
  it("nearlyEqual", () => {
    expect(nearlyEqual(1, 1 + 1e-12)).toBe(true);
    expect(nearlyEqual(1, 1.001)).toBe(false);
    expect(nearlyEqual(0, 0)).toBe(true);
    expect(nearlyEqual("a", "a")).toBe(true);
  });
});
