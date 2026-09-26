import { describe, expect, it } from "vitest";
import { comparisonRows, pairedSpeedup, type BenchmarkCase } from "./catalog";
import { computeLocation, rangeLabel, secondsLabel } from "./format";
const sample = (model: BenchmarkCase["model"], seconds: number, supported = true): BenchmarkCase => ({ id: model, comparison_id: "csvm", kind: "circuit", family: "circuit_csvm", group: "standard", label: {ko: "CSVM", en:"CSVM"}, payload:{}, model, supported, timings: {median_s:seconds,min_s:seconds*.9,max_s:seconds*1.1,p25_s:seconds,p75_s:seconds,repeats:5,samples_s:[],first_call_s:seconds+1} });
describe("performance presentation", () => {
  it("compares only paired supported models", () => {
    const rows = comparisonRows([sample("detailed", 1), sample("simple", .1), {...sample("mixed", .5), comparison_id:"mixed"}]);
    expect(rows).toHaveLength(2);
    expect(pairedSpeedup(rows[0])).toBe(10);
    expect(pairedSpeedup(rows[1])).toBeNull();
    expect(pairedSpeedup(comparisonRows([sample("detailed", 1), sample("simple", .1, false)])[0])).toBeNull();
  });
  it("distinguishes local PC from remote same-origin hosting", () => {
    expect(computeLocation("http://127.0.0.1:8000", true)).toBe("이 PC");
    expect(computeLocation("https://lab.kaist.ac.kr", true)).toBe("계산 서버");
  });
  it("formats small measurements without false zero precision", () => {
    expect(secondsLabel(.0001)).toBe("< 1 ms");
    expect(rangeLabel({low_s:.03,seconds:.05,high_s:.09})).toBe("30–90 ms");
    expect(rangeLabel({low_s:null,seconds:null,high_s:null})).toBe("—");
  });
});
