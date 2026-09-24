import { describe, expect, it } from "vitest";
import { columnsToCsv, tracesToCsv } from "./csv";

describe("csv", () => {
  it("pads uneven columns and blanks null/NaN", () => {
    expect(columnsToCsv([{ name: "a", values: [1, 2] }, { name: "b", values: [3, null, NaN] }])).toBe("a,b\n1,3\n2,\n,\n");
  });
  it("one x/y pair per trace, html stripped, duplicate names numbered", () => {
    const csv = tracesToCsv([
      { name: "V<sub>LU</sub>", x: [1, 2], y: [3, 4] },
      { name: "V<sub>LU</sub>", x: [5], y: [6] },
      { name: "empty", x: [], y: [] },
    ]);
    expect(csv.split("\n")[0]).toBe("VLU x,VLU y,VLU (2) x,VLU (2) y");
    expect(csv.split("\n")[2]).toBe("2,4,,");
  });
  it("heatmap as matrix", () => {
    const csv = tracesToCsv([{ type: "heatmap", x: [1, 2], y: [0.1], z: [[5, 6]] }]);
    expect(csv).toBe("y \\ x,1,2\n0.1,5,6\n");
  });
  it("quotes strings with commas", () => {
    expect(columnsToCsv([{ name: "a,b", values: ['x"y'] }])).toBe('"a,b"\n"x""y"\n');
  });
});
