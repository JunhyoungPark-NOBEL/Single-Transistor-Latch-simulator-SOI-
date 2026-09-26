import type { Layout } from "plotly.js";
import { describe, expect, it } from "vitest";
import { currentAxis, themedLayout } from "./theme";

describe("current scale switching", () => {
  it("invalidates the zoomed y range when the unit changes from log exponents to amperes", () => {
    const log = themedLayout("light", { xaxis: { range: [0, 4] }, yaxis: { ...currentAxis(true, "I"), range: [-12, -3] } });
    const linear = themedLayout("light", { xaxis: { range: [0, 4] }, yaxis: currentAxis(false, "I") });
    expect(log.yaxis?.autorange).toBe(false);
    expect(linear.yaxis?.autorange).toBe(true);
    expect(linear.yaxis?.range).toBeUndefined();
    expect(linear.yaxis?.uirevision).not.toBe(log.yaxis?.uirevision);
    expect(linear.xaxis?.uirevision).toBe(log.xaxis?.uirevision);
  });
  it("respects explicit axis revision and ranges while preserving scale through theme changes", () => {
    const layout: Partial<Layout> = { yaxis: { ...currentAxis(true, "I"), range: [-12, -3] } };
    expect(themedLayout("light", layout).yaxis?.uirevision).toBe(themedLayout("dark", layout).yaxis?.uirevision);
    expect(themedLayout("light", { yaxis: { ...currentAxis(false, "I"), autorange: false, range: [0, 1e-6], uirevision: "custom" } }).yaxis).toMatchObject({ autorange: false, range: [0, 1e-6], uirevision: "custom" });
  });
});
