// The engine's validation strings as shown in the table: no code ids, Korean words in the Korean UI,
// typographic minus signs (exponents untouched).
import { describe, expect, it } from "vitest";
import { checkText } from "./ValidationTab";

describe("validation check text", () => {
  it("names the base model instead of the code id gate_mean", () => {
    expect(checkText("ko", "확장 항이 모두 0이면 gate_mean과 동일")).toBe("확장 항이 모두 0이면 기본 모델과 같음");
    expect(checkText("ko", "identical to gate_mean (≤ 1e-12 V)")).toBe("기본 모델과 동일 (≤ 1e-12 V)");
    expect(checkText("en", "identical to gate_mean (≤ 1e-12 V)")).toBe("identical to the base model (≤ 1e-12 V)");
    expect(checkText("ko", "동적 MC 100회 스윕 (seed 2026092920)")).toBe("동적 MC 100회 스윕 (시드 2026092920)");
  });
  it("Korean words for the common English ones; σ for SD", () => {
    expect(checkText("ko", "mean V_LU 3.6442 V, SD 8.03 mV")).toBe("평균 V_LU 3.6442 V, σ 8.03 mV");
    expect(checkText("ko", "means ±10 mV, SD_LU ±10 mV, SD_LD ±3 mV")).toBe("평균 ±10 mV, σ_LU ±10 mV, σ_LD ±3 mV");
    expect(checkText("en", "mean V_LU 3.6442 V, SD 8.03 mV")).toBe("mean V_LU 3.6442 V, SD 8.03 mV");
  });
  it("typographic minus before digits, not inside exponents", () => {
    expect(checkText("en", "V_G = -3.9061 … -0.8100 V")).toBe("V_G = −3.9061 … −0.8100 V");
    expect(checkText("en", "max |ΔV_D| = 0.00e-12 V")).toBe("max |ΔV_D| = 0.00e-12 V");
  });
});
