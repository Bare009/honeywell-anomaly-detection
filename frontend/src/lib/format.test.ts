import { describe, expect, it } from "vitest";
import {
  formatPercent,
  formatValue,
  prettyType,
  riskBand,
  riskColor,
} from "./format";

describe("risk bands", () => {
  it("maps risk scores to the right band", () => {
    expect(riskBand(90)).toBe("critical");
    expect(riskBand(70)).toBe("high");
    expect(riskBand(45)).toBe("medium");
    expect(riskBand(10)).toBe("low");
  });

  it("gives a distinct colour per band", () => {
    const colours = new Set([riskColor(90), riskColor(70), riskColor(45), riskColor(10)]);
    expect(colours.size).toBe(4);
  });
});

describe("formatting helpers", () => {
  it("humanizes anomaly type names", () => {
    expect(prettyType("low_and_slow_exfil")).toBe("Low And Slow Exfil");
  });

  it("formats percentages and handles null", () => {
    expect(formatPercent(0.955)).toBe("95.5%");
    expect(formatPercent(null)).toBe("—");
  });

  it("formats mixed values", () => {
    expect(formatValue(3)).toBe("3");
    expect(formatValue(1.23456)).toBe("1.235");
    expect(formatValue(null)).toBe("—");
    expect(formatValue("India")).toBe("India");
  });
});
