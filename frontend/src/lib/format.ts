// Pure presentation helpers, shared across pages. Kept side-effect-free so they are unit-testable
// without a DOM or a backend.

export type RiskBand = "critical" | "high" | "medium" | "low";

export function riskBand(risk: number): RiskBand {
  if (risk >= 80) return "critical";
  if (risk >= 60) return "high";
  if (risk >= 40) return "medium";
  return "low";
}

const BAND_HEX: Record<RiskBand, string> = {
  critical: "#ef4444",
  high: "#f97316",
  medium: "#eab308",
  low: "#22c55e",
};

export function riskColor(risk: number): string {
  return BAND_HEX[riskBand(risk)];
}

const BAND_CLASSES: Record<RiskBand, string> = {
  critical: "bg-red-50 text-red-700 border-red-200",
  high: "bg-orange-50 text-orange-700 border-orange-200",
  medium: "bg-amber-50 text-amber-700 border-amber-200",
  low: "bg-green-50 text-green-700 border-green-200",
};

export function riskBadgeClasses(risk: number): string {
  return BAND_CLASSES[riskBand(risk)];
}

export function prettyType(anomalyType: string): string {
  return anomalyType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}
