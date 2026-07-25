import { riskBand, riskColor } from "../lib/format";

// A compact semicircular gauge drawn with plain SVG (no chart lib needed). Shows the risk value,
// its band, and the uncertainty half-width as a lighter arc segment around the needle position.
export default function RiskGauge({
  risk,
  uncertainty = 0,
  size = 160,
}: {
  risk: number;
  uncertainty?: number;
  size?: number;
}) {
  const clamped = Math.max(0, Math.min(100, risk));
  const color = riskColor(clamped);
  const radius = size / 2 - 12;
  const cx = size / 2;
  const cy = size / 2;

  // Map 0..100 onto a 180° arc (pi..0).
  const angle = (value: number) => Math.PI - (value / 100) * Math.PI;
  const point = (value: number, r = radius) => ({
    x: cx + r * Math.cos(angle(value)),
    y: cy - r * Math.sin(angle(value)),
  });

  const arc = (from: number, to: number, r = radius) => {
    const start = point(from, r);
    const end = point(to, r);
    // The gauge spans a 180deg arc, so any sub-span is <= 180deg: large-arc-flag is always 0.
    // sweep-flag 1 sweeps over the top from left (0) to right (100).
    return `M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${end.x} ${end.y}`;
  };

  const bandLow = Math.max(0, clamped - uncertainty);
  const bandHigh = Math.min(100, clamped + uncertainty);

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        <path d={arc(0, 100)} fill="none" stroke="#1e293b" strokeWidth={10} strokeLinecap="round" />
        {uncertainty > 0 && (
          <path
            d={arc(bandLow, bandHigh)}
            fill="none"
            stroke={color}
            strokeOpacity={0.3}
            strokeWidth={10}
            strokeLinecap="round"
          />
        )}
        <path d={arc(0, clamped)} fill="none" stroke={color} strokeWidth={10} strokeLinecap="round" />
        <circle cx={point(clamped).x} cy={point(clamped).y} r={5} fill={color} />
      </svg>
      <div className="-mt-2 text-center">
        <div className="text-3xl font-semibold" style={{ color }}>
          {clamped.toFixed(0)}
        </div>
        <div className="text-xs uppercase tracking-wide text-slate-400">
          {riskBand(clamped)} risk
          {uncertainty > 0 && <span className="text-slate-500"> · ±{uncertainty.toFixed(0)}</span>}
        </div>
      </div>
    </div>
  );
}
