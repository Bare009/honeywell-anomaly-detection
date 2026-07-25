import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureAttribution } from "../api/types";

// Horizontal bars of the top SHAP contributions. Red pushes toward the attack (raises risk),
// green pulls toward normal (lowers it) -- the sign is the story.
export default function ShapChart({ features }: { features: FeatureAttribution[] }) {
  if (!features.length) {
    return <p className="text-sm text-slate-500">No feature attributions available.</p>;
  }

  const data = features
    .slice()
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .map((f) => ({
      name: f.feature,
      value: f.contribution,
    }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ left: 12, right: 12 }}>
        <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <YAxis
          type="category"
          dataKey="name"
          width={150}
          tick={{ fill: "#cbd5e1", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
          formatter={(value) => {
            const num = Number(value);
            return [`${num >= 0 ? "+" : ""}${num.toFixed(3)}`, "SHAP"];
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((entry, index) => (
            <Cell key={index} fill={entry.value >= 0 ? "#ef4444" : "#22c55e"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
