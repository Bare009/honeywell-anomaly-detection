import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getMetrics } from "../api/client";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatNumber, formatPercent, prettyType } from "../lib/format";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div className="stat mt-1">{value}</div>
    </div>
  );
}

function ConfusionMatrix({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="th">actual \ pred</th>
            {classes.map((c) => (
              <th key={c} className="th whitespace-nowrap">
                {prettyType(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="td whitespace-nowrap font-medium">{prettyType(classes[i] ?? String(i))}</td>
              {row.map((cell, j) => (
                <td
                  key={j}
                  className="td text-center"
                  style={{ backgroundColor: `rgba(56, 189, 248, ${cell / max})` }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPerformance() {
  const { data, loading, error } = useApi(getMetrics, []);
  const m = data ?? {};
  const hasMetrics = m.pr_auc != null || m.macro_f1 != null;
  const budgetCurve = Array.isArray(m.precision_at_k) ? m.precision_at_k : [];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Model Performance</h1>
        <p className="text-sm text-slate-500">
          Imbalance-aware metrics: PR-AUC and recall within the alert budget lead, not raw accuracy.
        </p>
      </div>

      {loading && <Loading />}
      {error && <ErrorBox message={error} />}

      {!loading && !error && !hasMetrics && (
        <Empty message="No evaluation metrics yet. Run the Phase 9 evaluation to populate this page." />
      )}

      {hasMetrics && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            <Metric label="PR-AUC" value={formatNumber(m.pr_auc as number)} />
            <Metric label="ROC-AUC" value={formatNumber(m.roc_auc as number)} />
            <Metric
              label="Recall @ 1% budget"
              value={formatPercent(m.recall_at_1pct_budget as number)}
            />
            <Metric label="Macro-F1" value={formatNumber(m.macro_f1 as number)} />
            <Metric label="Calibration ECE" value={formatNumber(m.calibration_ece as number)} />
          </div>

          <div className="card">
            <div className="card-title mb-2">Alert-budget curve — precision@k</div>
            {budgetCurve.length === 0 ? (
              <Empty message="No precision@k curve in the metrics payload." />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart
                  data={budgetCurve.map((p, i) => ({ k: i + 1, precision: p }))}
                >
                  <XAxis dataKey="k" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                  />
                  <Line type="monotone" dataKey="precision" stroke="#38bdf8" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {Array.isArray(m.confusion_matrix) && m.confusion_matrix.length > 0 && (
            <div className="card">
              <div className="card-title mb-2">Confusion matrix</div>
              <ConfusionMatrix
                matrix={m.confusion_matrix as number[][]}
                classes={(m.class_order as string[]) ?? []}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
