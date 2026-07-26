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
                  style={{ backgroundColor: `rgba(59, 130, 246, ${(cell / max) * 0.85})` }}
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
        <h1 className="text-xl font-semibold text-slate-900">Model Performance</h1>
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
            <div className="card-title mb-2">Alert-budget curve — precision vs budget</div>
            <p className="mb-2 text-xs text-slate-400">
              Precision among the top X% of events by risk. The actionable region is a small budget;
              precision falls toward the ~1% base rate as the budget widens.
            </p>
            {budgetCurve.length === 0 ? (
              <Empty message="No budget curve in the metrics payload." />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart
                  data={budgetCurve
                    // Each sampled point i spans ~i/(N-1) of the ranked list, i.e. that alert budget.
                    .map((p, i) => ({
                      budget: (i / Math.max(1, budgetCurve.length - 1)) * 100,
                      precision: p,
                    }))
                    // Zoom into the analyst's decision region (top 20%).
                    .filter((pt) => pt.budget <= 20)}
                >
                  <XAxis
                    dataKey="budget"
                    type="number"
                    domain={[0, 20]}
                    tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                    tick={{ fill: "#64748b", fontSize: 11 }}
                  />
                  <YAxis domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 8 }}
                    formatter={(value) => [Number(value).toFixed(3), "precision"]}
                    labelFormatter={(v) => `top ${Number(v).toFixed(1)}% budget`}
                  />
                  <Line type="monotone" dataKey="precision" stroke="#3b82f6" dot={false} strokeWidth={2} />
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
