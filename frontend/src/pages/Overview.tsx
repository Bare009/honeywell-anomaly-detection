import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getDashboardSummary, listDetections } from "../api/client";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatDateTime, prettyType, riskBadgeClasses } from "../lib/format";

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div className="stat mt-1">{value}</div>
    </div>
  );
}

export default function Overview() {
  const summary = useApi(getDashboardSummary, []);
  const recent = useApi(() => listDetections({ sort: "risk", limit: 8, min_risk: 1 }), []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Overview</h1>
        <p className="text-sm text-slate-500">
          Live behavioral anomaly detection across users, service accounts and edge devices.
        </p>
      </div>

      {summary.loading && <Loading />}
      {summary.error && <ErrorBox message={summary.error} />}
      {summary.data && (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard label="Events scored" value={summary.data.n_detections.toLocaleString()} />
            <StatCard label="Anomalies" value={summary.data.n_anomalies.toLocaleString()} />
            <StatCard label="Campaigns" value={summary.data.n_campaigns.toLocaleString()} />
            <StatCard label="Analyst verdicts" value={summary.data.n_feedback.toLocaleString()} />
          </div>

          <div className="card">
            <div className="card-title mb-2">Anomalies by type</div>
            {Object.keys(summary.data.by_type).length === 0 ? (
              <Empty message="No anomalies yet — replay some events to populate the console." />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart
                  data={Object.entries(summary.data.by_type).map(([type, count]) => ({
                    type: prettyType(type),
                    count,
                  }))}
                  margin={{ bottom: 40 }}
                >
                  <XAxis
                    dataKey="type"
                    angle={-30}
                    textAnchor="end"
                    interval={0}
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                  />
                  <YAxis allowDecimals={false} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                  />
                  <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </>
      )}

      <div className="card">
        <div className="card-title mb-2">Highest-risk recent detections</div>
        {recent.loading && <Loading />}
        {recent.error && <ErrorBox message={recent.error} />}
        {recent.data && recent.data.length === 0 && <Empty message="Nothing above threshold." />}
        {recent.data && recent.data.length > 0 && (
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Entity</th>
                <th className="th">Type</th>
                <th className="th">Risk</th>
                <th className="th">When</th>
              </tr>
            </thead>
            <tbody>
              {recent.data.map((d) => (
                <tr key={d.detection_id} className="border-t border-slate-800">
                  <td className="td font-mono">{d.entity_id}</td>
                  <td className="td">{prettyType(d.anomaly_type)}</td>
                  <td className="td">
                    <span className={`badge ${riskBadgeClasses(d.risk_score)}`}>
                      {d.risk_score.toFixed(0)}
                    </span>
                  </td>
                  <td className="td text-slate-400">{formatDateTime(d.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
