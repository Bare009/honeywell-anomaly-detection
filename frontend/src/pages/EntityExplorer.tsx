import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Search, Snowflake } from "lucide-react";
import { getEntity } from "../api/client";
import type { Detection } from "../api/types";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatDateTime, prettyType, riskBadgeClasses } from "../lib/format";
import ExplanationDrawer from "../components/ExplanationDrawer";

export default function EntityExplorer() {
  const { entityId } = useParams();
  const navigate = useNavigate();
  const [term, setTerm] = useState(entityId ?? "");
  const [selected, setSelected] = useState<Detection | null>(null);

  const { data, loading, error } = useApi(
    () => (entityId ? getEntity(entityId) : Promise.resolve(null)),
    [entityId],
  );

  const history = (data?.detections ?? [])
    .slice()
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .map((d) => ({ t: new Date(d.timestamp).toLocaleDateString(), risk: d.risk_score }));

  const coldStart = data?.detections?.some((d) => d.cold_start) ?? false;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Entity Explorer</h1>
        <p className="text-sm text-slate-500">Behavioral history and detections for one entity.</p>
      </div>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (term.trim()) navigate(`/entities/${term.trim()}`);
        }}
      >
        <input
          value={term}
          onChange={(e) => setTerm(e.target.value)}
          placeholder="entity id, e.g. user_0007"
          className="w-72 rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200"
        />
        <button className="badge border-sky-500/30 bg-sky-500/10 text-sky-300">
          <Search className="h-3.5 w-3.5" /> Look up
        </button>
      </form>

      {!entityId && <Empty message="Enter an entity id to see its history." />}
      {entityId && loading && <Loading />}
      {entityId && error && <ErrorBox message={error} />}

      {data && (
        <>
          <div className="flex items-center gap-3">
            <div className="font-mono text-lg text-slate-100">{data.entity_id}</div>
            {coldStart && (
              <span className="badge border-sky-500/30 bg-sky-500/10 text-sky-300">
                <Snowflake className="h-3 w-3" /> cold start
              </span>
            )}
            <span className="text-sm text-slate-500">{data.n_detections} detections</span>
          </div>

          {history.length > 0 && (
            <div className="card">
              <div className="card-title mb-2">Risk over time</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={history}>
                  <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                  />
                  <Line type="monotone" dataKey="risk" stroke="#38bdf8" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="card">
            <div className="card-title mb-2">Detections</div>
            {data.detections.length === 0 ? (
              <Empty message="No detections recorded for this entity." />
            ) : (
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="th">Risk</th>
                    <th className="th">Type</th>
                    <th className="th">When</th>
                  </tr>
                </thead>
                <tbody>
                  {data.detections.map((d) => (
                    <tr
                      key={d.detection_id}
                      onClick={() => setSelected(d)}
                      className="cursor-pointer border-t border-slate-800 hover:bg-slate-800/40"
                    >
                      <td className="td">
                        <span className={`badge ${riskBadgeClasses(d.risk_score)}`}>
                          {d.risk_score.toFixed(0)}
                        </span>
                      </td>
                      <td className="td">{prettyType(d.anomaly_type)}</td>
                      <td className="td text-slate-400">{formatDateTime(d.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}

      <ExplanationDrawer detection={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
