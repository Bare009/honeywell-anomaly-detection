import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Search, Snowflake } from "lucide-react";
import { getEntity, listDetections } from "../api/client";
import type { Detection } from "../api/types";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatDateTime, prettyType, riskBadgeClasses } from "../lib/format";
import ExplanationDrawer from "../components/ExplanationDrawer";
import Pagination from "../components/Pagination";

const PAGE_SIZE = 50;

export default function EntityExplorer() {
  const { entityId } = useParams();
  const navigate = useNavigate();
  const [term, setTerm] = useState(entityId ?? "");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Detection | null>(null);

  useEffect(() => setPage(0), [entityId]);

  const { data, loading, error } = useApi(
    () => (entityId ? getEntity(entityId, page * PAGE_SIZE, PAGE_SIZE) : Promise.resolve(null)),
    [entityId, page],
  );

  // Landing state (no entity selected): offer the highest-risk entities as shortcuts.
  const top = useApi(
    () => (entityId ? Promise.resolve([]) : listDetections({ sort: "risk", limit: 200, min_risk: 1 })),
    [entityId],
  );
  const topEntities = (() => {
    const seen = new Set<string>();
    const out: Detection[] = [];
    for (const d of top.data ?? []) {
      if (seen.has(d.entity_id)) continue;
      seen.add(d.entity_id);
      out.push(d);
      if (out.length >= 12) break;
    }
    return out;
  })();

  const history = (data?.detections ?? [])
    .slice()
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .map((d) => ({ t: new Date(d.timestamp).toLocaleDateString(), risk: d.risk_score }));

  const coldStart = data?.detections?.some((d) => d.cold_start) ?? false;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Entity Explorer</h1>
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
          className="w-72 rounded border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-800"
        />
        <button className="badge border-blue-200 bg-blue-50 text-blue-700">
          <Search className="h-3.5 w-3.5" /> Look up
        </button>
      </form>

      {!entityId && (
        <div className="space-y-3">
          <p className="text-sm text-slate-500">
            Choose the highest-risk entities or search an entity id to see its history.
          </p>
          {top.loading && <Loading />}
          {top.error && <ErrorBox message={top.error} />}
          {top.data && topEntities.length === 0 && (
            <Empty message="No high-risk entities yet — replay some events to populate the console." />
          )}
          {topEntities.length > 0 && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-4">
              {topEntities.map((d) => (
                <button
                  key={d.entity_id}
                  onClick={() => navigate(`/entities/${d.entity_id}`)}
                  className="card text-left transition-colors hover:border-blue-300 hover:bg-blue-50/40"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm text-slate-800">{d.entity_id}</span>
                    <span className={`badge ${riskBadgeClasses(d.risk_score)}`}>
                      {d.risk_score.toFixed(0)}
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-slate-500">{prettyType(d.anomaly_type)}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {entityId && loading && <Loading />}
      {entityId && error && <ErrorBox message={error} />}

      {data && (
        <>
          <div className="flex items-center gap-3">
            <div className="font-mono text-lg text-slate-900">{data.entity_id}</div>
            {coldStart && (
              <span className="badge border-blue-200 bg-blue-50 text-blue-700">
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
                  <XAxis dataKey="t" tick={{ fill: "#64748b", fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 8 }}
                  />
                  <Line type="monotone" dataKey="risk" stroke="#3b82f6" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="card">
            <div className="card-title mb-2">Detections (highest risk first)</div>
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
                      className="cursor-pointer border-t border-slate-200 hover:bg-slate-50"
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
            <Pagination
              page={page}
              pageSize={PAGE_SIZE}
              total={data.n_detections}
              onChange={setPage}
            />
          </div>
        </>
      )}

      <ExplanationDrawer detection={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
