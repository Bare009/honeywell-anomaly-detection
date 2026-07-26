import { useEffect, useMemo, useState } from "react";
import { Snowflake } from "lucide-react";
import { listDetections } from "../api/client";
import { ANOMALY_CLASSES } from "../api/types";
import type { Detection } from "../api/types";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatDateTime, prettyType, riskBadgeClasses } from "../lib/format";
import ExplanationDrawer from "../components/ExplanationDrawer";
import Pagination from "../components/Pagination";

const PAGE_SIZE = 50;

export default function Alerts() {
  const [anomalyType, setAnomalyType] = useState("");
  const [entityType, setEntityType] = useState("");
  const [coldStart, setColdStart] = useState(false);
  const [minRisk, setMinRisk] = useState(0);
  const [sort, setSort] = useState<"risk" | "time">("risk");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Detection | null>(null);

  // Any filter change resets to the first page.
  useEffect(() => setPage(0), [anomalyType, entityType, coldStart, minRisk, sort]);

  const query = useMemo(
    () => ({
      sort,
      skip: page * PAGE_SIZE,
      limit: PAGE_SIZE,
      anomaly_type: anomalyType || undefined,
      entity_type: entityType || undefined,
      cold_start: coldStart || undefined,
      min_risk: minRisk || undefined,
    }),
    [sort, page, anomalyType, entityType, coldStart, minRisk],
  );

  const { data, loading, error } = useApi(() => listDetections(query), [query]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Ranked Alerts</h1>
          <p className="text-sm text-slate-500">Apply the required filters and click a row for the full explanation.</p>
        </div>
      </div>

      <div className="card flex flex-wrap items-end gap-3">
        <label className="text-xs text-slate-400">
          Type
          <select
            value={anomalyType}
            onChange={(e) => setAnomalyType(e.target.value)}
            className="mt-1 block rounded border border-slate-300 bg-white px-2 py-1 text-sm text-slate-800"
          >
            <option value="">All</option>
            {ANOMALY_CLASSES.filter((c) => c !== "normal").map((c) => (
              <option key={c} value={c}>
                {prettyType(c)}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-400">
          Entity type
          <select
            value={entityType}
            onChange={(e) => setEntityType(e.target.value)}
            className="mt-1 block rounded border border-slate-300 bg-white px-2 py-1 text-sm text-slate-800"
          >
            <option value="">All</option>
            <option value="user">User</option>
            <option value="service_account">Service account</option>
            <option value="edge_device">Edge device</option>
          </select>
        </label>
        <label className="text-xs text-slate-400">
          Min risk: {minRisk}
          <input
            type="range"
            min={0}
            max={100}
            value={minRisk}
            onChange={(e) => setMinRisk(Number(e.target.value))}
            className="mt-1 block w-40"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={coldStart} onChange={(e) => setColdStart(e.target.checked)} />
          Cold start only
        </label>
        <label className="text-xs text-slate-400">
          Sort
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as "risk" | "time")}
            className="mt-1 block rounded border border-slate-300 bg-white px-2 py-1 text-sm text-slate-800"
          >
            <option value="risk">Risk</option>
            <option value="time">Time</option>
          </select>
        </label>
      </div>

      <div className="card">
        {loading && <Loading />}
        {error && <ErrorBox message={error} />}
        {data && data.length === 0 && <Empty message="No detections match these filters." />}
        {data && data.length > 0 && (
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Risk</th>
                <th className="th">Entity</th>
                <th className="th">Type</th>
                <th className="th">Flags</th>
                <th className="th">When</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d) => (
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
                  <td className="td font-mono">{d.entity_id}</td>
                  <td className="td">{prettyType(d.anomaly_type)}</td>
                  <td className="td">
                    <div className="flex flex-wrap gap-1">
                      {d.cold_start && <Snowflake className="h-3.5 w-3.5 text-blue-600" />}
                      {d.detector_hits.map((h) => (
                        <span key={h} className="badge border-red-200 bg-red-50 text-red-700">
                          {h}
                        </span>
                      ))}
                      {d.campaign_id && (
                        <span className="badge border-slate-300 bg-slate-100 text-slate-600">
                          campaign
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="td text-slate-400">{formatDateTime(d.timestamp)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <Pagination
          page={page}
          pageSize={PAGE_SIZE}
          hasNext={Boolean(data && data.length === PAGE_SIZE)}
          onChange={setPage}
        />
      </div>

      <ExplanationDrawer detection={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
