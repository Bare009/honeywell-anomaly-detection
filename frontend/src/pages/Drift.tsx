import { getDrift } from "../api/client";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatNumber } from "../lib/format";

const STATUS_CLASSES: Record<string, string> = {
  stable: "border-green-200 bg-green-50 text-green-700",
  drifting: "border-red-200 bg-red-50 text-red-700",
  adapted: "border-blue-200 bg-blue-50 text-blue-700",
};

export default function Drift() {
  const { data, loading, error } = useApi(() => getDrift(200), []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Drift Monitor</h1>
        <p className="text-sm text-slate-500">
          Per-entity PSI. Gradual benign change is absorbed; abrupt shifts are flagged.
        </p>
      </div>

      {loading && <Loading />}
      {error && <ErrorBox message={error} />}
      {data && data.length === 0 && (
        <Empty message="No drift state recorded yet. Replay events to populate entity monitors." />
      )}
      {data && data.length > 0 && (
        <div className="card">
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Entity</th>
                <th className="th">Status</th>
                <th className="th">PSI</th>
                <th className="th">Samples</th>
              </tr>
            </thead>
            <tbody>
              {data.map((d) => (
                <tr key={d.entity_id} className="border-t border-slate-200">
                  <td className="td font-mono">{d.entity_id}</td>
                  <td className="td">
                    <span
                      className={`badge ${
                        STATUS_CLASSES[d.status ?? "stable"] ?? STATUS_CLASSES.stable
                      }`}
                    >
                      {d.status ?? "stable"}
                    </span>
                  </td>
                  <td className="td">{formatNumber(d.psi, 3)}</td>
                  <td className="td text-slate-400">{d.samples_seen ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
