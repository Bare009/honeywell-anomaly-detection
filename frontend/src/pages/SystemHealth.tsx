import { CheckCircle2, CircleSlash, XCircle } from "lucide-react";
import { getSystemHealth } from "../api/client";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox } from "../components/States";

function StatusIcon({ status }: { status: string }) {
  if (status === "ok") return <CheckCircle2 className="h-4 w-4 text-green-400" />;
  if (status === "disabled") return <CircleSlash className="h-4 w-4 text-slate-500" />;
  return <XCircle className="h-4 w-4 text-red-400" />;
}

export default function SystemHealth() {
  const { data, loading, error } = useApi(getSystemHealth, []);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">System Health</h1>
        <p className="text-sm text-slate-500">Read API, its dependencies and the loaded model build.</p>
      </div>

      {loading && <Loading />}
      {error && <ErrorBox message={error} />}
      {data && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="card">
            <div className="card-title mb-2">Service</div>
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt className="text-slate-400">Overall</dt>
                <dd className="flex items-center gap-2">
                  <StatusIcon status={data.status} /> {data.status}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Version</dt>
                <dd>{data.version}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Artifact schema</dt>
                <dd>{data.artifact_schema_version ?? "—"}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Artifacts ready</dt>
                <dd>{data.artifacts_ready ? "yes" : "no"}</dd>
              </div>
            </dl>
          </div>

          <div className="card">
            <div className="card-title mb-2">Dependencies</div>
            <ul className="space-y-2 text-sm">
              {Object.entries(data.dependencies).map(([name, dep]) => (
                <li key={name} className="flex items-center justify-between">
                  <span className="text-slate-300">{name}</span>
                  <span className="flex items-center gap-2 text-slate-400">
                    <StatusIcon status={dep.status} />
                    {dep.detail ?? dep.status}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
