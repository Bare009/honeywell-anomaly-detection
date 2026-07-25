import { useSearchParams } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { getCampaign, listCampaigns } from "../api/client";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatDateTime, prettyType, riskBadgeClasses } from "../lib/format";

export default function Storyline() {
  const [params, setParams] = useSearchParams();
  const selectedId = params.get("campaign");

  const list = useApi(() => listCampaigns(undefined, 100), []);
  const detail = useApi(
    () => (selectedId ? getCampaign(selectedId) : Promise.resolve(null)),
    [selectedId],
  );

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">Storyline</h1>
        <p className="text-sm text-slate-500">
          Related detections stitched into multi-stage attack campaigns.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="card lg:col-span-1">
          <div className="card-title mb-2">Campaigns</div>
          {list.loading && <Loading />}
          {list.error && <ErrorBox message={list.error} />}
          {list.data && list.data.length === 0 && <Empty message="No campaigns reconstructed yet." />}
          {list.data && (
            <ul className="space-y-1">
              {list.data.map((c) => (
                <li key={c.campaign_id}>
                  <button
                    onClick={() => setParams({ campaign: c.campaign_id })}
                    className={`flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-sm ${
                      c.campaign_id === selectedId
                        ? "bg-sky-500/15 text-sky-300"
                        : "text-slate-300 hover:bg-slate-800/60"
                    }`}
                  >
                    <span className="truncate font-mono text-xs">{c.entity_id}</span>
                    <span className={`badge ${riskBadgeClasses(c.max_risk)}`}>
                      {c.max_risk.toFixed(0)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card lg:col-span-2">
          {!selectedId && <Empty message="Select a campaign to see its kill chain." />}
          {selectedId && detail.loading && <Loading />}
          {selectedId && detail.error && <ErrorBox message={detail.error} />}
          {detail.data && (
            <div className="space-y-4">
              <div>
                <div className="font-mono text-sm text-slate-400">{detail.data.entity_id}</div>
                <div className="mt-1 flex flex-wrap items-center gap-1 text-sm text-slate-200">
                  {detail.data.kill_chain.map((stage, index) => (
                    <span key={index} className="flex items-center gap-1">
                      <span className="rounded bg-slate-800 px-2 py-0.5">{prettyType(stage)}</span>
                      {index < detail.data!.kill_chain.length - 1 && (
                        <ChevronRight className="h-4 w-4 text-slate-600" />
                      )}
                    </span>
                  ))}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {formatDateTime(detail.data.started_at)} → {formatDateTime(detail.data.last_activity)}
                  {" · "}peak risk {detail.data.max_risk.toFixed(0)}
                </div>
              </div>

              <ol className="relative space-y-3 border-l border-slate-700 pl-4">
                {detail.data.stages.map((stage, index) => (
                  <li key={index} className="relative">
                    <span className="absolute -left-[21px] top-1.5 h-2.5 w-2.5 rounded-full bg-sky-400" />
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-200">{prettyType(stage.anomaly_type)}</span>
                      <span className={`badge ${riskBadgeClasses(stage.risk_score)}`}>
                        {stage.risk_score.toFixed(0)}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500">{formatDateTime(stage.timestamp)}</div>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
