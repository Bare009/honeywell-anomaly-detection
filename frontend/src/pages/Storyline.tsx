import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { getCampaign, getDetection, listCampaigns } from "../api/client";
import type { Detection } from "../api/types";
import { useApi } from "../lib/useApi";
import { Loading, ErrorBox, Empty } from "../components/States";
import { formatDateTime, prettyType, riskBadgeClasses } from "../lib/format";
import ExplanationDrawer from "../components/ExplanationDrawer";

export default function Storyline() {
  const [params, setParams] = useSearchParams();
  const selectedId = params.get("campaign");
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);

  async function openStage(detectionId: string) {
    try {
      setSelectedDetection(await getDetection(detectionId));
    } catch {
      /* the detection may have aged out; ignore */
    }
  }

  const list = useApi(() => listCampaigns(undefined, 100), []);
  const detail = useApi(
    () => (selectedId ? getCampaign(selectedId) : Promise.resolve(null)),
    [selectedId],
  );

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Storyline</h1>
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
                        ? "bg-blue-50 text-blue-700"
                        : "text-slate-600 hover:bg-slate-100"
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
                <div className="mt-1 flex flex-wrap items-center gap-1 text-sm text-slate-700">
                  {detail.data.kill_chain.map((stage, index) => (
                    <span key={index} className="flex items-center gap-1">
                      <span className="rounded bg-slate-100 px-2 py-0.5">{prettyType(stage)}</span>
                      {index < detail.data!.kill_chain.length - 1 && (
                        <ChevronRight className="h-4 w-4 text-slate-400" />
                      )}
                    </span>
                  ))}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {formatDateTime(detail.data.started_at)} → {formatDateTime(detail.data.last_activity)}
                  {" · "}peak risk {detail.data.max_risk.toFixed(0)}
                </div>
              </div>

              <ol className="relative space-y-3 border-l border-slate-200 pl-4">
                {detail.data.stages.map((stage, index) => (
                  <li key={index} className="relative">
                    <span className="absolute -left-[21px] top-1.5 h-2.5 w-2.5 rounded-full bg-blue-500" />
                    <button
                      onClick={() => openStage(stage.detection_id)}
                      className="w-full rounded px-2 py-1 text-left hover:bg-slate-100"
                      title="Show the full explanation for this stage"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-700">{prettyType(stage.anomaly_type)}</span>
                        <span className={`badge ${riskBadgeClasses(stage.risk_score)}`}>
                          {stage.risk_score.toFixed(0)}
                        </span>
                      </div>
                      <div className="text-xs text-slate-500">{formatDateTime(stage.timestamp)}</div>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>

      <ExplanationDrawer detection={selectedDetection} onClose={() => setSelectedDetection(null)} />
    </div>
  );
}
