import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { GitBranch, Snowflake, X } from "lucide-react";
import type { Detection } from "../api/types";
import { formatDateTime, formatValue, prettyType, riskBadgeClasses } from "../lib/format";
import RiskGauge from "./RiskGauge";
import ShapChart from "./ShapChart";
import CounterfactualPanel from "./CounterfactualPanel";
import SequenceHighlight from "./SequenceHighlight";
import MitreChips from "./MitreChips";
import FeedbackButtons from "./FeedbackButtons";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="card-title">{title}</h3>
      {children}
    </div>
  );
}

// The explanation drawer: everything an analyst needs to understand one alert, in one slide-over.
export default function ExplanationDrawer({
  detection,
  onClose,
}: {
  detection: Detection | null;
  onClose: () => void;
}) {
  if (!detection) return null;
  const { explanation: ex } = detection;
  const confidence = detection.anomaly_type_probs[detection.anomaly_type];

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative z-50 flex h-full w-full max-w-xl flex-col overflow-y-auto border-l border-slate-800 bg-slate-900 p-5 shadow-2xl">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <div className="font-mono text-sm text-slate-400">{detection.entity_id}</div>
            <div className="text-lg font-semibold text-slate-100">
              {prettyType(detection.anomaly_type)}
            </div>
            <div className="text-xs text-slate-500">{formatDateTime(detection.timestamp)}</div>
          </div>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-800">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="mb-4 flex items-center gap-6">
          <RiskGauge risk={detection.risk_score} uncertainty={detection.risk_uncertainty} />
          <div className="space-y-2 text-sm">
            <div>
              <span className={`badge ${riskBadgeClasses(detection.risk_score)}`}>
                {prettyType(detection.anomaly_type)}
              </span>{" "}
              {confidence != null && (
                <span className="text-slate-400">{(confidence * 100).toFixed(0)}% confidence</span>
              )}
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              {detection.cold_start && (
                <span className="badge border-sky-500/30 bg-sky-500/10 text-sky-300">
                  <Snowflake className="h-3 w-3" /> cold start
                </span>
              )}
              {detection.drift_flag && (
                <span className="badge border-amber-500/30 bg-amber-500/10 text-amber-300">
                  drift
                </span>
              )}
              {detection.detector_hits.map((hit) => (
                <span key={hit} className="badge border-red-500/30 bg-red-500/10 text-red-300">
                  {hit}
                </span>
              ))}
            </div>
            {detection.campaign_id && (
              <Link
                to={`/storyline?campaign=${detection.campaign_id}`}
                className="inline-flex items-center gap-1 text-xs text-sky-400 hover:underline"
              >
                <GitBranch className="h-3.5 w-3.5" /> part of campaign {detection.campaign_id}
              </Link>
            )}
          </div>
        </div>

        <div className="space-y-5">
          {ex.narrative && (
            <Section title="Narrative">
              <p className="text-sm text-slate-300">{ex.narrative}</p>
              <span className="text-xs text-slate-600">source: {ex.narrative_source}</span>
            </Section>
          )}

          <Section title="Top contributing features (SHAP)">
            <ShapChart features={ex.top_features} />
          </Section>

          <Section title="Make it normal (counterfactual)">
            <CounterfactualPanel cf={ex.counterfactual} />
          </Section>

          <Section title="Command sequence">
            <SequenceHighlight steps={ex.sequence_attribution} />
          </Section>

          {ex.baseline_comparison && Object.keys(ex.baseline_comparison.fields).length > 0 && (
            <Section title="Versus this entity's baseline">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="th">Feature</th>
                    <th className="th">Observed</th>
                    <th className="th">Typical</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(ex.baseline_comparison.fields).map(([name, cmp]) => (
                    <tr key={name} className={cmp.deviates ? "text-red-300" : ""}>
                      <td className="td font-mono">{name}</td>
                      <td className="td">{formatValue(cmp.observed)}</td>
                      <td className="td text-slate-400">{formatValue(cmp.typical)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          )}

          <Section title="MITRE ATT&CK">
            <MitreChips techniques={ex.mitre} />
          </Section>

          <Section title="Analyst feedback">
            <FeedbackButtons detectionId={detection.detection_id} />
          </Section>
        </div>
      </div>
    </div>
  );
}
