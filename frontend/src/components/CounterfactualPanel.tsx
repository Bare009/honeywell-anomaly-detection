import { ArrowRight, Wand2 } from "lucide-react";
import type { Counterfactual } from "../api/types";
import { formatValue } from "../lib/format";

// The "nearest-normal" explanation (D2): the smallest set of changes that would have cleared the alert.
export default function CounterfactualPanel({ cf }: { cf: Counterfactual | null }) {
  if (!cf) {
    return <p className="text-sm text-slate-500">No counterfactual computed.</p>;
  }
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 text-sm text-slate-300">
        <Wand2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-sky-400" />
        <span>{cf.summary ?? "No counterfactual summary."}</span>
      </div>
      {cf.changes.length > 0 && (
        <ul className="space-y-1">
          {cf.changes.map((change, index) => (
            <li
              key={index}
              className="flex items-center gap-2 rounded-md bg-slate-800/50 px-3 py-1.5 text-sm"
            >
              <span className="font-mono text-slate-300">{change.feature}</span>
              <span className="text-slate-500">{formatValue(change.actual)}</span>
              <ArrowRight className="h-3.5 w-3.5 text-slate-500" />
              <span className="text-green-400">{formatValue(change.suggested)}</span>
            </li>
          ))}
        </ul>
      )}
      {cf.original_risk != null && cf.resulting_risk != null && (
        <div className="text-xs text-slate-500">
          risk {cf.original_risk.toFixed(0)} → {cf.resulting_risk.toFixed(0)}
          {cf.found ? " (would clear the alert)" : " (still above threshold)"}
        </div>
      )}
    </div>
  );
}
