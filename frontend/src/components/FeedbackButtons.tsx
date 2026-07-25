import { useState } from "react";
import { Check, ThumbsDown } from "lucide-react";
import { postFeedback } from "../api/client";
import type { Feedback } from "../api/types";

// Analyst feedback (D6): confirm or dismiss. On success we show the adjustment the system applied,
// so the loop is visible.
export default function FeedbackButtons({ detectionId }: { detectionId: string }) {
  const [applied, setApplied] = useState<Feedback | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(verdict: "confirmed" | "false_positive") {
    setBusy(true);
    setError(null);
    try {
      setApplied(await postFeedback(detectionId, verdict));
    } catch {
      setError("Feedback failed. Is the API reachable?");
    } finally {
      setBusy(false);
    }
  }

  if (applied) {
    const adj = applied.applied;
    return (
      <div className="text-sm text-slate-300">
        Recorded <span className="font-medium">{applied.analyst_verdict.replace("_", " ")}</span>.
        {adj && (
          <span className="text-slate-500">
            {" "}
            Entity threshold offset {adj.previous_value?.toFixed(0)} → {adj.new_value?.toFixed(0)}.
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <button
        disabled={busy}
        onClick={() => send("confirmed")}
        className="badge border-red-500/30 bg-red-500/10 text-red-300 hover:bg-red-500/20 disabled:opacity-50"
      >
        <Check className="h-3.5 w-3.5" /> Confirm attack
      </button>
      <button
        disabled={busy}
        onClick={() => send("false_positive")}
        className="badge border-slate-600 bg-slate-700/40 text-slate-200 hover:bg-slate-700/70 disabled:opacity-50"
      >
        <ThumbsDown className="h-3.5 w-3.5" /> False positive
      </button>
      {error && <span className="text-xs text-red-400">{error}</span>}
    </div>
  );
}
