import type { SequenceStepAttribution } from "../api/types";

// Renders the command sequence with each step shaded by its share of the surprise, so the
// anomalous span jumps out.
export default function SequenceHighlight({ steps }: { steps: SequenceStepAttribution[] }) {
  if (!steps.length) {
    return <p className="text-sm text-slate-500">No command-sequence signal for this event.</p>;
  }
  const max = Math.max(...steps.map((s) => s.score), 1e-6);
  return (
    <div className="flex flex-wrap gap-1.5">
      {steps.map((step, index) => {
        const intensity = Math.min(1, step.score / max);
        return (
          <span
            key={index}
            title={`surprise ${(step.score * 100).toFixed(0)}%`}
            className="rounded px-2 py-1 font-mono text-xs text-slate-100"
            style={{ backgroundColor: `rgba(239, 68, 68, ${0.15 + intensity * 0.65})` }}
          >
            {step.token}
          </span>
        );
      })}
    </div>
  );
}
