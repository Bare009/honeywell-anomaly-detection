import type { MitreTechnique } from "../api/types";

// MITRE ATT&CK technique chips, linking out to the framework so an analyst can pivot.
export default function MitreChips({ techniques }: { techniques: MitreTechnique[] }) {
  if (!techniques.length) {
    return <p className="text-sm text-slate-500">No MITRE mapping.</p>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {techniques.map((t) => (
        <a
          key={t.technique_id}
          href={t.url ?? "#"}
          target="_blank"
          rel="noreferrer"
          className="badge border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100"
          title={t.tactic ?? undefined}
        >
          {t.technique_id} · {t.name}
        </a>
      ))}
    </div>
  );
}
