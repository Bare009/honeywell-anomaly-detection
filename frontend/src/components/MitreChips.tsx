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
          className="badge border-sky-500/30 bg-sky-500/10 text-sky-300 hover:bg-sky-500/20"
          title={t.tactic ?? undefined}
        >
          {t.technique_id} · {t.name}
        </a>
      ))}
    </div>
  );
}
