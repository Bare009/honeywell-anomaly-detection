import { Loader2 } from "lucide-react";

// Small shared loading / error / empty placeholders so pages read consistently.
export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-slate-400">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
      {message}
      <div className="mt-1 text-xs text-red-400/70">
        Is the read API running? Try <code>docker compose up -d</code> and replay some events.
      </div>
    </div>
  );
}

export function Empty({ message }: { message: string }) {
  return <div className="py-8 text-sm text-slate-500">{message}</div>;
}
