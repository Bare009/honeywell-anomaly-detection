import { ChevronLeft, ChevronRight } from "lucide-react";

// Simple prev/next pager. `page` is 0-indexed. Pass either `total` (to show "page X of Y") or just
// `hasNext` when the total is unknown.
export default function Pagination({
  page,
  pageSize,
  total,
  hasNext,
  onChange,
}: {
  page: number;
  pageSize: number;
  total?: number;
  hasNext?: boolean;
  onChange: (page: number) => void;
}) {
  const lastPage = total != null ? Math.max(0, Math.ceil(total / pageSize) - 1) : undefined;
  const canPrev = page > 0;
  const canNext = lastPage != null ? page < lastPage : Boolean(hasNext);

  if (!canPrev && !canNext) return null;

  const label =
    total != null
      ? `Page ${page + 1} of ${(lastPage ?? 0) + 1} · ${total.toLocaleString()} total`
      : `Page ${page + 1}`;

  const button = "badge border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40";

  return (
    <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
      <span>{label}</span>
      <div className="flex gap-2">
        <button className={button} disabled={!canPrev} onClick={() => onChange(page - 1)}>
          <ChevronLeft className="h-3.5 w-3.5" /> Prev
        </button>
        <button className={button} disabled={!canNext} onClick={() => onChange(page + 1)}>
          Next <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
