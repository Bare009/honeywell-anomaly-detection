import { useEffect, useState } from "react";

// Minimal data-fetching hook: run an async function, expose {data, loading, error}. Cancels state
// updates after unmount so a slow response never writes to a gone component.
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fn()
      .then((result) => alive && setData(result))
      .catch((err) => alive && setError(err?.message ?? String(err)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error, setData };
}
