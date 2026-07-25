"""Shared MongoDB-backed store accessor for the read API routers.

The read API and the scoring service use the same :class:`~serving.store.MongoStore`, so the query
logic lives in one place. The store is a process-wide singleton -- constructing it only creates
driver handles, no connection -- so routers can depend on it cheaply.
"""

from __future__ import annotations

from functools import lru_cache

from serving.store import MongoStore


@lru_cache(maxsize=1)
def get_store() -> MongoStore:
    """Return the process-wide MongoDB store used by the read API."""
    return MongoStore()


__all__ = ["get_store"]
