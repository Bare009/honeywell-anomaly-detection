"""HTTP routers for the read API.

Phase 0 registers ``system`` only. Phase 7 adds ``detections``, ``entities``,
``campaigns``, ``metrics``, ``drift``, ``feedback``, ``dashboard`` and ``ws``.
"""

from api.routers import system

__all__ = ["system"]
