"""Shared foundations for the behavioral anomaly detection system.

This package holds everything that both the offline (training) plane and the
online (serving) plane depend on:

- :mod:`common.config`   -- a single ``Settings`` object read from the environment.
- :mod:`common.models`   -- the Pydantic v2 domain contracts (events, detections, ...).
- :mod:`common.seed`     -- the global determinism utility (seed 42 everywhere).
- :mod:`common.database` -- lazy MongoDB / Redis clients plus index bootstrap.

Nothing in this package imports from the model, feature or serving packages, so it
is always safe to import and never creates a circular dependency.
"""

from common.config import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
