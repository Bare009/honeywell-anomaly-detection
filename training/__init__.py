"""Offline training plane -- the only place models are ever fitted.

Each script is runnable as a module (``python -m training.<name>``) and writes
version-stamped state into ``artifacts/``. The serving plane loads those artifacts and
never retrains.

Modules arrive in Phases 2-5: ``build_baselines``, ``train_baseline``, ``train_sequence``,
``train_classifier``, ``build_artifacts``.
"""
