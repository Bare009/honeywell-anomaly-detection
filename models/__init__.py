"""The three-tier detection stack plus fusion, detectors and drift.

Tier 1 (``baseline``) is unsupervised and handles cold start and zero-day deviation.
Tier 2 (``sequence``) is order-aware. Tier 3 (``classifier``) names the anomaly type.
``risk`` fuses all three into a calibrated 0-100 score with an uncertainty band, and
``detectors`` adds deterministic geometric checks. ``drift`` tracks PSI per entity.

Modules arrive in Phases 3-6: ``baseline``, ``sequence``, ``classifier``, ``detectors``,
``risk``, ``drift``.
"""
