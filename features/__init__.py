"""Feature engineering pipeline with strict train/serve parity.

There is exactly one ``featurize()`` entry point, used identically by the offline training
scripts and the online scoring path. Encoders and scalers are fitted offline and persisted
to ``artifacts/``; unseen categories become novelty flags rather than crashes.

Modules arrive in Phase 2: ``event_features``, ``session_features``, ``entity_window``,
``sequences``, ``geo``, ``cohorts``, ``encoders``, ``featurize``.
"""
