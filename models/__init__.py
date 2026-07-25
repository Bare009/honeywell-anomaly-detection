"""The three-tier detection stack plus fusion, detectors and drift.

One technique per tier, chosen deliberately rather than stacked:

* Tier 1 (``baseline``) -- a tabular **Autoencoder** over entity-relative features. Unsupervised,
  so it handles zero-day deviation; cold start is inherited from the feature layer, whose inputs
  are already blended toward a cohort prior. Per-feature reconstruction error is the explanation.
* Tier 2 (``sequence``) -- a **GRU** trained to predict the next command. Order-aware; per-step
  negative log-likelihood serves as both the score and the attribution.
* Tier 3 (``classifier``) -- calibrated LightGBM multi-class, which names the anomaly type.

``risk`` fuses all three into a calibrated 0-100 score with an uncertainty band, and
``detectors`` adds deterministic geometric checks. ``drift`` tracks PSI per entity.

Modules arrive in Phases 3-6: ``baseline``, ``sequence``, ``classifier``, ``detectors``,
``risk``, ``drift``.
"""
