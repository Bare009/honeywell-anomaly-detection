"""Explainability layer (Deliverable #5).

Every detection carries feature attributions (SHAP), a counterfactual "nearest-normal"
suggestion, per-step sequence attribution, a MITRE ATT&CK mapping and a plain-language
narrative. The narrative is the only optional piece and always has a deterministic
template fallback -- it never influences a score or verdict.

Modules arrive in Phase 6: ``shap_explainer``, ``counterfactual``,
``sequence_attribution``, ``mitre_map``, ``narrative``.
"""
