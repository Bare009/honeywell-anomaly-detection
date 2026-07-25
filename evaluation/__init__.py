"""Evaluation harness and experiments (backs Deliverable #7).

Imbalance-aware metrics only: PR-AUC, ROC-AUC, recall at the analyst alert budget,
precision@k, macro-F1 and calibration error -- never raw accuracy. Includes the cold-start
ablation, the drift adaptation experiment and campaign-reconstruction accuracy.

Modules arrive in Phase 9: ``metrics``, ``coldstart_experiment``, ``drift_experiment``,
``campaign_experiment``, ``report_figures``.
"""
