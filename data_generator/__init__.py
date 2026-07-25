"""Synthetic behavioral telemetry generator (Deliverable #1).

Produces the labeled dataset the whole system is trained and evaluated on: per-entity
normal profiles, benign traffic, eight injected attack behaviors, correlated multi-stage
campaigns, and baked-in concept drift. Ground-truth labels are written separately from
the feature-bearing events.

Modules arrive in Phase 1: ``profiles``, ``normal``, ``attacks``, ``campaigns``,
``drift``, ``generate``.
"""
