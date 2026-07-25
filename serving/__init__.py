"""Online scoring plane.

A stateless FastAPI service that loads artifacts once at startup and scores events in
milliseconds: featurize, run the detector tiers, fuse into a calibrated risk score,
explain, link into a campaign, persist. It validates input and returns 4xx on bad data --
it never retrains and never crashes the request loop.

Full pipeline arrives in Phase 7; ``app`` starts life in Phase 0 as a health endpoint.
"""
