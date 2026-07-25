"""Service layer for the read API.

Routers stay thin: they validate and shape HTTP, while these modules own the actual
queries and aggregation. That split keeps the data access testable without a running
HTTP server.
"""
