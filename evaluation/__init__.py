"""Offline evaluation and the honesty report (Deliverable #7).

Every number here is computed on the held-out **test** split -- the split neither the models nor
the fusion weights ever saw. The modules are deliberately separate from training so the report
cannot accidentally quote a training-time number, and each writes a machine-readable artifact that
``report.py`` assembles into ``REPORT.md``.
"""
