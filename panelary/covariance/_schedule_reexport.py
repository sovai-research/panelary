"""Thin re-export of the shared schedule (coordination note D1).

``panelary.covariance.Schedule`` / ``Refit`` are the objects defined in
:mod:`panelary.core._schedule`; there is no covariance-specific schedule.
"""

from __future__ import annotations

from panelary.core._schedule import Refit, Schedule

__all__ = ["Refit", "Schedule"]
