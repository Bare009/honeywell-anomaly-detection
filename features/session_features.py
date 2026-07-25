"""Session-level features.

A session is the natural unit of intent: one login, one device, one location, a coherent run of
actions. Some behaviors are only visible at this level. A brute-force burst is a property of a
session, not of any single failed attempt. Lateral movement is a session touching many
unrelated resources, while each individual access looks routine.

These features describe the session **as observed so far**, not the finished session. That
constraint is what makes them usable online: the scorer sees event 3 of a session that may run
to 20, and must decide now. Computing them from the completed session offline would be a
train/serve mismatch that inflates offline metrics.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from common.models import Event
from features.entity_window import EntityState, EventSummary

#: Feature names produced by :func:`compute_session_features`. Must stay a subset of
#: :data:`features.event_features.NUMERIC_FEATURE_NAMES`.
SESSION_FEATURE_NAMES: List[str] = [
    "session_event_index",
    "session_distinct_resources",
    "session_auth_failures",
    "is_new_session",
]


def compute_session_features(event: Event, state: EntityState) -> Dict[str, float]:
    """Session-so-far features for one event.

    ``state`` must hold the entity's window as of *before* this event.
    """
    prior_events = state.session_events(event.session_id)

    resources = {summary.resource for summary in prior_events}
    resources.add(event.resource_accessed)

    failures = sum(1 for summary in prior_events if not summary.auth_success)
    if not event.auth_success:
        failures += 1

    return {
        # Position within the session. Early events carry less context, and the models can learn
        # to be more cautious about them rather than treating them like mid-session events.
        "session_event_index": float(len(prior_events)),
        "session_distinct_resources": float(len(resources)),
        "session_auth_failures": float(failures),
        "is_new_session": 0.0 if state.is_known_session(event.session_id) else 1.0,
    }


def summarize_session(events: Sequence[Event]) -> Dict[str, float]:
    """Aggregate a completed session.

    Offline analysis only -- the dashboard's entity view and the report figures use this. It is
    deliberately **not** part of :func:`features.featurize.featurize`, because a finished session
    is information the online scorer does not have at decision time.
    """
    if not events:
        return {}

    ordered = sorted(events, key=lambda event: event.timestamp)
    resources = {event.resource_accessed for event in ordered}
    countries = {event.geo.country for event in ordered}
    macs = {event.device_fingerprint.mac for event in ordered}
    failures = sum(1 for event in ordered if not event.auth_success)

    span_seconds = (ordered[-1].timestamp - ordered[0].timestamp).total_seconds()
    commands: List[str] = []
    for event in ordered:
        for token in event.command_sequence:
            if not commands or commands[-1] != token:
                commands.append(token)

    return {
        "event_count": float(len(ordered)),
        "distinct_resources": float(len(resources)),
        "distinct_countries": float(len(countries)),
        "distinct_devices": float(len(macs)),
        "auth_failures": float(failures),
        "auth_failure_ratio": failures / len(ordered),
        "span_seconds": float(max(0.0, span_seconds)),
        "total_bytes_out": float(sum(event.bytes_out for event in ordered)),
        "total_bytes_in": float(sum(event.bytes_in for event in ordered)),
        "command_count": float(len(commands)),
    }


def group_by_session(events: Iterable[Event]) -> Dict[Optional[str], List[Event]]:
    """Group events by ``session_id``, preserving time order within each group."""
    grouped: Dict[Optional[str], List[Event]] = {}
    for event in events:
        grouped.setdefault(event.session_id, []).append(event)
    for group in grouped.values():
        group.sort(key=lambda event: event.timestamp)
    return grouped


def _overlap_length(accumulated: Sequence[str], incoming: Sequence[str]) -> int:
    """Longest suffix of ``accumulated`` that is also a prefix of ``incoming``."""
    limit = min(len(accumulated), len(incoming))
    for size in range(limit, 0, -1):
        if list(accumulated[-size:]) == list(incoming[:size]):
            return size
    return 0


def session_command_sequence(events: Sequence[Event]) -> List[str]:
    """The full de-duplicated command sequence of a session.

    Each event carries a *rolling window* of the commands issued so far, so consecutive events
    overlap heavily. Naive concatenation repeats tokens, and suppressing only adjacent duplicates
    is not enough either: windows ``[login, view]`` then ``[login, view, logout]`` would yield
    ``login, view, login, view, logout``.

    So each incoming window is merged at its **longest overlap** with what has been reconstructed
    so far. An identical repeated window therefore contributes nothing, and a window that
    advanced by one action contributes exactly that action.
    """
    ordered = sorted(events, key=lambda event: event.timestamp)
    sequence: List[str] = []
    for event in ordered:
        tokens = [token for token in event.command_sequence if token]
        if not tokens:
            continue
        overlap = _overlap_length(sequence, tokens)
        sequence.extend(tokens[overlap:])
    return sequence


__all__ = [
    "SESSION_FEATURE_NAMES",
    "compute_session_features",
    "summarize_session",
    "group_by_session",
    "session_command_sequence",
]
