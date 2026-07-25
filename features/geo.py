"""Geographic math.

This is the canonical implementation. The feature pipeline uses it, and so will the
deterministic impossible-travel detector in Phase 5, so a single definition of "how far
apart" and "how fast" is shared by the features a model learns from and the rule that
overrides it.

All distances are kilometres and all velocities km/h.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Tuple

from common.config import settings

#: Mean Earth radius. Good enough: the error from assuming a sphere is well under 1%, far
#: below the uncertainty in IP geolocation itself.
EARTH_RADIUS_KM = 6371.0

#: Below this gap two events are treated as simultaneous. Without a floor, two events in the
#: same second from different cities would produce a near-infinite velocity, and any feature
#: derived from it would swamp everything else in the vector.
MIN_ELAPSED_SECONDS = 60.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def elapsed_hours(earlier: datetime, later: datetime) -> float:
    """Hours between two timestamps, floored so velocity cannot explode.

    Returns at least ``MIN_ELAPSED_SECONDS`` expressed in hours, and never a negative value.
    """
    seconds = (later - earlier).total_seconds()
    return max(abs(seconds), MIN_ELAPSED_SECONDS) / 3600.0


def geo_velocity_kmh(
    lat1: float,
    lon1: float,
    at1: datetime,
    lat2: float,
    lon2: float,
    at2: datetime,
) -> float:
    """Implied travel speed between two located events.

    This is the core of impossible-travel detection: a human cannot exceed roughly the speed
    of a commercial aircraft, so a higher implied speed means the two sessions were not the
    same person travelling.
    """
    distance = haversine_km(lat1, lon1, lat2, lon2)
    return distance / elapsed_hours(at1, at2)


def is_impossible_travel(
    lat1: float,
    lon1: float,
    at1: datetime,
    lat2: float,
    lon2: float,
    at2: datetime,
    threshold_kmh: Optional[float] = None,
    min_distance_km: float = 500.0,
) -> Tuple[bool, float]:
    """Whether two events imply physically impossible travel.

    Parameters
    ----------
    threshold_kmh:
        Speed above which travel is implausible. Defaults to
        ``settings.impossible_travel_kmh`` (900 km/h, roughly a commercial jet).
    min_distance_km:
        Distance floor. Two points a few kilometres apart can imply a silly velocity purely
        from geolocation jitter and a short time gap, so short hops never fire regardless of
        computed speed.

    Returns
    -------
    (fired, velocity_kmh)
        The velocity is returned either way so it can be shown in an explanation.
    """
    limit = settings.impossible_travel_kmh if threshold_kmh is None else threshold_kmh
    distance = haversine_km(lat1, lon1, lat2, lon2)
    velocity = distance / elapsed_hours(at1, at2)

    if distance < min_distance_km:
        return False, velocity
    return velocity > limit, velocity


def max_distance_from_km(
    points: list[Tuple[float, float]], lat: float, lon: float
) -> float:
    """Farthest any point lies from a reference location.

    Used as an entity's geographic spread: how far it roams from its usual place. An entity that
    always connects from one city has a tight spread, so a new location is genuinely surprising;
    for someone who already travels widely it is not.

    Measured from the centroid rather than as the largest pairwise distance. Both describe spread
    equally well, but this is O(n) where all-pairs is O(n^2) -- and this runs on the per-event
    hot path, where the all-pairs version cost 65% of total feature time.
    """
    if not points:
        return 0.0
    return max(haversine_km(lat, lon, point_lat, point_lon) for point_lat, point_lon in points)


def centroid(points: list[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Spherical centroid of a set of latitude/longitude points.

    Averaging degrees directly breaks across the antimeridian, so the points are converted
    to unit vectors, averaged, and converted back.
    """
    if not points:
        return None

    x = y = z = 0.0
    for lat, lon in points:
        phi, lam = math.radians(lat), math.radians(lon)
        x += math.cos(phi) * math.cos(lam)
        y += math.cos(phi) * math.sin(lam)
        z += math.sin(phi)

    count = float(len(points))
    x, y, z = x / count, y / count, z / count

    hypotenuse = math.sqrt(x * x + y * y)
    if hypotenuse < 1e-12 and abs(z) < 1e-12:
        return points[0]  # degenerate: antipodal points average to the centre

    return math.degrees(math.atan2(z, hypotenuse)), math.degrees(math.atan2(y, x))


__all__ = [
    "EARTH_RADIUS_KM",
    "MIN_ELAPSED_SECONDS",
    "haversine_km",
    "elapsed_hours",
    "geo_velocity_kmh",
    "is_impossible_travel",
    "max_distance_from_km",
    "centroid",
]
