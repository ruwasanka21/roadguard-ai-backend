"""
Google Elevation API — batch fetch for real slope data.

The Flutter Dart side currently hardcodes slopePercent = 0.0.
This module fetches the true elevation at each segment midpoint,
so steep mountain roads correctly inflate the risk score.
"""
import os
import httpx
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

ELEVATION_URL = "https://maps.googleapis.com/maps/api/elevation/json"
MAX_POINTS_PER_REQUEST = 256   # Google hard limit per GET URL


async def fetch_elevations(points: List[Tuple[float, float]]) -> List[float]:
    """
    Batch-fetch elevations for a list of (lat, lng) tuples.
    Splits into chunks if > MAX_POINTS_PER_REQUEST.

    Returns:
        List of elevation values in metres, same order as input.
        Falls back to 0.0 for any point that fails.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set — returning zero elevations")
        return [0.0] * len(points)

    results: List[float] = []
    chunks = _chunk(points, MAX_POINTS_PER_REQUEST)

    async with httpx.AsyncClient(timeout=10.0) as client:
        for chunk in chunks:
            locations = "|".join(f"{lat},{lng}" for lat, lng in chunk)
            try:
                resp = await client.get(
                    ELEVATION_URL,
                    params={"locations": locations, "key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                if data["status"] != "OK":
                    logger.error("Elevation API error: %s", data["status"])
                    results.extend([0.0] * len(chunk))
                else:
                    results.extend(r["elevation"] for r in data["results"])
            except Exception as exc:
                logger.exception("Elevation fetch failed for chunk: %s", exc)
                results.extend([0.0] * len(chunk))

    return results


def compute_slopes(
    elevations: List[float],
    distances:  List[float],
) -> List[float]:
    """
    Compute rise-over-run slope (%) between consecutive elevation points.

    elevations[i] = elevation at the START of segment i
    elevations has len = num_segments + 1 (start of each + end of last)
    distances[i]  = length of segment i in metres

    Returns list of slope_percent values, one per segment.
    """
    elev   = np.array(elevations, dtype=float)
    dist   = np.array(distances,  dtype=float)
    # Avoid division by zero on zero-length edges
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = np.where(dist > 0, (elev[1:] - elev[:-1]) / dist * 100.0, 0.0)
    return slopes.tolist()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
