"""
Geometry primitives — vectorised with NumPy.

All functions work on both scalars and 1-D arrays so that bulk
segment operations are computed in a single C-level loop.
"""
import numpy as np

EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius in metres


def haversine_meters(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Great-circle distance between two points (or two arrays of points).

    Formula:
        a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
        d = 2R · atan2(√a, √(1−a))
    """
    φ1  = np.radians(lat1)
    φ2  = np.radians(lat2)
    dφ  = np.radians(lat2 - lat1)
    dλ  = np.radians(lon2 - lon1)
    a   = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def bearing_deg(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Initial (forward) bearing from point 1 → point 2, in degrees [0, 360).

    Formula:
        y = sin(Δλ) · cos(φ₂)
        x = cos(φ₁)·sin(φ₂) − sin(φ₁)·cos(φ₂)·cos(Δλ)
        θ = atan2(y, x)  shifted to [0°, 360°)
    """
    φ1  = np.radians(lat1)
    φ2  = np.radians(lat2)
    dλ  = np.radians(lon2 - lon1)
    y   = np.sin(dλ) * np.cos(φ2)
    x   = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(dλ)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def bearing_delta(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Absolute angular difference between two bearing arrays, always in [0°, 180°].

    Plain subtraction fails at the 0°/360° wrap (e.g. 5° and 355° → 350°, not 10°).
    The modulo + fold trick handles wrap correctly.
    """
    diff = np.abs(b2 - b1) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


def signed_bearing_delta(b1: np.ndarray | float, b2: np.ndarray | float) -> np.ndarray | float:
    """
    Signed angular difference between two bearings in [-180°, 180°].
    Positive values indicate a right turn, negative values indicate a left turn.
    """
    diff = (b2 - b1) % 360.0
    if isinstance(diff, np.ndarray):
        return np.where(diff > 180.0, diff - 360.0, diff)
    return diff - 360.0 if diff > 180.0 else diff



def smooth_bearings_savgol(bearings: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Savitzky-Golay smoother applied in bearing-space.

    Why SG instead of a simple Gaussian?
    - Gaussian blurs across all frequencies → softens the actual peak angle.
    - Savitzky-Golay fits a local polynomial → attenuates noise while preserving
      the shape of a sharp peak (hairpin apex stays sharp).

    The 0°/360° wrap-around is handled by smoothing the sine and cosine
    components separately and recombining with atan2.
    """
    from scipy.signal import savgol_filter

    rad   = np.radians(bearings)
    sine  = savgol_filter(np.sin(rad), window_length=window, polyorder=polyorder)
    cosine= savgol_filter(np.cos(rad), window_length=window, polyorder=polyorder)
    return (np.degrees(np.arctan2(sine, cosine)) + 360) % 360


def slope_percent(elev_start: float, elev_end: float, distance_m: float) -> float:
    """Rise-over-run as a percentage. +ve = uphill, -ve = downhill."""
    if distance_m <= 0:
        return 0.0
    return ((elev_end - elev_start) / distance_m) * 100.0
