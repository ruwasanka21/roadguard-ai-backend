"""
Route Analyzer — main pipeline orchestrator.

Pipeline stages:
  1. Compute raw edge bearings          (NumPy vectorised)
  2. Savitzky-Golay bearing smoother    (SciPy — better peak preservation)
  3. Adaptive segmentation              (150 m in curves, 250 m on straights)
  4. Local-peak bearing delta           (O(n) single scan per segment)
  5. BendCategory classification
  6. Elevation fetch → real slope %     (async Google Elevation API)
  7. Risk scoring + cluster counting
"""
import numpy as np
import logging
from typing  import List

from models.request  import AnalyzeRequest
from models.response import SegmentResponse
from services.geometry  import (
    haversine_meters,
    bearing_deg,
    bearing_delta,
    signed_bearing_delta,
    smooth_bearings_savgol,
)
from services.elevation import fetch_elevations, compute_slopes
from services.risk_scorer import (
    _Seg,
    category_from_angle,
    score_segments,
    assign_cluster_counts,
)

logger = logging.getLogger(__name__)

# ── Segmentation constants ────────────────────────────────────────────────────
DEFAULT_SEG_LEN   = 250.0   # metres — straight / gentle road
CURVE_SEG_LEN     = 150.0   # metres — tight curves (keeps hairpin in one segment)
CURVE_ANGLE_THRESH = 45.0   # accumulated degrees that triggers shorter segments


class RouteAnalyzerService:

    @staticmethod
    async def analyze(req: AnalyzeRequest) -> List[SegmentResponse]:

        # ── Convert to NumPy arrays for bulk operations ───────────────────────
        lats = np.array([p.lat for p in req.polyline])
        lngs = np.array([p.lng for p in req.polyline])
        n    = len(lats)

        # ── Stage 1: Compute raw edge bearings ───────────────────────────────
        #   One bearing per GPS edge: shape (n-1,)
        raw_bearings = bearing_deg(lats[:-1], lngs[:-1], lats[1:], lngs[1:])
        logger.debug("Raw bearings computed: %d edges", len(raw_bearings))

        # ── Stage 2: Savitzky-Golay smoother ─────────────────────────────────
        #   window=5, polyorder=2 → removes GPS jitter while keeping peak sharpness
        #   Falls back to raw if polyline is too short for the window
        if len(raw_bearings) >= 5:
            smooth = smooth_bearings_savgol(raw_bearings, window=5, polyorder=2)
        else:
            smooth = raw_bearings.copy()
        logger.debug("Bearings smoothed with Savitzky-Golay")

        # ── Precompute edge distances (vectorised Haversine) ──────────────────
        edge_dists = haversine_meters(lats[:-1], lngs[:-1], lats[1:], lngs[1:])

        # ── Stage 3 + 4 + 5: Adaptive segmentation + local-peak + classify ───
        raw_segments = _build_segments(lats, lngs, smooth, edge_dists)
        logger.info("Built %d raw segments", len(raw_segments))

        # ── Stage 6: Fetch real elevation → convert to slope % ───────────────
        #   Query start-of-each-segment + end-of-last-segment
        seg_points = [(s.start_lat, s.start_lng) for s in raw_segments]
        last = raw_segments[-1]
        seg_points.append((last.end_lat, last.end_lng))

        elevations = await fetch_elevations(seg_points)
        distances  = [s.distance_meters for s in raw_segments]
        slopes     = compute_slopes(elevations, distances)

        for seg, slope in zip(raw_segments, slopes):
            seg.slope_percent = slope
        logger.debug("Elevation & slope data attached")

        # ── Stage 7a: Cluster counting ────────────────────────────────────────
        raw_segments = assign_cluster_counts(raw_segments)

        # ── Stage 7b: Risk scoring ─────────────────────────────────────────────
        scored = score_segments(raw_segments, req)
        logger.info("Scoring complete — %d segments", len(scored))

        return _to_response(scored)


# ── Adaptive segmentation ─────────────────────────────────────────────────────

def _build_segments(
    lats:       np.ndarray,
    lngs:       np.ndarray,
    smooth:     np.ndarray,
    edge_dists: np.ndarray,
) -> List[_Seg]:
    """
    Walk the polyline, accumulating distance and turning angle.
    Cut a new segment when:
      - accumulated distance ≥ 250 m  (straight / gentle)
      - accumulated distance ≥ 150 m  AND accumulated angle > 45° (curve)
      - last point reached
    """
    segments:  List[_Seg] = []
    seg_start: int        = 0
    acc_dist:  float      = 0.0
    acc_angle: float      = 0.0

    for i in range(1, len(lats)):
        acc_dist  += float(edge_dists[i - 1])

        # Accumulate turning angle from consecutive smoothed bearings
        if i < len(smooth):
            acc_angle += float(bearing_delta(smooth[i - 1], smooth[i]))

        # Adaptive boundary length
        effective_len = CURVE_SEG_LEN if acc_angle > CURVE_ANGLE_THRESH else DEFAULT_SEG_LEN
        is_last       = (i == len(lats) - 1)

        if acc_dist >= effective_len or is_last:
            # ── Local-peak continuous turn detection ───────────────────────────
            #   Instead of checking the angle of exactly two adjacent points,
            #   accumulate the turning angle while the road keeps turning in
            #   the same direction. This correctly captures hairpins spread over
            #   multiple smaller GPS coordinates.
            from_idx = max(seg_start - 1, 0)
            to_idx   = min(i, len(smooth) - 1)

            peak_delta   = 0.0
            current_turn = 0.0
            current_sign = 0

            for k in range(from_idx, to_idx):
                d = float(signed_bearing_delta(smooth[k], smooth[k + 1]))
                sign = 1 if d > 0 else -1 if d < 0 else 0
                
                # If turning in same direction, accumulate the turn size
                if sign == current_sign or current_sign == 0:
                    current_turn += d
                else:
                    # Direction flipped (snaking road). Save peak and start new.
                    if abs(current_turn) > peak_delta:
                        peak_delta = abs(current_turn)
                    current_turn = d
                
                if sign != 0:
                    current_sign = sign

            if abs(current_turn) > peak_delta:
                peak_delta = abs(current_turn)

            cat = category_from_angle(peak_delta)

            segments.append(_Seg(
                segment_index           = len(segments),
                start_lat               = float(lats[seg_start]),
                start_lng               = float(lngs[seg_start]),
                end_lat                 = float(lats[i]),
                end_lng                 = float(lngs[i]),
                bearing_change          = round(peak_delta, 2),
                is_sharp_turn           = peak_delta > 60.0,
                bend_category           = cat,
                consecutive_sharp_count = 0,       # filled by cluster counter
                look_ahead_meters       = 300.0,   # overwritten by scorer
                distance_meters         = round(acc_dist, 1),
                slope_percent           = 0.0,     # overwritten by elevation step
            ))

            seg_start = i
            acc_dist  = 0.0
            acc_angle = 0.0

    return segments


# ── Response serialiser ────────────────────────────────────────────────────────

def _to_response(segments: List[_Seg]) -> List[SegmentResponse]:
    return [
        SegmentResponse(
            segment_index           = s.segment_index,
            start_lat               = s.start_lat,
            start_lng               = s.start_lng,
            end_lat                 = s.end_lat,
            end_lng                 = s.end_lng,
            bearing_change          = s.bearing_change,
            is_sharp_turn           = s.is_sharp_turn,
            bend_category           = s.bend_category,
            consecutive_sharp_count = s.consecutive_sharp_count,
            look_ahead_meters       = s.look_ahead_meters,
            distance_meters         = s.distance_meters,
            slope_percent           = round(s.slope_percent, 2),
            risk_score              = s.risk_score,
            risk_level              = s.risk_level,
        )
        for s in segments
    ]
