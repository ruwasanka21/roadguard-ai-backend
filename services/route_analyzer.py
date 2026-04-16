"""
Route Analyzer — completely rebuilt pipeline.

ROOT CAUSE OF PREVIOUS FAILURE:
  Google Directions polylines for long routes are highly compressed — one GPS
  point can be placed every 100–300 m. With the old 250 m distance-based
  segmentation, each segment contained only 1-2 points. Measuring the max
  single-step bearing change of 1-2 points meant a real 130° hairpin that
  is spread across 8 GPS points was always seen as multiple tiny 15° steps,
  every one of which was classified as "Gentle / Low Risk".

NEW APPROACH  (point-count based segmentation + entry-exit bearing):
  1. Segment by GPS-POINT COUNT (every N_PTS points), not by distance.
     This adapts automatically to sparse & dense polylines alike.
  2. Measure bend severity as bearing_delta(entry_bearing, exit_bearing),
     i.e. total road-direction change across the segment.
     A hairpin across 8 GPS points now reads as ~120°, not 15°.
  3. Secondary SLIDING-WINDOW scan: for every GPS point, also compute the
     bearing change over a ±HALF_WIN window. Use whichever is larger.
     This catches sharp single-apex bends that the segment method might
     straddle across two segment boundaries.
  4. Keep the Savitzky-Golay smoother but with a smaller window (3) to
     reduce over-smoothing on sparse polylines.
"""
import numpy as np
import logging
from typing import List

from models.request  import AnalyzeRequest
from models.response import SegmentResponse
from services.geometry import (
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

# ── Tuning constants ──────────────────────────────────────────────────────────
N_PTS          = 6      # GPS points per segment (adapts to polyline density)
MIN_SEG_PTS    = 2      # never cut a segment shorter than this many points
MAX_SEG_DIST   = 300.0  # hard-cap in metres (prevents huge straight segments)
SG_WINDOW      = 3      # Savitzky-Golay window — 3 is minimal but avoids oversmoothing
HALF_WIN       = 4      # ± half-window for sliding-window scan in points
SHARP_THRESH   = 60.0   # bearing change (°) to flag is_sharp_turn


class RouteAnalyzerService:

    @staticmethod
    async def analyze(req: AnalyzeRequest) -> List[SegmentResponse]:

        lats = np.array([p.lat for p in req.polyline])
        lngs = np.array([p.lng for p in req.polyline])
        n    = len(lats)

        # ── Stage 1: raw edge bearings  (shape n-1) ───────────────────────────
        raw_bearings = bearing_deg(lats[:-1], lngs[:-1], lats[1:], lngs[1:])
        logger.debug("Raw bearings: %d edges", len(raw_bearings))

        # ── Stage 2: gentle SG smoothing (window=3, removes GPS jitter only) ─
        if len(raw_bearings) >= SG_WINDOW:
            smooth = smooth_bearings_savgol(raw_bearings,
                                            window=SG_WINDOW, polyorder=1)
        else:
            smooth = raw_bearings.copy()

        # ── Stage 3: edge distances ────────────────────────────────────────────
        edge_dists = haversine_meters(lats[:-1], lngs[:-1], lats[1:], lngs[1:])

        # ── Stage 4: point-count-based segmentation ───────────────────────────
        raw_segments = _build_segments(lats, lngs, smooth, edge_dists, n)
        logger.info("Built %d raw segments from %d GPS points", len(raw_segments), n)

        # ── Stage 5: elevation → slope ────────────────────────────────────────
        seg_points = [(s.start_lat, s.start_lng) for s in raw_segments]
        seg_points.append((raw_segments[-1].end_lat, raw_segments[-1].end_lng))
        elevations = await fetch_elevations(seg_points)
        slopes     = compute_slopes(elevations, [s.distance_meters for s in raw_segments])
        for seg, slope in zip(raw_segments, slopes):
            seg.slope_percent = slope

        # ── Stage 6: cluster counter + risk scorer ────────────────────────────
        raw_segments = assign_cluster_counts(raw_segments)
        scored       = score_segments(raw_segments, req)
        logger.info("Scoring complete — %d segments", len(scored))

        return _to_response(scored)


# ── Point-count segmentation ──────────────────────────────────────────────────

def _build_segments(
    lats:       np.ndarray,
    lngs:       np.ndarray,
    smooth:     np.ndarray,
    edge_dists: np.ndarray,
    n:          int,
) -> List[_Seg]:
    """
    Walk polyline, cutting a new segment every N_PTS GPS points (or when
    MAX_SEG_DIST metres are accumulated, whichever comes first).

    Bend severity uses ENTRY-vs-EXIT bearing across the whole segment:
        peak_delta = bearing_delta(smooth[seg_start], smooth[i-1])

    This works correctly regardless of polyline density because it measures
    the TOTAL direction change — not the max of tiny individual steps.

    Additionally, a ±HALF_WIN sliding-window scan is compared and the
    larger of the two values is used so single-apex bends on segment
    boundaries are never missed.
    """
    segments:  List[_Seg] = []
    seg_start: int        = 0
    acc_dist:  float      = 0.0
    pt_count:  int        = 0

    def _flush(end_idx: int):
        nonlocal seg_start, acc_dist, pt_count
        if end_idx <= seg_start:
            return

        # ── Entry-vs-exit total bearing change ─────────────────────────────
        b_start = int(min(seg_start,     len(smooth) - 1))
        b_end   = int(min(end_idx - 1,   len(smooth) - 1))
        entry_exit_delta = float(bearing_delta(smooth[b_start], smooth[b_end]))

        # ── Sliding-window scan within this segment (catches apex on edge) ─
        sw_peak = 0.0
        for mid in range(b_start, b_end + 1):
            lo = max(mid - HALF_WIN, 0)
            hi = min(mid + HALF_WIN, len(smooth) - 1)
            if hi > lo:
                d = float(bearing_delta(smooth[lo], smooth[hi]))
                if d > sw_peak:
                    sw_peak = d

        peak_delta = max(entry_exit_delta, sw_peak)
        cat = category_from_angle(peak_delta)

        seg_dist = max(acc_dist, 0.1)
        segments.append(_Seg(
            segment_index           = len(segments),
            start_lat               = float(lats[seg_start]),
            start_lng               = float(lngs[seg_start]),
            end_lat                 = float(lats[end_idx]),
            end_lng                 = float(lngs[end_idx]),
            bearing_change          = round(peak_delta, 2),
            is_sharp_turn           = peak_delta >= SHARP_THRESH,
            bend_category           = cat,
            consecutive_sharp_count = 0,
            look_ahead_meters       = 300.0,
            distance_meters         = round(seg_dist, 1),
            slope_percent           = 0.0,
        ))
        seg_start = end_idx
        acc_dist  = 0.0
        pt_count  = 0

    for i in range(1, n):
        acc_dist += float(edge_dists[i - 1])
        pt_count += 1

        is_last     = (i == n - 1)
        hit_pts     = (pt_count >= N_PTS)
        hit_dist    = (acc_dist >= MAX_SEG_DIST)

        if (hit_pts or hit_dist or is_last) and pt_count >= MIN_SEG_PTS:
            _flush(i)
        elif is_last and pt_count < MIN_SEG_PTS:
            # Merge tiny tail into previous segment or flush as-is
            if segments:
                prev = segments[-1]
                prev.end_lat = float(lats[i])
                prev.end_lng = float(lngs[i])
                prev.distance_meters = round(prev.distance_meters + acc_dist, 1)
            else:
                _flush(i)

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
