"""
Route Analyzer — main pipeline orchestrator.

KEY INSIGHT:
  Google Directions polylines vary enormously in density:
    - Urban short route (5 km):  ~300 points → 1 point per 17 m
    - Mountain long route (75 km): ~400 points → 1 point per 190 m

  The ONLY correct approach is to adapt the segment size to the ACTUAL
  average point spacing of the received polyline.

  Target: ~300 m of road per segment. If points are 200 m apart, use
  N_PTS=2 (1 edge = 200 m, 2 = 400 m). If points are 15 m apart, use
  N_PTS=20 (covers 300 m).

  Bend severity is always measured as the TOTAL direction change from the
  first bearing in the segment to the last (entry-vs-exit). A hairpin
  spread over 8 GPS edges reads as the full ~130°.

  A secondary sliding-window scan (±HALF_WIN) ensures bends that straddle
  two segment boundaries are still captured.
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

TARGET_SEG_DIST_M = 300.0  # aim for ~300 m of road per segment
MIN_SEG_EDGES     = 2      # never flush a segment with fewer edges than this
SG_WINDOW         = 3      # Savitzky-Golay smoother window (minimal, sparse-friendly)
SHARP_THRESH      = 60.0   # degrees → is_sharp_turn = True


class RouteAnalyzerService:

    @staticmethod
    async def analyze(req: AnalyzeRequest) -> List[SegmentResponse]:

        lats = np.array([p.lat for p in req.polyline])
        lngs = np.array([p.lng for p in req.polyline])
        n    = len(lats)

        # ── Stage 1: raw edge bearings  (shape n-1) ───────────────────────────
        raw_bearings = bearing_deg(lats[:-1], lngs[:-1], lats[1:], lngs[1:])
        logger.debug("Raw bearings: %d edges", len(raw_bearings))

        # ── Stage 2: gentle SG smoothing ──────────────────────────────────────
        win = SG_WINDOW if len(raw_bearings) >= SG_WINDOW else len(raw_bearings)
        if win >= 2:
            smooth = smooth_bearings_savgol(raw_bearings, window=win, polyorder=1)
        else:
            smooth = raw_bearings.copy()

        # ── Stage 3: edge distances ────────────────────────────────────────────
        edge_dists = haversine_meters(lats[:-1], lngs[:-1], lats[1:], lngs[1:])

        # ── Stage 4: ADAPTIVE point-count-based segmentation ──────────────────
        #   Compute actual average GPS point spacing for THIS polyline.
        #   Then set N_PTS so that each segment covers ~TARGET_SEG_DIST_M metres.
        n_edges      = len(edge_dists)
        total_dist_m = float(edge_dists.sum())
        avg_spacing  = total_dist_m / n_edges if n_edges > 0 else 100.0

        # N_PTS: how many GPS edges per segment
        # Dense (15m/pt)  → N_PTS = 300/15 = 20
        # Sparse (200m/pt) → N_PTS = max(2, 300/200) = 2
        N_PTS = max(MIN_SEG_EDGES, int(round(TARGET_SEG_DIST_M / avg_spacing)))

        # Sliding window ±HALF_WIN: covers roughly ±TARGET_SEG_DIST_M / 2
        HALF_WIN = max(2, N_PTS // 2)

        logger.info(
            "Polyline: %d pts, avg_spacing=%.1f m → N_PTS=%d, HALF_WIN=%d",
            n, avg_spacing, N_PTS, HALF_WIN,
        )

        raw_segments = _build_segments(lats, lngs, smooth, edge_dists,
                                       N_PTS, HALF_WIN)
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


# ── Adaptive point-count segmentation ────────────────────────────────────────

def _build_segments(
    lats:       np.ndarray,
    lngs:       np.ndarray,
    smooth:     np.ndarray,  # shape (n-1,)
    edge_dists: np.ndarray,  # shape (n-1,)
    N_PTS:      int,
    HALF_WIN:   int,
) -> List[_Seg]:
    """
    Cut the polyline into segments of N_PTS GPS edges.

    For each segment, bend severity = max(entry_vs_exit, sliding_window_peak):
      • entry_vs_exit:
          bearing_delta(smooth[seg_start], smooth[seg_end])
          = total direction change across the whole segment
      • sliding_window_peak:
          For every index in the segment, measure bearing_delta over
          a ±HALF_WIN window.  Catches single-apex bends that straddle
          two segment boundaries.
    """
    segments:   List[_Seg] = []
    seg_start:  int        = 0   # point index of segment start
    edge_count: int        = 0   # edges accumulated in current segment
    acc_dist:   float      = 0.0

    n = len(lats)

    def _flush(end_point_idx: int):
        nonlocal seg_start, edge_count, acc_dist
        if end_point_idx <= seg_start:
            return

        # bearing indices: smooth[i] is the bearing of edge i→i+1
        b_lo = seg_start                              # first edge in segment
        b_hi = min(end_point_idx - 1, len(smooth) - 1)  # last edge in segment

        # ── Entry-vs-exit: total direction change across the segment ──────────
        if b_hi > b_lo:
            entry_exit = float(bearing_delta(smooth[b_lo], smooth[b_hi]))
        else:
            entry_exit = 0.0

        # ── Sliding-window: scan ±HALF_WIN around each edge index ─────────────
        sw_peak = entry_exit
        for mid in range(b_lo, b_hi + 1):
            lo = max(mid - HALF_WIN, 0)
            hi = min(mid + HALF_WIN, len(smooth) - 1)
            if hi > lo:
                d = float(bearing_delta(smooth[lo], smooth[hi]))
                if d > sw_peak:
                    sw_peak = d

        peak_delta = sw_peak
        cat        = category_from_angle(peak_delta)

        segments.append(_Seg(
            segment_index           = len(segments),
            start_lat               = float(lats[seg_start]),
            start_lng               = float(lngs[seg_start]),
            end_lat                 = float(lats[end_point_idx]),
            end_lng                 = float(lngs[end_point_idx]),
            bearing_change          = round(peak_delta, 2),
            is_sharp_turn           = peak_delta >= SHARP_THRESH,
            bend_category           = cat,
            consecutive_sharp_count = 0,
            look_ahead_meters       = 300.0,
            distance_meters         = round(max(acc_dist, 0.1), 1),
            slope_percent           = 0.0,
        ))

        seg_start  = end_point_idx
        edge_count = 0
        acc_dist   = 0.0

    for i in range(1, n):
        edge_count += 1
        acc_dist   += float(edge_dists[i - 1])
        is_last     = (i == n - 1)

        if (edge_count >= N_PTS and edge_count >= MIN_SEG_EDGES) or is_last:
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
