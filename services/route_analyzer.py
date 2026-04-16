"""
Route Analyzer v3 — Direction-Grouping Bend Detection.

WHY THIS APPROACH?
  All previous distance-based and point-count approaches failed because
  Google Directions polylines have wildly different GPS point densities:
    - Urban 5 km route:    ~300 pts → spacing 17 m/pt
    - Mountain 75 km route: ~400 pts → spacing 190 m/pt

  Any fixed segment size works for one density but breaks for the other.

  Direction-grouping is DENSITY-AGNOSTIC:
    - Groups consecutive GPS edges that turn in the same direction
    - Right turns accumulate into one right-bend segment
    - Left turns accumulate into one left-bend segment
    - GPS jitter is suppressed by a NOISE_BAND dead zone
    - Opposite-direction reversal = natural bend boundary
    - Works identically for a 10 m/pt urban polyline and a 300 m/pt highway

ALGORITHM (5 steps):
  1. Compute raw edge bearings [] shape (n-1,)
  2. Compute signed turn delta at each interior waypoint:
       delta[k] = between edge k-1 and edge k, in (-180, +180]
       +ve = right turn, -ve = left turn
  3. Walk all waypoints, holding a "current group":
       |delta| < NOISE_BAND  → skip (GPS jitter)
       sign matches group    → accumulate turn
       sign differs          → flush group as a segment, start new
  4. Flush final group. Split any group > MAX_SEG_M for rendering.
  5. Score: accumulated bearing_change → BendCategory → risk score.
"""
import numpy as np
import logging
from typing import List, Tuple

from models.request  import AnalyzeRequest
from models.response import SegmentResponse
from services.geometry import (
    haversine_meters,
    bearing_deg,
    signed_bearing_delta,
)
from services.elevation import fetch_elevations, compute_slopes
from services.risk_scorer import (
    _Seg,
    category_from_angle,
    score_segments,
    assign_cluster_counts,
)

logger = logging.getLogger(__name__)

# ── Algorithm constants ───────────────────────────────────────────────────────
NOISE_BAND     = 3.0      # bearing changes smaller than this are GPS jitter — ignored
SHARP_THRESH   = 45.0     # accumulated turn (°) → is_sharp_turn = True
MAX_SEG_M      = 2000.0   # cap straight/gentle groups for UI rendering (metres)
MIN_SEG_M      = 10.0     # don't create sub-10 m micro-segments at route edges


class RouteAnalyzerService:

    @staticmethod
    async def analyze(req: AnalyzeRequest) -> List[SegmentResponse]:

        lats = np.array([p.lat for p in req.polyline])
        lngs = np.array([p.lng for p in req.polyline])
        n    = len(lats)

        if n < 2:
            return []

        # ── Stage 1: raw edge bearings ─────────────────────────────────────────
        raw_b = bearing_deg(lats[:-1], lngs[:-1], lats[1:], lngs[1:])

        # ── Stage 2: edge distances ────────────────────────────────────────────
        edge_dists = haversine_meters(lats[:-1], lngs[:-1], lats[1:], lngs[1:])

        # ── Stage 3: signed turn deltas at interior waypoints ─────────────────
        #   delta[k] = turn at interior waypoint k+1
        #   (between edge k and edge k+1),  k = 0 … n-3
        if n >= 3:
            deltas = np.array([
                float(signed_bearing_delta(raw_b[k], raw_b[k + 1]))
                for k in range(len(raw_b) - 1)
            ])
        else:
            deltas = np.zeros(0)

        logger.info("Polyline: %d pts, %d edges, %d delta(s)",
                    n, len(raw_b), len(deltas))

        # ── Stage 4: direction-grouping segmentation ───────────────────────────
        raw_segments = _group_by_direction(lats, lngs, raw_b, edge_dists, deltas)
        logger.info("Detected %d segments via direction-grouping", len(raw_segments))

        # ── Stage 5: elevation → slope ─────────────────────────────────────────
        seg_pts = [(s.start_lat, s.start_lng) for s in raw_segments]
        seg_pts.append((raw_segments[-1].end_lat, raw_segments[-1].end_lng))
        elevations = await fetch_elevations(seg_pts)
        slopes     = compute_slopes(elevations,
                                    [s.distance_meters for s in raw_segments])
        for seg, slope in zip(raw_segments, slopes):
            seg.slope_percent = slope

        # ── Stage 6: cluster counting + risk scoring ───────────────────────────
        raw_segments = assign_cluster_counts(raw_segments)
        scored       = score_segments(raw_segments, req)
        logger.info("Scoring complete — %d segments", len(scored))

        return _to_response(scored)


# ── Direction-grouping core ───────────────────────────────────────────────────

def _group_by_direction(
    lats:       np.ndarray,
    lngs:       np.ndarray,
    raw_b:      np.ndarray,   # edge bearings, shape (n-1,)
    edge_dists: np.ndarray,   # edge lengths,  shape (n-1,)
    deltas:     np.ndarray,   # signed waypoint turns, shape (n-2,) or empty
) -> List[_Seg]:
    """
    Walk through all GPS waypoints.  At each interior waypoint, we have a
    signed bearing delta.  Accumulate same-direction deltas into one group.
    When the direction reverses (ignoring jitter), flush the group and start
    a new one.

    Result: one _Seg per contiguous same-direction stretch of road.
    """
    n        = len(lats)
    segments: List[_Seg] = []

    # State of the current group
    seg_start  = 0       # point index where this group started
    acc_turn   = 0.0     # accumulated |delta| (°) in this direction
    acc_dist   = 0.0     # accumulated road distance (m)
    cur_dir    = 0       # +1 = right, -1 = left, 0 = undecided

    def _flush(end_pt: int):
        """Close out the current group and append a _Seg."""
        nonlocal seg_start, acc_turn, acc_dist, cur_dir

        if end_pt <= seg_start or acc_dist < MIN_SEG_M:
            return

        seg = _Seg(
            segment_index           = len(segments),
            start_lat               = float(lats[seg_start]),
            start_lng               = float(lngs[seg_start]),
            end_lat                 = float(lats[end_pt]),
            end_lng                 = float(lngs[end_pt]),
            bearing_change          = round(acc_turn, 2),
            is_sharp_turn           = acc_turn >= SHARP_THRESH,
            bend_category           = category_from_angle(acc_turn),
            consecutive_sharp_count = 0,
            look_ahead_meters       = 300.0,
            distance_meters         = round(acc_dist, 1),
            slope_percent           = 0.0,
        )
        segments.append(seg)

        seg_start = end_pt
        acc_turn  = 0.0
        acc_dist  = 0.0
        cur_dir   = 0

    for i in range(1, n):
        # Accrue this edge's distance into the current group
        acc_dist += float(edge_dists[i - 1])

        # delta index: delta between edge i-1 and edge i is deltas[i-1]
        #              valid only when i < n-1 (interior waypoints only)
        if i < n - 1 and len(deltas) > 0:
            d   = float(deltas[i - 1])
            abs_d = abs(d)

            if abs_d >= NOISE_BAND:
                new_dir = 1 if d > 0 else -1

                if cur_dir == 0:
                    # First real turn in this group — just pick a direction
                    cur_dir   = new_dir
                    acc_turn += abs_d

                elif new_dir == cur_dir:
                    # Continuing in the same direction — accumulate
                    acc_turn += abs_d

                else:
                    # Direction REVERSED — this waypoint is the boundary.
                    # The incoming edge belongs to the OLD group (already
                    # added to acc_dist above).  Flush the old group, then
                    # begin the new group at this waypoint.
                    _flush(i)
                    cur_dir   = new_dir
                    acc_turn += abs_d
                    # acc_dist was reset in _flush; the OUTGOING edge from
                    # waypoint i will be added in the NEXT iteration.

            # else: |d| < NOISE_BAND → jitter, ignore, stay in current group

        # Distance cap: split very long segments for UI rendering quality
        if acc_dist >= MAX_SEG_M:
            _flush(i)

    # Flush whatever remains
    _flush(n - 1)

    # Safety: if nothing was produced (e.g. only 2 points), return one segment
    if not segments:
        segments.append(_Seg(
            segment_index=0,
            start_lat=float(lats[0]),   start_lng=float(lngs[0]),
            end_lat=float(lats[-1]),    end_lng=float(lngs[-1]),
            bearing_change=0.0, is_sharp_turn=False,
            bend_category=category_from_angle(0.0),
            consecutive_sharp_count=0, look_ahead_meters=300.0,
            distance_meters=round(float(edge_dists.sum()), 1),
            slope_percent=0.0,
        ))

    # Re-index
    for idx, s in enumerate(segments):
        s.segment_index = idx

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
