"""
Route Analyzer v4 — Enhanced Direction-Grouping with Reversal Tolerance.

ENHANCEMENTS OVER v3:
  1. NOISE_BAND lowered 3.0° → 1.5°
       A gentle 40° highway curve often accumulates only 1–2° per GPS edge.
       The old 3° dead-zone silently filtered ALL of these, missing the bend.

  2. REVERSAL_TOLERANCE added (10°)
       Real bends recorded by Google Directions are NOT perfectly smooth; GPS
       encoding sometimes inserts a brief opposite-direction edge mid-bend.
       Previously, a single -5° blip inside a right-bend triggered a "flush",
       instantly splitting one 80° bend into 5 small fragments each showing ~15°
       (below the detection threshold).
       With tolerance: we absorb up to 10° of opposite-direction turning before
       we declare a genuine new section.

  3. Sliding-window peak scan (secondary signal)
       For every segment produced by direction-grouping, also scan ±SCAN_WIN
       raw-delta sums to catch any localised spike that may have been dampened
       by the tolerance absorption.  The final bearing_change = max(group, scan).

ALGORITHM:
  1. Compute raw edge bearings & distances (vectorised NumPy).
  2. Compute signed turn delta at each interior waypoint.
  3. Walk waypoints with direction-grouping + tolerance:
       |delta| < NOISE_BAND          → skip (pure jitter)
       same direction as current grp → accumulate; reset reversal counter
       opposite direction             → add to reversal counter
             reversal < TOLERANCE    → absorb (stay in current group)
             reversal ≥ TOLERANCE    → flush group, start new
  4. Sliding-window scan on each segment for localised peaks.
  5. Elevation + slope fetch (async).
  6. Cluster counting + risk scoring.
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
NOISE_BAND          = 1.5    # °  — deltas smaller than this are GPS jitter
REVERSAL_TOLERANCE  = 10.0   # °  — absorb up to this much opposite-turn before split
SCAN_WIN            = 6      # edges — half-window for localised peak scan
SHARP_THRESH        = 65.0   # °  — must match category_from_angle sharp threshold
MAX_SEG_M           = 500.0  # m  — cap for straight/gentle groups (rendering)
MIN_SEG_M           = 5.0    # m  — ignore zero-length artefacts
INTERSECTION_RADIUS = 60.0  # m  — raised to 60m to cover roundabout footprints


def _is_intersection_point(lat: float, lng: float, req: AnalyzeRequest) -> bool:
    """
    Returns True if (lat, lng) is within INTERSECTION_RADIUS metres of any
    Google Directions step endpoint. Such points are planned navigation turns
    (e.g. 'Turn right at the junction'), NOT dangerous road bends.
    """
    if not req.step_endpoints:
        return False
    for ep in req.step_endpoints:
        dist = haversine_meters(
            np.array([lat]),  np.array([lng]),
            np.array([ep.lat]), np.array([ep.lng])
        )[0]
        if dist <= INTERSECTION_RADIUS:
            return True
    return False


class RouteAnalyzerService:

    @staticmethod
    async def analyze(req: AnalyzeRequest) -> List[SegmentResponse]:

        lats = np.array([p.lat for p in req.polyline])
        lngs = np.array([p.lng for p in req.polyline])
        n    = len(lats)

        if n < 2:
            return []

        # ── Stage 1: raw edge bearings & distances ────────────────────────────
        raw_b      = bearing_deg(lats[:-1], lngs[:-1], lats[1:], lngs[1:])
        edge_dists = haversine_meters(lats[:-1], lngs[:-1], lats[1:], lngs[1:])

        # ── Stage 2: signed deltas at interior waypoints ──────────────────────
        #   delta[k] = turn at waypoint k+1, between edge k and edge k+1
        n_edges = len(raw_b)
        if n_edges >= 2:
            deltas = np.array([
                float(signed_bearing_delta(raw_b[k], raw_b[k + 1]))
                for k in range(n_edges - 1)
            ])
        else:
            deltas = np.zeros(0)

        logger.info("Polyline: %d pts, %d edges, %d interior deltas",
                    n, n_edges, len(deltas))

        # ── Stage 3: direction-grouping with reversal tolerance ───────────────
        raw_segments, seg_edge_ranges = _group_by_direction_tolerant(
            lats, lngs, raw_b, edge_dists, deltas, req
        )
        logger.info("Direction-grouping → %d segments", len(raw_segments))

        # ── Stage 4: (Boost pass removed — direction-grouping alone is exact) ────

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


# ── Direction-grouping with reversal tolerance ────────────────────────────────

def _group_by_direction_tolerant(
    lats:       np.ndarray,
    lngs:       np.ndarray,
    raw_b:      np.ndarray,
    edge_dists: np.ndarray,
    deltas:     np.ndarray,
    req:        AnalyzeRequest = None,
):
    """
    Walk the polyline. Group consecutive edges that turn in the same direction.
    Small opposite-direction blips (< REVERSAL_TOLERANCE total) are absorbed
    without splitting the current group.

    Returns (segments, edge_ranges) where edge_ranges[i] = (start_edge, end_edge)
    are the RAW_B indices that belong to segment i.
    """
    n        = len(lats)
    segments:    List[_Seg]          = []
    edge_ranges: List[tuple]         = []   # (start_edge, end_edge) per segment

    seg_start    = 0
    acc_turn     = 0.0   # total turn in the PRIMARY direction of this group
    acc_dist     = 0.0
    cur_dir      = 0     # +1 right, -1 left, 0 undecided
    pending_opp  = 0.0   # absorbed opposite-turn so far in this group

    def _flush(end_pt: int):
        nonlocal seg_start, acc_turn, acc_dist, cur_dir, pending_opp
        if end_pt <= seg_start or acc_dist < MIN_SEG_M:
            return
        start_edge = seg_start          # edge i goes from lats[i] → lats[i+1]
        end_edge   = end_pt - 1        # last edge ending at end_pt
        dir_str    = "right" if cur_dir == 1 else "left" if cur_dir == -1 else "straight"
        seg_points = [{"lat": float(lats[k]), "lng": float(lngs[k])} for k in range(seg_start, end_pt + 1)]

        # ── Intersection suppression ──────────────────────────────────────────
        # If this segment's START point is near a navigation instruction endpoint,
        # the turn is a deliberate driver action (e.g. "Turn right at junction"),
        # NOT an unexpected road bend. Zero out the angle so it is not flagged.
        at_intersection = _is_intersection_point(
            float(lats[seg_start]), float(lngs[seg_start]), req
        )
        effective_turn = 0.0 if at_intersection else acc_turn
        if at_intersection:
            logger.debug(
                "Seg %d suppressed — intersection at (%.5f, %.5f), raw angle=%.1f°",
                len(segments), lats[seg_start], lngs[seg_start], acc_turn,
            )

        segments.append(_Seg(
            segment_index           = len(segments),
            start_lat               = float(lats[seg_start]),
            start_lng               = float(lngs[seg_start]),
            end_lat                 = float(lats[end_pt]),
            end_lng                 = float(lngs[end_pt]),
            points                  = seg_points,
            bearing_change          = round(effective_turn, 2),
            turn_direction          = dir_str if not at_intersection else "straight",
            is_sharp_turn           = effective_turn >= SHARP_THRESH,
            bend_category           = category_from_angle(effective_turn),
            consecutive_sharp_count = 0,
            look_ahead_meters       = 300.0,
            distance_meters         = round(acc_dist, 1),
            slope_percent           = 0.0,
        ))
        edge_ranges.append((start_edge, end_edge))
        seg_start   = end_pt
        acc_turn    = 0.0
        acc_dist    = 0.0
        cur_dir     = 0
        pending_opp = 0.0

    for i in range(1, n):
        acc_dist += float(edge_dists[i - 1])

        # delta[i-1] is the turn at waypoint i (between edge i-1 and edge i)
        if i < n - 1 and (i - 1) < len(deltas):
            d       = float(deltas[i - 1])
            abs_d   = abs(d)

            if abs_d < NOISE_BAND:
                # Pure jitter — ignore, stay in current group
                pass

            else:
                new_dir = 1 if d > 0 else -1

                if cur_dir == 0:
                    # First real turn in this group
                    cur_dir   = new_dir
                    acc_turn += abs_d
                    pending_opp = 0.0

                elif new_dir == cur_dir:
                    # Continuing in the same direction — accumulate
                    acc_turn   += abs_d
                    pending_opp = 0.0   # same-dir turn resets tolerance counter

                else:
                    # OPPOSITE direction
                    pending_opp += abs_d

                    if pending_opp >= REVERSAL_TOLERANCE:
                        # Enough opposite turning accumulated → real direction change
                        _flush(i)
                        cur_dir     = new_dir
                        acc_turn    = pending_opp   # credit the whole opposite run
                        pending_opp = 0.0
                    # else: absorb — don't flip cur_dir, don't add to acc_turn

        # Distance cap: split very long segments for map rendering
        if acc_dist >= MAX_SEG_M:
            _flush(i)

    # Flush the final group
    _flush(n - 1)

    # Safety fallback for tiny 2-point routes
    if not segments:
        segments.append(_Seg(
            segment_index=0,
            start_lat=float(lats[0]),  start_lng=float(lngs[0]),
            end_lat=float(lats[-1]),   end_lng=float(lngs[-1]),
            points=[{"lat": float(lats[k]), "lng": float(lngs[k])} for k in range(len(lats))],
            bearing_change=0.0,        turn_direction="straight",
            is_sharp_turn=False,       bend_category=category_from_angle(0.0),
            consecutive_sharp_count=0, look_ahead_meters=300.0,
            distance_meters=round(float(edge_dists.sum()), 1),
            slope_percent=0.0,
        ))

    for idx, s in enumerate(segments):
        s.segment_index = idx

    return segments, edge_ranges




# ── Response serialiser ────────────────────────────────────────────────────────

def _to_response(segments: List[_Seg]) -> List[SegmentResponse]:
    return [
        SegmentResponse(
            segment_index           = s.segment_index,
            start_lat               = s.start_lat,
            start_lng               = s.start_lng,
            end_lat                 = s.end_lat,
            end_lng                 = s.end_lng,
            points                  = s.points,
            bearing_change          = s.bearing_change,
            turn_direction          = s.turn_direction,
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
