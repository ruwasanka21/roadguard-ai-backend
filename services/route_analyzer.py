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
SHARP_THRESH        = 60.0   # °  — is_sharp_turn = True above this
MAX_SEG_M           = 2000.0 # m  — cap for straight/gentle groups (rendering)
MIN_SEG_M           = 5.0    # m  — ignore zero-length artefacts


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
            lats, lngs, raw_b, edge_dists, deltas
        )
        logger.info("Direction-grouping → %d segments", len(raw_segments))

        # ── Stage 4: sliding-window peak scan (boost under-detected bends) ────
        _boost_with_scan_window(raw_segments, raw_b, seg_edge_ranges)

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
        segments.append(_Seg(
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
            bearing_change=0.0,        is_sharp_turn=False,
            bend_category=category_from_angle(0.0),
            consecutive_sharp_count=0, look_ahead_meters=300.0,
            distance_meters=round(float(edge_dists.sum()), 1),
            slope_percent=0.0,
        ))

    for idx, s in enumerate(segments):
        s.segment_index = idx

    return segments, edge_ranges


# ── Secondary: sliding-window peak boost ─────────────────────────────────────

def _boost_with_scan_window(
    segments:    List[_Seg],
    raw_b:       np.ndarray,
    edge_ranges: List[tuple],
) -> None:
    """
    For every segment, compute the maximum raw-delta sum across a window that
    extends ±SCAN_WIN edges beyond the segment's ACTUAL edge range.
    This catches single-edge sharp corners that direction-grouping partly absorbed.
    edge_ranges[i] = (start_edge, end_edge) indices into raw_b.
    """
    if len(raw_b) < 2:
        return

    # Precompute absolute single-step bearing changes at each edge boundary
    raw_abs = np.array([
        float(bearing_delta(raw_b[k], raw_b[k + 1]))
        for k in range(len(raw_b) - 1)
    ])

    for seg, (start_edge, end_edge) in zip(segments, edge_ranges):
        # Extend ±SCAN_WIN beyond the segment's actual edge span
        lo = max(0, start_edge - SCAN_WIN)
        hi = min(len(raw_abs), end_edge + SCAN_WIN + 1)  # +1: slice is exclusive

        if hi > lo:
            scan_peak = float(np.sum(raw_abs[lo:hi]))
        else:
            scan_peak = 0.0

        # Only upgrade, never downgrade
        if scan_peak > seg.bearing_change:
            seg.bearing_change = round(scan_peak, 2)
            seg.bend_category  = category_from_angle(seg.bearing_change)
            seg.is_sharp_turn  = seg.bearing_change >= SHARP_THRESH


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
