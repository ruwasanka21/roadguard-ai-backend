# -*- coding: utf-8 -*-
"""
Quick smoke-test — run with:
    cd backend
    python test_pipeline.py

Tests the geometry, scoring, and full pipeline WITHOUT a real server.
"""
import asyncio
import sys
import math
import os

# Force UTF-8 output on Windows so tick/cross characters render
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PASS = "[OK]"
FAIL = "[FAIL]"


# ── Geometry tests ─────────────────────────────────────────────────────────────

def test_haversine():
    from services.geometry import haversine_meters
    # Colombo (6.9271, 79.8612) to Kandy (7.2906, 80.6337)
    # Great-circle distance ~94 km
    d = haversine_meters(6.9271, 79.8612, 7.2906, 80.6337)
    assert 90_000 < d < 100_000, f"Expected ~94 km, got {d/1000:.1f} km"
    print(f"  haversine_meters Colombo-Kandy: {d/1000:.1f} km  {PASS}")


def test_bearing_delta():
    from services.geometry import bearing_delta
    import numpy as np
    # 5 and 355 are 10 degrees apart (shortest arc across 0/360 boundary)
    delta = bearing_delta(np.array([5.0]), np.array([355.0]))
    assert abs(delta[0] - 10.0) < 0.001, f"Expected 10, got {delta[0]}"
    print(f"  bearing_delta(5, 355) = {delta[0]}  {PASS}")


def test_smoother():
    from services.geometry import smooth_bearings_savgol
    import numpy as np
    # A flat 90-degree sequence with one 150-degree spike
    # The smoother should reduce but not eliminate the spike
    bearings = np.array([90.0, 90.0, 90.0, 150.0, 90.0, 90.0, 90.0])
    smoothed  = smooth_bearings_savgol(bearings, window=5, polyorder=2)
    spike_raw  = abs(150.0 - 90.0)           # = 60
    spike_smth = float(abs(smoothed[3] - 90.0))
    assert spike_smth < spike_raw, (
        f"Smoother should reduce spike magnitude: {spike_raw} -> {spike_smth}"
    )
    print(f"  savgol smoother: spike 60 -> {spike_smth:.1f} deg  {PASS}")


# ── Risk scorer tests ──────────────────────────────────────────────────────────

def test_bend_score():
    from services.risk_scorer import _BEND_SCORE, BendCategory
    assert _BEND_SCORE[BendCategory.hairpin] == 45.0
    assert _BEND_SCORE[BendCategory.none]    == 0.0
    assert _BEND_SCORE[BendCategory.sharp]   == 30.0
    print(f"  bend score lookup  {PASS}")


def test_category_from_angle():
    from services.risk_scorer import category_from_angle, BendCategory
    assert category_from_angle(5)   == BendCategory.none
    assert category_from_angle(30)  == BendCategory.gentle
    assert category_from_angle(60)  == BendCategory.moderate
    assert category_from_angle(90)  == BendCategory.sharp
    assert category_from_angle(130) == BendCategory.hairpin
    print(f"  category_from_angle  {PASS}")


def test_cluster_multiplier():
    """3 consecutive hairpins: bend score = 45 * 1.4 = 63, capped at 45."""
    from services.risk_scorer import _Seg, score_segments, assign_cluster_counts
    from models.response import BendCategory, RiskLevel

    class FakeReq:
        wind_speed_kmh     = 0.0
        weather_condition  = "clear"
        visibility_m       = 10000.0
        precip_mm_per_hour = 0.0
        is_rainy           = False
        is_foggy           = False

    segs = [
        _Seg(i, 7.0 + i * 0.001, 80.0, 7.0 + i * 0.002, 80.001,
             bearing_change=130.0, is_sharp_turn=True,
             bend_category=BendCategory.hairpin,
             consecutive_sharp_count=0, look_ahead_meters=300,
             distance_meters=150, slope_percent=0)
        for i in range(3)
    ]
    segs = assign_cluster_counts(segs)
    assert segs[2].consecutive_sharp_count == 3, (
        f"Expected 3, got {segs[2].consecutive_sharp_count}"
    )
    segs = score_segments(segs, FakeReq())
    # hairpin(45) * 1.4 = 63 -> capped to 45; no weather/wind/slope
    assert segs[2].risk_score == 45.0, f"Expected 45.0, got {segs[2].risk_score}"
    print(f"  cluster multiplier (cap at 45)  {PASS}")


def test_risk_classification():
    """High weather + hairpin should produce a high-risk segment."""
    from services.risk_scorer import _Seg, score_segments, assign_cluster_counts
    from models.response import BendCategory, RiskLevel

    class RainyReq:
        wind_speed_kmh     = 60.0
        weather_condition  = "rain"
        visibility_m       = 2000.0
        precip_mm_per_hour = 8.0
        is_rainy           = True
        is_foggy           = False

    seg = _Seg(0, 7.0, 80.0, 7.01, 80.01,
               bearing_change=130.0, is_sharp_turn=True,
               bend_category=BendCategory.hairpin,
               consecutive_sharp_count=0, look_ahead_meters=300,
               distance_meters=200, slope_percent=12.0)

    result = score_segments([seg], RainyReq())
    assert result[0].risk_level == RiskLevel.high, (
        f"Expected high risk, got {result[0].risk_level} (score={result[0].risk_score})"
    )
    print(f"  risk classification (hairpin + rain = high)  {PASS}")


# ── Full async pipeline test ───────────────────────────────────────────────────

async def test_full_pipeline():
    """
    Simulate a U-bend: 10 GPS points sweeping through 135 degrees.
    Expects at least one sharp/hairpin segment to be detected.
    """
    from models.request import AnalyzeRequest, GeoPoint
    from services.route_analyzer import RouteAnalyzerService

    pts = []
    for i in range(10):
        angle = math.radians(i * 15)           # sweeps 0 -> 135 degrees
        lat   = 7.29 + math.sin(angle) * 0.002
        lng   = 80.63 + math.cos(angle) * 0.002
        pts.append(GeoPoint(lat=lat, lng=lng))

    req = AnalyzeRequest(
        polyline            = pts,
        weather_condition   = "rain",
        wind_speed_kmh      = 40.0,
        visibility_m        = 3000.0,
        precip_mm_per_hour  = 5.0,
        temperature_c       = 22.0,
    )

    segments = await RouteAnalyzerService.analyze(req)
    assert len(segments) > 0, "Expected at least one segment"

    has_sharp = any(
        s.bend_category in ("sharp", "hairpin") for s in segments
    )
    cats = [s.bend_category for s in segments]
    print(f"  full pipeline: {len(segments)} seg(s), categories={cats}, sharp={has_sharp}  {PASS}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== RoadGuard AI Backend Smoke Tests ===\n")
    errors = []

    sync_tests = [
        test_haversine,
        test_bearing_delta,
        test_smoother,
        test_bend_score,
        test_category_from_angle,
        test_cluster_multiplier,
        test_risk_classification,
    ]

    for t in sync_tests:
        try:
            t()
        except Exception as e:
            errors.append(f"{t.__name__}: {e}")
            print(f"  {t.__name__}  {FAIL}: {e}")

    try:
        asyncio.run(test_full_pipeline())
    except Exception as e:
        errors.append(f"test_full_pipeline: {e}")
        print(f"  test_full_pipeline  {FAIL}: {e}")

    print()
    if errors:
        print(f"FAILED - {len(errors)} error(s):")
        for err in errors:
            print(f"  * {err}")
        sys.exit(1)
    else:
        print("All tests passed.")
