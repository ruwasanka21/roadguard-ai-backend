# -*- coding: utf-8 -*-
"""
Smoke-tests for the RoadGuard AI backend.
Run with:   cd backend && python test_pipeline.py
"""
import asyncio
import sys
import math
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PASS = "[OK]"
FAIL = "[FAIL]"


# ── Geometry ───────────────────────────────────────────────────────────────────

def test_haversine():
    from services.geometry import haversine_meters
    d = haversine_meters(6.9271, 79.8612, 7.2906, 80.6337)
    assert 90_000 < d < 100_000, f"Expected ~94 km, got {d/1000:.1f} km"
    print(f"  haversine Colombo→Kandy: {d/1000:.1f} km  {PASS}")


def test_bearing_delta():
    from services.geometry import bearing_delta
    import numpy as np
    delta = bearing_delta(np.array([5.0]), np.array([355.0]))
    assert abs(delta[0] - 10.0) < 0.001
    print(f"  bearing_delta(5°, 355°) = {delta[0]}°  {PASS}")


def test_signed_bearing_delta():
    from services.geometry import signed_bearing_delta
    # 355 → 5 is a +10° right turn
    d = float(signed_bearing_delta(355.0, 5.0))
    assert abs(d - 10.0) < 0.1, f"Expected +10, got {d}"
    # 5 → 355 is a -10° left turn
    d2 = float(signed_bearing_delta(5.0, 355.0))
    assert abs(d2 + 10.0) < 0.1, f"Expected -10, got {d2}"
    print(f"  signed_bearing_delta wrap: +10 / -10  {PASS}")


# ── Risk scorer ────────────────────────────────────────────────────────────────

def test_bend_score():
    from services.risk_scorer import _BEND_SCORE, BendCategory
    assert _BEND_SCORE[BendCategory.hairpin] == 55.0
    assert _BEND_SCORE[BendCategory.none]    == 0.0
    assert _BEND_SCORE[BendCategory.sharp]   == 35.0
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
    from services.risk_scorer import _Seg, score_segments, assign_cluster_counts, BendCategory

    class FakeReq:
        wind_speed_kmh     = 0.0
        weather_condition  = "clear"
        visibility_m       = 10000.0
        precip_mm_per_hour = 0.0
        is_rainy           = False
        is_foggy           = False

    segs = [
        _Seg(i, 7.0 + i*0.001, 80.0, 7.0+i*0.002, 80.001,
             bearing_change=130.0, is_sharp_turn=True,
             bend_category=BendCategory.hairpin,
             consecutive_sharp_count=0, look_ahead_meters=300,
             distance_meters=150, slope_percent=0)
        for i in range(3)
    ]
    segs = assign_cluster_counts(segs)
    assert segs[2].consecutive_sharp_count == 3
    segs = score_segments(segs, FakeReq())
    assert segs[2].risk_score == 65.0, f"Expected 65.0, got {segs[2].risk_score}"
    print(f"  cluster multiplier (cap at 65)  {PASS}")


def test_risk_classification():
    from services.risk_scorer import _Seg, score_segments, BendCategory, RiskLevel

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
    assert result[0].risk_level == RiskLevel.high, \
        f"Expected High, got {result[0].risk_level} (score={result[0].risk_score})"
    print(f"  risk classification (hairpin + rain = high)  {PASS}")


# ── Direction-grouping pipeline ────────────────────────────────────────────────

async def test_u_bend():
    """A U-turn (0→180° sweep in ONE direction) must be a single hairpin segment."""
    from models.request import AnalyzeRequest, GeoPoint
    from services.route_analyzer import RouteAnalyzerService

    # 10 points sweeping 135° all turning RIGHT
    pts = [
        GeoPoint(lat=7.29 + math.sin(math.radians(i * 15)) * 0.002,
                 lng=80.63 + math.cos(math.radians(i * 15)) * 0.002)
        for i in range(10)
    ]
    req = AnalyzeRequest(polyline=pts, weather_condition="clear",
                         wind_speed_kmh=0, visibility_m=10000,
                         precip_mm_per_hour=0, temperature_c=25)
    segs = await RouteAnalyzerService.analyze(req)
    assert len(segs) >= 1
    best = max(segs, key=lambda s: s.bearing_change)
    assert best.bend_category in ("sharp", "hairpin"), \
        f"Expected sharp/hairpin, got {best.bend_category} ({best.bearing_change}°)"
    print(f"  U-bend test: {len(segs)} seg(s), best={best.bend_category} "
          f"({best.bearing_change}°)  {PASS}")


async def test_sparse_three_hairpins():
    """
    3 hairpins on a long sparse route (220 m per GPS point).
    The direction-grouping algorithm must detect all 3 as sharp/hairpin.
    """
    from models.request import AnalyzeRequest, GeoPoint
    from services.route_analyzer import RouteAnalyzerService

    pts = []
    lat, lng = 7.00, 80.60
    for _ in range(3):
        # Short straight (sparse GPS)
        for _ in range(4):
            pts.append(GeoPoint(lat=lat, lng=lng))
            lat += 0.002           # ~220 m north
        # Hairpin: 6 points sweeping 150° all in same direction
        for step in range(6):
            ang = math.radians(step * 25)
            pts.append(GeoPoint(lat=lat + math.sin(ang)*0.0005,
                                lng=lng + math.cos(ang)*0.0005))
        lat += 0.002

    req = AnalyzeRequest(polyline=pts, weather_condition="clear",
                         wind_speed_kmh=0, visibility_m=10000,
                         precip_mm_per_hour=0, temperature_c=25)
    segs = await RouteAnalyzerService.analyze(req)
    sharp = sum(1 for s in segs if s.bend_category in ("sharp", "hairpin"))
    cats  = [s.bend_category for s in segs]
    assert sharp >= 2, f"Expected ≥2 sharp/hairpin, got {sharp}. cats={cats}"
    print(f"  sparse 3-hairpin: {len(segs)} seg(s), sharp/hairpin={sharp}  {PASS}")


async def test_s_curve():
    """S-curve: right bend then left bend → 2 separate bend segments."""
    from models.request import AnalyzeRequest, GeoPoint
    from services.route_analyzer import RouteAnalyzerService

    # Right arc (8 pts, +120°  total)
    right_pts = [
        GeoPoint(lat=7.00 + math.sin(math.radians(i*15))*0.002,
                 lng=80.00 + math.cos(math.radians(i*15))*0.002)
        for i in range(9)
    ]
    # Left arc (8 pts, -120° total) — reverse the direction
    left_pts = [
        GeoPoint(lat=7.00 + 0.01 + math.sin(math.radians(-i*15 + 180))*0.002,
                 lng=80.00 + 0.03 + math.cos(math.radians(-i*15 + 180))*0.002)
        for i in range(1, 9)
    ]
    pts = right_pts + left_pts
    req = AnalyzeRequest(polyline=pts, weather_condition="clear",
                         wind_speed_kmh=0, visibility_m=10000,
                         precip_mm_per_hour=0, temperature_c=25)
    segs = await RouteAnalyzerService.analyze(req)
    # Both bends should be detected
    notable = [s for s in segs if s.bend_category != "none"]
    assert len(notable) >= 2, \
        f"Expected ≥2 notable bends in S-curve, got {len(notable)}. cats={[s.bend_category for s in segs]}"
    print(f"  S-curve: {len(segs)} seg(s), notable bends={len(notable)}  {PASS}")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== RoadGuard AI Backend Smoke Tests ===\n")
    errors = []

    sync_tests = [
        test_haversine,
        test_bearing_delta,
        test_signed_bearing_delta,
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

    async_tests = [
        test_u_bend,
        test_sparse_three_hairpins,
        test_s_curve,
    ]

    for t in async_tests:
        try:
            asyncio.run(t())
        except Exception as e:
            errors.append(f"{t.__name__}: {e}")
            print(f"  {t.__name__}  {FAIL}: {e}")

    print()
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  * {err}")
        sys.exit(1)
    else:
        print("All tests passed.")
