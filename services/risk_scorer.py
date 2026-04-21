"""
Risk scorer — weighted feature classification.

Score composition (0-100):
  bend severity   -> 0-70 pts   (BendCategory graduated scale)
  cluster bonus   -> x1.3 multiplier when consecutiveSharpCount >= 3
  weather         -> 0-30 pts
  wind            -> 0-10 pts   (linear above 30 km/h)
  slope           -> 0-5 pts    (bonus when |slope| > 10%)

Classification:
  score >= 50  -> high    (sharp bend alone = High)
  score >= 25  -> medium  (moderate bend alone = Medium)
  else         -> low

Category angle thresholds (tuned for mountain driving):
  hairpin  >= 100 degrees
  sharp    >=  60 degrees   (66 degrees IS a sharp mountain turn!)
  moderate >=  30 degrees
  gentle   >=  10 degrees

Dynamic look-ahead:
  hairpin  -> 500 m
  sharp    -> 250 m
  moderate -> 150 m (250 m in rain/fog)
  other    -> 250 m
"""
from dataclasses import dataclass
from models.response import BendCategory, RiskLevel, SegmentResponse


# -- Bend score lookup --------------------------------------------------------
# Calibrated so that:
#   - A sharp bend (65-100deg) ALONE = Medium (45 pts, threshold 60)
#   - A sharp bend + rain/fog       = High   (45+20 = 65 pts)
#   - A hairpin (100deg+) ALONE     = High   (65 pts)
#   - A roundabout suppressed by intersection filter = 0 pts
_BEND_SCORE: dict[BendCategory, float] = {
    BendCategory.none:     0.0,
    BendCategory.gentle:   8.0,   # 20-35 deg  -> Low
    BendCategory.moderate: 20.0,  # 35-65 deg  -> Low/Medium
    BendCategory.sharp:    45.0,  # 65-100 deg -> Medium alone, High + weather
    BendCategory.hairpin:  65.0,  # 100+ deg   -> High alone
}


# -- Internal segment dataclass (lives only inside the pipeline) --------------

@dataclass
class _Seg:
    segment_index:           int
    start_lat:               float
    start_lng:               float
    end_lat:                 float
    end_lng:                 float
    points:                  list[dict]
    bearing_change:          float
    turn_direction:          str
    is_sharp_turn:           bool
    bend_category:           BendCategory
    consecutive_sharp_count: int
    look_ahead_meters:       float
    distance_meters:         float
    slope_percent:           float
    risk_score:              float = 0.0
    risk_level:              RiskLevel = RiskLevel.low


def score_segments(segments: list[_Seg], req) -> list[_Seg]:
    """Apply the risk scorer to every segment. Returns same list, mutated."""
    for seg in segments:
        seg.risk_score, seg.risk_level, seg.look_ahead_meters = _score(seg, req)
    return segments


def _score(seg: _Seg, req) -> tuple[float, RiskLevel, float]:
    score = 0.0

    # -- 1. Bend severity -----------------------------------------------------
    bend_pts = _BEND_SCORE[seg.bend_category]

    # Cluster multiplier: >= 3 consecutive sharp/hairpin segments -> x1.3
    # Cap at 80 so moderate clusters can also reach High range.
    if seg.consecutive_sharp_count >= 3:
        bend_pts = min(bend_pts * 1.3, 80.0)

    score += bend_pts

    # -- 2. Weather -----------------------------------------------------------
    score += _weather_score(req)

    # -- 3. Wind (linear ramp 30->100 km/h maps to 0->10 pts) ----------------
    wind = req.wind_speed_kmh
    if wind > 30:
        score += min(((wind - 30) / 70.0) * 10.0, 10.0)

    # -- 4. Slope (bonus for steep hills) -------------------------------------
    if abs(seg.slope_percent) > 10:
        score += 5.0

    score = min(score, 100.0)
    level = _classify(score)
    ahead = _look_ahead(seg, req)

    return round(score, 2), level, ahead


def _weather_score(req) -> float:
    cond = req.weather_condition.lower()
    if cond == "thunderstorm":                        return 25.0
    if cond == "rain":
        return 15.0 + min(req.precip_mm_per_hour * 2, 10.0)
    if cond == "drizzle":                             return 10.0  # was 15
    if cond == "snow":                                return 22.0
    if cond in ("fog", "mist", "haze"):               return 18.0
    if cond in ("smoke", "dust", "sand"):             return 12.0
    if req.visibility_m < 500:                        return 18.0  # very low vis
    if req.visibility_m < 1000:                       return 12.0
    if req.visibility_m < 3000:                       return 6.0
    return 0.0


def _classify(score: float) -> RiskLevel:
    if score >= 60: return RiskLevel.high    # raised from 50
    if score >= 25: return RiskLevel.medium
    return RiskLevel.low


def _look_ahead(seg: _Seg, req) -> float:
    if seg.bend_category == BendCategory.hairpin:
        return 250.0
    if seg.bend_category == BendCategory.sharp:
        return 250.0
    if seg.bend_category == BendCategory.moderate:
        return 150.0 if (req.is_rainy or req.is_foggy) else 250.0
    return 250.0


# -- Cluster counter ----------------------------------------------------------

def assign_cluster_counts(segments: list[_Seg]) -> list[_Seg]:
    """
    Linear post-pass. Counts how many consecutive sharp/hairpin segments
    end at (and include) each segment.
    """
    run = 0
    for seg in segments:
        if seg.is_sharp_turn:
            run += 1
        else:
            run = 0
        seg.consecutive_sharp_count = run if seg.is_sharp_turn else 0
    return segments


# -- BendCategory classifier --------------------------------------------------

def category_from_angle(angle: float) -> BendCategory:
    """Calibrated thresholds:
      hairpin  >= 100 degrees  (true hairpin / very sharp bend)
      sharp    >=  65 degrees  (needs weather to reach High risk)
      moderate >=  35 degrees
      gentle   >=  20 degrees
    """
    if angle >= 100: return BendCategory.hairpin
    if angle >=  65: return BendCategory.sharp
    if angle >=  35: return BendCategory.moderate
    if angle >=  20: return BendCategory.gentle
    return BendCategory.none
