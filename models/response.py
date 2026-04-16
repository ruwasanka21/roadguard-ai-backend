"""Pydantic response models — what the API returns to Flutter."""
from pydantic import BaseModel
from typing   import List
from enum     import Enum


class BendCategory(str, Enum):
    none     = "none"
    gentle   = "gentle"
    moderate = "moderate"
    sharp    = "sharp"
    hairpin  = "hairpin"


class RiskLevel(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"


class SegmentResponse(BaseModel):
    segment_index:           int
    start_lat:               float
    start_lng:               float
    end_lat:                 float
    end_lng:                 float
    bearing_change:          float   # peak bearing delta in degrees (0–180)
    turn_direction:          str     # "left", "right", or "straight"
    is_sharp_turn:           bool
    bend_category:           BendCategory
    consecutive_sharp_count: int
    look_ahead_meters:       float
    distance_meters:         float
    slope_percent:           float   # real elevation data from Google API
    risk_score:              float   # 0–100
    risk_level:              RiskLevel


class AnalyzeResponse(BaseModel):
    segments: List[SegmentResponse]
