"""Pydantic request model — what Flutter POSTs to /analyze-route."""
from pydantic import BaseModel, Field
from typing  import List, Optional


class GeoPoint(BaseModel):
    lat: float
    lng: float


class AnalyzeRequest(BaseModel):
    polyline:           List[GeoPoint] = Field(..., min_length=2)
    weather_condition:  str   = "clear"   # "rain" | "fog" | "snow" | "clear" ...
    wind_speed_kmh:     float = 0.0
    visibility_m:       float = 10000.0
    precip_mm_per_hour: float = 0.0
    temperature_c:      float = 25.0

    # ── Intersection suppression ───────────────────────────────────────────────
    # End-point of every Google Directions step (i.e., every planned turn).
    # Any sharp angle that occurs within INTERSECTION_RADIUS metres of one of
    # these points is treated as a navigation-instruction turn, NOT a road bend.
    step_endpoints: Optional[List[GeoPoint]] = Field(default=None)

    # Convenience properties used by the scorer
    @property
    def is_rainy(self) -> bool:
        return self.weather_condition.lower() in ("rain", "drizzle", "thunderstorm")

    @property
    def is_foggy(self) -> bool:
        return self.weather_condition.lower() in ("fog", "mist", "haze")
