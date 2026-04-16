"""
RoadGuard AI — FastAPI Backend
Sharp Bend Detection & Risk Scoring Pipeline
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from models.request  import AnalyzeRequest
from models.response import AnalyzeResponse
from services.route_analyzer import RouteAnalyzerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RoadGuard AI backend starting up ...")
    yield
    logger.info("RoadGuard AI backend shutting down ...")


app = FastAPI(
    title="RoadGuard AI",
    description="Sharp bend detection & road risk analysis API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/debug/test-sparse")
async def debug_test_sparse():
    """
    Test the backend with a known 3-hairpin sparse polyline.
    Call this in a browser to verify the algorithm is working correctly.
    Returns bend categories and risk scores for each segment.
    """
    import math
    from models.request import AnalyzeRequest, GeoPoint

    pts = []
    lat, lng = 7.00, 80.60
    for _ in range(3):
        for j in range(5):
            pts.append(GeoPoint(lat=lat, lng=lng))
            lat += 0.002
        for step in range(6):
            angle = math.radians(step * 25)
            pts.append(GeoPoint(lat=lat + math.sin(angle)*0.0005,
                                lng=lng + math.cos(angle)*0.0005))
        lat += 0.002

    req = AnalyzeRequest(
        polyline=pts, weather_condition="clear",
        wind_speed_kmh=0, visibility_m=10000,
        precip_mm_per_hour=0, temperature_c=25,
    )
    from services.route_analyzer import RouteAnalyzerService
    segments = await RouteAnalyzerService.analyze(req)
    return {
        "total_segments": len(segments),
        "sharp_or_hairpin": sum(1 for s in segments if s.bend_category in ("sharp","hairpin")),
        "segments": [
            {
                "idx": s.segment_index,
                "category": s.bend_category,
                "bearing_change": s.bearing_change,
                "risk_score": s.risk_score,
                "risk_level": s.risk_level,
            }
            for s in segments
        ]
    }


@app.post("/analyze-route", response_model=AnalyzeResponse)
async def analyze_route(req: AnalyzeRequest):
    """
    Accepts a decoded polyline + weather context.
    Returns a list of scored RouteSegments ready for Flutter map overlays.
    """
    if len(req.polyline) < 2:
        raise HTTPException(status_code=400, detail="Polyline must have at least 2 points")

    logger.info("Analyzing route: %d points", len(req.polyline))

    try:
        segments = await RouteAnalyzerService.analyze(req)
        return AnalyzeResponse(segments=segments)
    except Exception as e:
        logger.exception("Route analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
