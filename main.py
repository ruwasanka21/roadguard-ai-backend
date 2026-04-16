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
