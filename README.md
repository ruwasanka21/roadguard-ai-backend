# RoadGuard AI — Python Backend

Sharp bend detection & road risk analysis API for the RoadGuard AI Flutter app.

## Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115 |
| Geometry | NumPy 1.26 (vectorised Haversine, bearing) |
| Signal processing | SciPy 1.13 (Savitzky-Golay smoother) |
| HTTP client | httpx (async Google Elevation API) |
| Validation | Pydantic v2 |
| Server | Uvicorn (ASGI) |

## Setup

```bash
cd backend

# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

## Run the server

```bash
# Development (auto-reload on file change)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive API docs are available at **http://localhost:8000/docs** while running.

## Run tests

```bash
python test_pipeline.py
```

## API

### `POST /analyze-route`

Request body:
```json
{
  "polyline": [
    {"lat": 7.2906, "lng": 80.6337},
    {"lat": 7.2950, "lng": 80.6400}
  ],
  "weather_condition":  "rain",
  "wind_speed_kmh":     40.0,
  "visibility_m":       3000.0,
  "precip_mm_per_hour": 5.0,
  "temperature_c":      22.0
}
```

Response:
```json
{
  "segments": [
    {
      "segment_index":           0,
      "start_lat":               7.2906,
      "start_lng":               80.6337,
      "end_lat":                 7.2950,
      "end_lng":                 80.6400,
      "bearing_change":          87.3,
      "is_sharp_turn":           true,
      "bend_category":           "sharp",
      "consecutive_sharp_count": 1,
      "look_ahead_meters":       600.0,
      "distance_meters":         218.4,
      "slope_percent":           3.2,
      "risk_score":              72.5,
      "risk_level":              "high"
    }
  ]
}
```

## Flutter integration

In `lib/services/route_analyzer_service.dart`, set `_baseUrl` to:

| Environment | URL |
|---|---|
| Android emulator | `http://10.0.2.2:8000` |
| Physical device (same LAN) | `http://192.168.x.x:8000` |
| Production | `https://your-domain.com` |

## Pipeline stages

```
POST /analyze-route
        │
        ▼
1. Compute raw edge bearings          (NumPy vectorised)
2. Savitzky-Golay bearing smoother    (SciPy — preserves peak sharpness)
3. Adaptive segmentation              150 m in curves / 250 m on straights
4. Local-peak bearing delta           O(n) single scan per window
5. BendCategory classification        none/gentle/moderate/sharp/hairpin
6. Google Elevation API               Real slope % at each midpoint
7. Risk scoring                       Weighted features → 0–100 score
8. Hairpin cluster counting           ≥3 consecutive → ×1.4 multiplier
        │
        ▼
    JSON response → Flutter map overlays + driver alerts
```
