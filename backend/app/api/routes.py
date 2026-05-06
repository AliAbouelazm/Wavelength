import json
import os
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExploreTrack,
    HealthResponse,
    ModelMetrics,
    MoodProfile,
    RecommendRequest,
    RecommendResponse,
)
from app.recommender.engine import (
    analyze_features,
    get_explore_sample,
    get_mood_profiles,
    recommend,
)

METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "metrics.json")

router = APIRouter()

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]

VALID_MOODS = {"Happy", "Energetic", "Melancholic", "Focused", "Calm", "Intense"}


@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@router.get("/moods", response_model=List[MoodProfile])
async def moods():
    try:
        return get_mood_profiles()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_tracks(req: RecommendRequest):
    if req.mood not in VALID_MOODS:
        raise HTTPException(status_code=400, detail=f"Invalid mood. Choose from {sorted(VALID_MOODS)}.")
    try:
        tracks = recommend(
            mood=req.mood,
            energy_filter=req.energy_filter,
            valence_filter=req.valence_filter,
            tempo_filter=req.tempo_filter,
            n=req.n,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"mood": req.mood, "tracks": tracks}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    features = {col: getattr(req, col) for col in FEATURE_COLS}
    try:
        result = analyze_features(features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result


@router.get("/explore", response_model=List[ExploreTrack])
async def explore():
    try:
        return get_explore_sample(2000)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/model/metrics", response_model=ModelMetrics)
async def model_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run python -m app.model.evaluate first.",
        )
    with open(METRICS_PATH) as f:
        return json.load(f)
