from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class RecommendRequest(BaseModel):
    mood: str
    energy_filter: Optional[float] = None
    valence_filter: Optional[float] = None
    tempo_filter: Optional[float] = None
    n: int = 20


class ExplanationItem(BaseModel):
    feature: str
    direction: str
    magnitude: float
    phrase: str


class TrackResult(BaseModel):
    track_name: str
    artists: str
    album_name: str
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness_norm: float
    speechiness: float
    tempo_norm: float
    tempo_raw: float
    valence: float
    similarity_score: float
    prediction_confidence: float
    explanation: List[ExplanationItem]


class RecommendResponse(BaseModel):
    mood: str
    tracks: List[TrackResult]


class AnalyzeRequest(BaseModel):
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness_norm: float
    speechiness: float
    tempo_norm: float
    valence: float
    energy_valence_ratio: float
    acoustic_energy_contrast: float
    danceability_tempo_score: float
    mood_index: float


class AnalyzeResponse(BaseModel):
    mood: str
    confidence: Dict[str, float]
    explanation: List[ExplanationItem]


class MoodProfile(BaseModel):
    mood: str
    track_count: int
    avg_features: Dict[str, float]


class ExploreTrack(BaseModel):
    track_name: str
    artists: str
    energy: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float
    tempo_norm: float
    mood: str


class ModelEntry(BaseModel):
    f1_macro: float
    accuracy: float
    classification_report: Dict[str, Any]


class ModelMetrics(BaseModel):
    models: Dict[str, ModelEntry]
    best_model: str


class HealthResponse(BaseModel):
    status: str
