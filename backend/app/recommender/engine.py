import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.model.predict import load_preprocessor, load_active_model, predict_batch_labels, predict_proba_single
from app.recommender.explainer import TreeModelExplainer, PermutationExplainer, build_explanations

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_tracks.parquet")

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]

SELECTABLE_MOODS = ["Happy", "Energetic", "Melancholic", "Focused", "Calm", "Intense"]

_tracks: pd.DataFrame = None
_scaled_features: np.ndarray = None
_mood_avgs: dict = {}
_explainer = None
_model = None
_model_type: str = None
_class_labels: list = None
_preprocessor: dict = None


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_engine():
    global _tracks, _scaled_features, _mood_avgs, _explainer
    global _model, _model_type, _class_labels, _preprocessor

    _preprocessor = load_preprocessor()
    scaler = _preprocessor["scaler"]
    _class_labels = _preprocessor["class_labels"]

    _model, _model_type = load_active_model()

    print("Loading track dataset...")
    tracks = pd.read_parquet(DATA_PATH)

    X = tracks[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    print(f"Running batch prediction on {len(tracks)} tracks...")
    predicted = predict_batch_labels(X_scaled, _model, _model_type, _class_labels)
    tracks = tracks.copy()
    tracks["predicted_mood"] = predicted

    _tracks = tracks
    _scaled_features = X_scaled

    for mood in SELECTABLE_MOODS:
        mask = tracks["predicted_mood"] == mood
        if mask.sum() > 0:
            _mood_avgs[mood] = tracks.loc[mask, FEATURE_COLS].mean().values
        else:
            _mood_avgs[mood] = tracks[FEATURE_COLS].mean().values

    # Build explainer
    if _model_type in ("random_forest", "xgboost"):
        _explainer = TreeModelExplainer(_model)
    else:
        device = _get_device()
        _explainer = PermutationExplainer(_model, device)

    counts = tracks["predicted_mood"].value_counts()
    for mood in SELECTABLE_MOODS:
        print(f"  {mood}: {counts.get(mood, 0)} tracks")

    print("Engine ready.")


def _rebuild_engineered(base: np.ndarray) -> np.ndarray:
    """Recompute engineered features from adjusted base features."""
    # base order: acousticness(0) danceability(1) energy(2) instrumentalness(3)
    #             liveness(4) loudness_norm(5) speechiness(6) tempo_norm(7) valence(8)
    acousticness = base[0]
    danceability = base[1]
    energy = base[2]
    valence = base[8]
    tempo_norm = base[7]

    energy_valence_ratio = energy / (valence + 0.001)
    acoustic_energy_contrast = acousticness - energy
    # tempo_norm * 250 approximates raw tempo for danceability_tempo_score
    danceability_tempo_score = danceability * (tempo_norm * 250 / 200.0)
    mood_index = valence * 0.4 + energy * 0.3 + danceability * 0.2 + acousticness * 0.1

    result = base.copy()
    result[9] = energy_valence_ratio
    result[10] = acoustic_energy_contrast
    result[11] = danceability_tempo_score
    result[12] = mood_index
    return result


def recommend(
    mood: str,
    energy_filter: float = None,
    valence_filter: float = None,
    tempo_filter: float = None,
    n: int = 20,
) -> list:
    if _tracks is None:
        raise RuntimeError("Engine not loaded. Call load_engine() first.")

    scaler = _preprocessor["scaler"]
    class_idx = _class_labels.index(mood)

    mood_mask = (_tracks["predicted_mood"] == mood).values
    if mood_mask.sum() == 0:
        return []

    mood_scaled = _scaled_features[mood_mask]
    mood_tracks = _tracks[mood_mask].reset_index(drop=True)

    # Build target vector in original feature space
    target_orig = _mood_avgs[mood].copy()
    if energy_filter is not None:
        target_orig[FEATURE_COLS.index("energy")] = float(energy_filter)
    if valence_filter is not None:
        target_orig[FEATURE_COLS.index("valence")] = float(valence_filter)
    if tempo_filter is not None:
        target_orig[FEATURE_COLS.index("tempo_norm")] = float(tempo_filter) / 250.0

    target_orig = _rebuild_engineered(target_orig)
    target_scaled = scaler.transform([target_orig])

    sims = cosine_similarity(mood_scaled, target_scaled).flatten()
    top_idx = np.argsort(sims)[::-1][:n]

    results = []
    for i in top_idx:
        row = mood_tracks.iloc[i]
        raw_feats = {col: float(row[col]) for col in FEATURE_COLS}
        x_scaled = mood_scaled[i].reshape(1, -1)

        probs = predict_proba_single(x_scaled, _model, _model_type)
        confidence = float(probs[class_idx])

        raw_expl = _explainer.explain(x_scaled, class_idx, raw_feats)
        explanation = build_explanations(raw_expl, mood)

        results.append({
            "track_name": str(row["track_name"]),
            "artists": str(row["artists"]),
            "album_name": str(row.get("album_name", "")),
            "acousticness": float(row["acousticness"]),
            "danceability": float(row["danceability"]),
            "energy": float(row["energy"]),
            "instrumentalness": float(row["instrumentalness"]),
            "liveness": float(row["liveness"]),
            "loudness_norm": float(row["loudness_norm"]),
            "speechiness": float(row["speechiness"]),
            "tempo_norm": float(row["tempo_norm"]),
            "tempo_raw": float(row.get("tempo_raw", row["tempo_norm"] * 250)),
            "valence": float(row["valence"]),
            "similarity_score": round(float(sims[i]), 4),
            "prediction_confidence": round(confidence, 4),
            "explanation": explanation,
        })

    return results


def get_mood_profiles() -> list:
    if _tracks is None:
        raise RuntimeError("Engine not loaded.")
    out = []
    for mood in SELECTABLE_MOODS:
        mask = _tracks["predicted_mood"] == mood
        count = int(mask.sum())
        if count > 0:
            avg = _tracks.loc[mask, FEATURE_COLS].mean().to_dict()
        else:
            avg = {col: 0.0 for col in FEATURE_COLS}
        out.append({
            "mood": mood,
            "track_count": count,
            "avg_features": {k: round(float(v), 4) for k, v in avg.items()},
        })
    return out


def analyze_features(features: dict) -> dict:
    if _model is None:
        raise RuntimeError("Engine not loaded.")

    scaler = _preprocessor["scaler"]
    class_labels = _class_labels

    feat_array = np.array([[features[col] for col in FEATURE_COLS]])
    x_scaled = scaler.transform(feat_array)

    probs = predict_proba_single(x_scaled, _model, _model_type)
    pred_idx = int(np.argmax(probs))
    pred_mood = class_labels[pred_idx]

    raw_expl = _explainer.explain(x_scaled, pred_idx, features)
    explanation = build_explanations(raw_expl, pred_mood)

    return {
        "mood": pred_mood,
        "confidence": {class_labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "explanation": explanation,
    }


def get_explore_sample(n: int = 2000) -> list:
    if _tracks is None:
        raise RuntimeError("Engine not loaded.")
    sample = _tracks.sample(min(n, len(_tracks)), random_state=1)
    out = []
    for _, row in sample.iterrows():
        out.append({
            "track_name": str(row["track_name"]),
            "artists": str(row["artists"]),
            "energy": float(row["energy"]),
            "valence": float(row["valence"]),
            "danceability": float(row["danceability"]),
            "acousticness": float(row["acousticness"]),
            "instrumentalness": float(row["instrumentalness"]),
            "tempo_norm": float(row["tempo_norm"]),
            "mood": str(row["predicted_mood"]),
        })
    return out
