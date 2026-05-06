"""
Fits the StandardScaler on all processed tracks and saves preprocessor.pkl.
Run this after preprocessor.py and before train.py.
"""
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_tracks.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]

CLASS_LABELS = ["Calm", "Energetic", "Focused", "Happy", "Intense", "Melancholic", "Neutral"]


def build_preprocessor():
    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    scaler.fit(X)

    payload = {
        "feature_cols": FEATURE_COLS,
        "class_labels": CLASS_LABELS,
        "scaler": scaler,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Scaler fitted on {len(X)} samples")
    print(f"Saved preprocessor to {OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    build_preprocessor()
