import os
import pandas as pd
import numpy as np
from app.data.loader import load_spotify_data

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "processed_tracks.parquet")

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]

AUDIO_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
]


def _assign_mood(row) -> str:
    v = row["valence"]
    e = row["energy"]
    d = row["danceability"]
    a = row["acousticness"]
    i = row["instrumentalness"]

    if v > 0.7 and e > 0.6:
        return "Happy"
    if e > 0.8 and d > 0.7:
        return "Energetic"
    if v < 0.3 and e < 0.5:
        return "Melancholic"
    if i > 0.4 and 0.3 <= e <= 0.7:
        return "Focused"
    if a > 0.6 and e < 0.4:
        return "Calm"
    if e > 0.8 and v < 0.4:
        return "Intense"
    return "Neutral"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["track_id", "track_name", "artists", "album_name"]
    df = df[meta_cols + AUDIO_COLS].copy()

    df = df.dropna(subset=AUDIO_COLS)
    df = df.drop_duplicates(subset=["track_id"])
    df = df.reset_index(drop=True)

    # Store raw values needed for display and engineered features
    df["tempo_raw"] = df["tempo"]

    # Engineered features use raw tempo before normalization
    df["energy_valence_ratio"] = df["energy"] / (df["valence"] + 0.001)
    df["acoustic_energy_contrast"] = df["acousticness"] - df["energy"]
    df["danceability_tempo_score"] = df["danceability"] * (df["tempo"] / 200.0)
    df["mood_index"] = (
        df["valence"] * 0.4
        + df["energy"] * 0.3
        + df["danceability"] * 0.2
        + df["acousticness"] * 0.1
    )

    # Mood labels assigned from raw values
    df["mood"] = df.apply(_assign_mood, axis=1)

    # Normalize loudness (-60 to 0) and tempo (0 to 250) into 0-1
    df["loudness_norm"] = ((df["loudness"] - (-60)) / 60.0).clip(0, 1)
    df["tempo_norm"] = (df["tempo"] / 250.0).clip(0, 1)

    df = df.drop(columns=["loudness", "tempo"])

    dist = df["mood"].value_counts()
    print("Mood label distribution:")
    print(dist.to_string())
    for mood, count in dist.items():
        if count < 500:
            print(f"WARNING: {mood} has only {count} samples (fewer than 500)")

    return df


if __name__ == "__main__":
    print("Loading dataset...")
    raw = load_spotify_data()
    print(f"Raw rows: {len(raw)}")

    print("Preprocessing...")
    df = preprocess(raw)
    print(f"Processed rows: {len(df)}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
