"""
Loads all three saved models, evaluates on the test split, writes metrics.json,
and marks the best model by F1 macro as active.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.model.train import MoodMLP

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_tracks.parquet")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate_all():
    print("Loading data and preprocessor...")
    df = pd.read_parquet(DATA_PATH)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    scaler = preprocessor["scaler"]
    class_labels = preprocessor["class_labels"]

    le = LabelEncoder()
    le.classes_ = np.array(class_labels)

    X = scaler.transform(df[FEATURE_COLS].values)
    y = le.transform(df["mood"].values)

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    y_test_str = le.inverse_transform(y_test)

    device = _get_device()
    results = {}

    # Random Forest
    rf_path = os.path.join(MODELS_DIR, "random_forest", "model.pkl")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf = pickle.load(f)
        preds = le.inverse_transform(rf.predict(X_test))
        f1 = f1_score(y_test_str, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test_str, preds)
        report = classification_report(y_test_str, preds, output_dict=True, zero_division=0)
        results["random_forest"] = {"f1_macro": round(f1, 4), "accuracy": round(acc, 4), "classification_report": report}
        print(f"Random Forest  F1={f1:.4f}  acc={acc:.4f}")

    # XGBoost
    xgb_path = os.path.join(MODELS_DIR, "xgboost", "model.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            xgb = pickle.load(f)
        preds = le.inverse_transform(xgb.predict(X_test))
        f1 = f1_score(y_test_str, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test_str, preds)
        report = classification_report(y_test_str, preds, output_dict=True, zero_division=0)
        results["xgboost"] = {"f1_macro": round(f1, 4), "accuracy": round(acc, 4), "classification_report": report}
        print(f"XGBoost        F1={f1:.4f}  acc={acc:.4f}")

    # MLP
    mlp_pt = os.path.join(MODELS_DIR, "mlp", "model.pt")
    mlp_cfg = os.path.join(MODELS_DIR, "mlp", "config.json")
    if os.path.exists(mlp_pt):
        with open(mlp_cfg) as f:
            cfg = json.load(f)
        mlp = MoodMLP(cfg["num_classes"])
        mlp.load_state_dict(torch.load(mlp_pt, map_location="cpu"))
        mlp.eval()
        with torch.no_grad():
            logits = mlp(torch.FloatTensor(X_test))
            preds = le.inverse_transform(torch.argmax(logits, dim=1).numpy())
        f1 = f1_score(y_test_str, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test_str, preds)
        report = classification_report(y_test_str, preds, output_dict=True, zero_division=0)
        results["mlp"] = {"f1_macro": round(f1, 4), "accuracy": round(acc, 4), "classification_report": report}
        print(f"MLP            F1={f1:.4f}  acc={acc:.4f}")

    if not results:
        raise RuntimeError("No trained models found. Run python -m app.model.train first.")

    best_model = max(results, key=lambda m: results[m]["f1_macro"])
    print(f"\nBest model: {best_model} (F1={results[best_model]['f1_macro']})")

    output = {"models": results, "best_model": best_model}
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    return output


if __name__ == "__main__":
    evaluate_all()
