import os
import pickle
import json
import numpy as np
import torch
from app.model.train import MoodMLP

PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")

_preprocessor = None
_active_model = None
_active_model_type = None
_all_models = {}


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        with open(PREPROCESSOR_PATH, "rb") as f:
            _preprocessor = pickle.load(f)
    return _preprocessor


def load_all_models():
    global _all_models
    if _all_models:
        return _all_models

    rf_path = os.path.join(MODELS_DIR, "random_forest", "model.pkl")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            _all_models["random_forest"] = pickle.load(f)

    xgb_path = os.path.join(MODELS_DIR, "xgboost", "model.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            _all_models["xgboost"] = pickle.load(f)

    mlp_pt = os.path.join(MODELS_DIR, "mlp", "model.pt")
    mlp_cfg = os.path.join(MODELS_DIR, "mlp", "config.json")
    if os.path.exists(mlp_pt):
        with open(mlp_cfg) as f:
            cfg = json.load(f)
        mlp = MoodMLP(cfg["num_classes"])
        mlp.load_state_dict(torch.load(mlp_pt, map_location="cpu"))
        mlp.eval()
        _all_models["mlp"] = mlp

    return _all_models


def load_active_model():
    global _active_model, _active_model_type

    if _active_model is not None:
        return _active_model, _active_model_type

    if not os.path.exists(METRICS_PATH):
        raise RuntimeError("metrics.json not found. Run python -m app.model.evaluate first.")

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    best = metrics["best_model"]
    models = load_all_models()

    if best not in models:
        raise RuntimeError(f"Best model '{best}' not found in saved_models/.")

    _active_model = models[best]
    _active_model_type = best
    print(f"Active model: {best}")
    return _active_model, _active_model_type


def predict_proba_single(features_scaled: np.ndarray, model, model_type: str) -> np.ndarray:
    """Returns probability array of shape (n_classes,)."""
    x = features_scaled.reshape(1, -1)
    if model_type == "mlp":
        device = _get_device()
        model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(x).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs
    else:
        return model.predict_proba(x)[0]


def predict_batch_labels(X_scaled: np.ndarray, model, model_type: str, class_labels: list) -> list:
    """Returns list of mood label strings for a batch."""
    if model_type == "mlp":
        device = _get_device()
        model.to(device)
        model.eval()
        batch_size = 2048
        all_preds = []
        for i in range(0, len(X_scaled), batch_size):
            chunk = torch.FloatTensor(X_scaled[i : i + batch_size]).to(device)
            with torch.no_grad():
                preds = torch.argmax(model(chunk), dim=1).cpu().numpy()
            all_preds.extend(preds)
        return [class_labels[p] for p in all_preds]
    else:
        preds = model.predict(X_scaled)
        # sklearn tree models return class labels directly when fit with string targets,
        # but we fit with integer indices via LabelEncoder, so convert back
        return [class_labels[int(p)] for p in preds]
