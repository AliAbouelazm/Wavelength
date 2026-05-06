import numpy as np
import shap

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]

BOUNDED_FEATURES = {
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence", "mood_index",
}

FEATURE_DISPLAY = {
    "acousticness": "Acousticness",
    "danceability": "Danceability",
    "energy": "Energy",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "loudness_norm": "Loudness",
    "speechiness": "Speechiness",
    "tempo_norm": "Tempo",
    "valence": "Valence",
    "energy_valence_ratio": "Energy/Valence ratio",
    "acoustic_energy_contrast": "Acoustic/Energy contrast",
    "danceability_tempo_score": "Danceability tempo score",
    "mood_index": "Mood index",
}


def _level(feature: str, value: float) -> str:
    if feature in BOUNDED_FEATURES:
        if value > 0.66:
            return "High"
        if value < 0.33:
            return "Low"
        return "Moderate"
    return "Strong" if abs(value) > 0.5 else "Moderate"


def _build_phrase(feature: str, raw_value: float, direction: str, mood: str) -> str:
    level = _level(feature, raw_value)
    name = FEATURE_DISPLAY.get(feature, feature)
    val_str = f"{raw_value:.2f}"
    if direction == "increases":
        return f"{level} {name} ({val_str}) drives {mood} classification"
    return f"{level} {name} ({val_str}) works against {mood} classification"


class TreeModelExplainer:
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X_scaled: np.ndarray, class_idx: int, raw_values: dict) -> list:
        sv = self.explainer.shap_values(X_scaled)

        # Handle both list-of-arrays (RF) and 3D array (XGBoost)
        if isinstance(sv, list):
            class_shap = sv[class_idx][0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            class_shap = sv[0, :, class_idx]
        else:
            class_shap = sv[0]

        return _top3(class_shap, raw_values)


class PermutationExplainer:
    """Permutation importance for MLP -- computed per prediction, not globally."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def explain(self, X_scaled: np.ndarray, class_idx: int, raw_values: dict) -> list:
        import torch

        self.model.eval()
        x = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            base_prob = torch.softmax(self.model(x), dim=1)[0, class_idx].item()

        importances = np.zeros(len(FEATURE_COLS))
        for i in range(len(FEATURE_COLS)):
            x_perm = x.clone()
            # Shuffle that feature across the single sample by zeroing it
            x_perm[0, i] = 0.0
            with torch.no_grad():
                perm_prob = torch.softmax(self.model(x_perm), dim=1)[0, class_idx].item()
            importances[i] = base_prob - perm_prob

        return _top3(importances, raw_values)


def _top3(shap_vals: np.ndarray, raw_values: dict) -> list:
    order = np.argsort(np.abs(shap_vals))[::-1][:3]
    max_mag = np.abs(shap_vals[order[0]]) if len(order) > 0 else 1.0

    result = []
    for idx in order:
        feat = FEATURE_COLS[idx]
        sv = shap_vals[idx]
        raw_val = raw_values.get(feat, 0.0)
        direction = "increases" if sv >= 0 else "decreases"
        magnitude = float(abs(sv) / max_mag) if max_mag != 0 else 0.0
        result.append({
            "feature": feat,
            "direction": direction,
            "magnitude": round(magnitude, 4),
            "phrase": "",  # filled in by explain_track with mood context
            "_sv": sv,
            "_raw": raw_val,
        })
    return result


def build_explanations(raw_result: list, mood: str) -> list:
    out = []
    for item in raw_result:
        phrase = _build_phrase(item["feature"], item["_raw"], item["direction"], mood)
        out.append({
            "feature": item["feature"],
            "direction": item["direction"],
            "magnitude": item["magnitude"],
            "phrase": phrase,
        })
    return out
