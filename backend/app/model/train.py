"""
Trains Random Forest, XGBoost, and a PyTorch MLP on processed track data.
All three models are trained on StandardScaler-transformed features.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_tracks.parquet")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "preprocessor.pkl")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness_norm", "speechiness", "tempo_norm", "valence",
    "energy_valence_ratio", "acoustic_energy_contrast",
    "danceability_tempo_score", "mood_index",
]


class MoodMLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _train_mlp(X_train, y_train, X_val, y_val, num_classes, class_weights_array, device):
    model = MoodMLP(num_classes).to(device)
    weight_tensor = torch.FloatTensor(class_weights_array).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.LongTensor(y_val)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_v)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        val_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        print(f"  epoch {epoch + 1:02d}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print(f"  early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model


def train_all():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    scaler = preprocessor["scaler"]
    class_labels = preprocessor["class_labels"]

    le = LabelEncoder()
    le.classes_ = np.array(class_labels)

    X = scaler.transform(df[FEATURE_COLS].values)
    y_str = df["mood"].values
    y = le.transform(y_str)

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    classes = np.arange(len(class_labels))
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = {i: float(cw[i]) for i in range(len(cw))}

    results = {}
    device = _get_device()
    print(f"Device: {device}")

    # Random Forest
    print("\nTraining Random Forest...")
    rf_dir = os.path.join(MODELS_DIR, "random_forest")
    os.makedirs(rf_dir, exist_ok=True)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_preds_str = le.inverse_transform(rf_preds)
    y_test_str = le.inverse_transform(y_test)
    rf_f1 = f1_score(y_test_str, rf_preds_str, average="macro", zero_division=0)
    rf_acc = accuracy_score(y_test_str, rf_preds_str)
    rf_report = classification_report(y_test_str, rf_preds_str, output_dict=True, zero_division=0)
    print(f"RF   F1 macro={rf_f1:.4f}  acc={rf_acc:.4f}")
    with open(os.path.join(rf_dir, "model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    results["random_forest"] = {"f1_macro": rf_f1, "accuracy": rf_acc, "classification_report": rf_report}

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_dir = os.path.join(MODELS_DIR, "xgboost")
    os.makedirs(xgb_dir, exist_ok=True)
    sample_weights = np.array([cw[yi] for yi in y_train])
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_preds = xgb.predict(X_test)
    xgb_preds_str = le.inverse_transform(xgb_preds)
    xgb_f1 = f1_score(y_test_str, xgb_preds_str, average="macro", zero_division=0)
    xgb_acc = accuracy_score(y_test_str, xgb_preds_str)
    xgb_report = classification_report(y_test_str, xgb_preds_str, output_dict=True, zero_division=0)
    print(f"XGB  F1 macro={xgb_f1:.4f}  acc={xgb_acc:.4f}")
    with open(os.path.join(xgb_dir, "model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    results["xgboost"] = {"f1_macro": xgb_f1, "accuracy": xgb_acc, "classification_report": xgb_report}

    # MLP
    print("\nTraining MLP...")
    mlp_dir = os.path.join(MODELS_DIR, "mlp")
    os.makedirs(mlp_dir, exist_ok=True)
    mlp = _train_mlp(X_train, y_train, X_val, y_val, len(class_labels), cw, device)
    mlp.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        mlp_preds = torch.argmax(mlp(X_test_t), dim=1).cpu().numpy()
    mlp_preds_str = le.inverse_transform(mlp_preds)
    mlp_f1 = f1_score(y_test_str, mlp_preds_str, average="macro", zero_division=0)
    mlp_acc = accuracy_score(y_test_str, mlp_preds_str)
    mlp_report = classification_report(y_test_str, mlp_preds_str, output_dict=True, zero_division=0)
    print(f"MLP  F1 macro={mlp_f1:.4f}  acc={mlp_acc:.4f}")
    torch.save(mlp.state_dict(), os.path.join(mlp_dir, "model.pt"))
    with open(os.path.join(mlp_dir, "config.json"), "w") as f:
        json.dump({"num_classes": len(class_labels)}, f)
    results["mlp"] = {"f1_macro": mlp_f1, "accuracy": mlp_acc, "classification_report": mlp_report}

    return results, le, X_test, y_test


if __name__ == "__main__":
    results, _, _, _ = train_all()
    print("\nDone. Run python -m app.model.evaluate to select the best model.")
