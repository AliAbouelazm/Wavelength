# Wavelength

Wavelength learns what sonic features drive emotional states in music and surfaces tracks that match how a user wants to feel. A user picks a mood, optionally adjusts energy, mood tone, and tempo sliders, and receives up to 20 recommendations backed by a real trained classifier. Each result includes a plain-English explanation of why the model selected that track, powered by SHAP for tree models and permutation importance for the neural network.

## Architecture

```
Browser (React 18 + Vite)
         |
         | HTTP/JSON
         v
  FastAPI (Python 3.11)
         |
         +-- GET  /moods            --> mood profiles and track counts
         +-- POST /recommend        --> top-20 tracks with SHAP explanations
         +-- POST /analyze          --> predict mood from raw audio features
         +-- GET  /explore          --> 2000-track sample for scatter plot
         +-- GET  /model/metrics    --> model comparison table
         +-- GET  /health           --> status check
         |
         +-- Active model (RF / XGBoost / MLP, best by F1 macro)
         +-- processed_tracks.parquet (~100k tracks with engineered features)
```

## Component table

| Component | Technology | Purpose |
|---|---|---|
| Frontend | React 18, Vite, CSS Modules | Mood selector, results grid, feature space explorer |
| Backend | FastAPI, Python 3.11 | REST API, rate limiting, model serving |
| ML models | Random Forest, XGBoost, PyTorch MLP | 7-class mood classification |
| Training data | maharshipandya/spotify-tracks-dataset (HuggingFace) | ~114k Spotify tracks with audio features |
| Explainability | SHAP TreeExplainer, permutation importance | Per-track feature attribution |
| Charts | Recharts | Radar charts, scatter plot |
| Deployment | Render (backend), Vercel (frontend) | Production hosting |

## Mood label engineering

Mood labels are assigned from raw Spotify audio features using hard rules applied in priority order. The first matching rule wins.

| Mood | Rule |
|---|---|
| Happy | valence > 0.7 and energy > 0.6 |
| Energetic | energy > 0.8 and danceability > 0.7 |
| Melancholic | valence < 0.3 and energy < 0.5 |
| Focused | instrumentalness > 0.4 and energy between 0.3 and 0.7 |
| Calm | acousticness > 0.6 and energy < 0.4 |
| Intense | energy > 0.8 and valence < 0.4 |
| Neutral | everything else |

Four additional features are engineered on top of the nine raw features: energy/valence ratio, acoustic/energy contrast, danceability tempo score, and a composite mood index. These are fed to the classifier alongside the normalized audio features.

## Model comparison

Fill in after running `python -m app.model.evaluate`:

| Model | F1 macro | Accuracy |
|---|---|---|
| Random Forest | 0.9996 | 0.9998 |
| XGBoost | 0.9962 | 0.9973 |
| MLP | 0.9354 | 0.9547 |

The model with the highest F1 macro is automatically selected as active.

## How SHAP explanations work

For Random Forest and XGBoost, SHAP TreeExplainer computes exact Shapley values: the contribution of each feature to the model's output for a specific prediction. A positive SHAP value means the feature pushed the prediction toward that mood class; a negative value means it pushed away from it.

For the MLP, permutation importance is used per prediction: each feature is zeroed out one at a time and the drop in the predicted class probability is recorded as the importance score.

The top three features by absolute contribution are shown for each track, along with a plain-English phrase describing what the feature value was and what effect it had.

## Running locally

Install backend dependencies and run the data pipeline:

```bash
cd wavelength/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m app.data.preprocessor
python -m app.model.features
python -m app.model.train
python -m app.model.evaluate
```

Start the API server:

```bash
uvicorn app.main:app --reload --port 8000
```

Install frontend dependencies and start the dev server:

```bash
cd wavelength/frontend
npm install
npm run dev
```

The app runs at http://localhost:5173. Create `frontend/.env.local` to override the API URL:

```
VITE_API_URL=http://localhost:8000
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `VITE_API_URL` | Backend base URL used by the frontend | `http://localhost:8000` |

## Deployment

### Backend on Render

1. Push the repo to GitHub.
2. Create a new Web Service on Render, connect the repo, set root directory to `wavelength/backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Upload trained model artifacts (`saved_models/`, `preprocessor.pkl`, `metrics.json`, `processed_tracks.parquet`) via Render disk or commit them to the repo.

### Frontend on Vercel

1. Import the repo on Vercel.
2. Set root directory to `wavelength/frontend`.
3. Add environment variable `VITE_API_URL` pointing to your Render service URL.
4. Deploy. Vercel detects Vite automatically.

## Live demo

https://wavelength.aliabouelazm.com
