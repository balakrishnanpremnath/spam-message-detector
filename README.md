# 📩 Spam Message Detection

A production-grade NLP project that classifies SMS/email messages as **Spam** or **Ham**, with a tuned Linear SVM (F1 = 0.957), a Streamlit app with confidence scores + batch prediction, automated tests, and Docker/CI support.

## Highlights

- **Cleaned + stemmed text** (URLs, emails, phone numbers normalized; Porter stemming) before vectorization
- **Hyperparameter tuning** — 5-fold `GridSearchCV` per model, scored on F1
- **Calibrated confidence scores** — `CalibratedClassifierCV` wraps the SVM so predictions come with a spam probability, not just a hard label
- **Single deployable artifact** — cleaning, TF-IDF, and the classifier are one sklearn `Pipeline`, saved as `models/spam_pipeline.pkl`
- **App**: single-message check, CSV batch prediction with downloadable results, and a model-info tab
- **Tests**: 11 unit tests covering preprocessing and model behavior
- **Docker + GitHub Actions CI**: lint → test → Docker build on every push

## Results (tuned, on the SMS Spam Collection dataset — 5,169 unique messages)

| Model                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Naive Bayes           | 0.985    | 0.968     | 0.916  | 0.941    |
| Logistic Regression   | 0.988    | 0.969     | 0.939  | 0.953    |
| **Linear SVM (best)** | **0.989**| 0.976     | 0.939  | **0.957**|

Best hyperparameters: `C=1`, `max_features=5000`.

Confusion matrix (best model, 1,034 held-out test messages):

```
              Predicted Ham   Predicted Spam
Actual Ham         900              3
Actual Spam          8            123
```

## Project Structure

```
spam-message-detector/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_pipeline.pkl      # cleaning + TF-IDF + classifier, one artifact
│   └── results.json           # metrics shown in the app's "Model info" tab
│
├── notebooks/
│   └── spam_detection.ipynb   # full walkthrough, executed with real outputs
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── .github/workflows/ci.yml   # lint -> test -> docker build
├── Dockerfile
├── .dockerignore
├── .flake8
├── preprocessing.py           # shared cleaning/stemming used by train.py and app.py
├── app.py                     # Streamlit app
├── train.py                   # full training + tuning pipeline
├── requirements.txt
├── requirements-dev.txt       # + pytest, flake8
└── .gitignore
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has three tabs:
- **Check a message** — single prediction with a confidence bar and session history
- **Batch check (CSV)** — upload a CSV with a `message` column, get predictions + a downloadable results file
- **Model info** — the metrics table and confusion matrix above, read live from `models/results.json`

### Retrain from scratch

```bash
python train.py
```

This re-tunes all three models with `GridSearchCV` and overwrites `models/spam_pipeline.pkl` and `models/results.json`.

### Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
flake8 .
```

### Run with Docker

```bash
docker build -t spam-message-detector .
docker run -p 8501:8501 spam-message-detector
```

Then open `http://localhost:8501`.

### Deploy

- **Streamlit Community Cloud**: push this repo to GitHub, point Streamlit Cloud at `app.py` — no extra config needed since `models/spam_pipeline.pkl` is already committed.
- **Any container host** (Render, Fly.io, Cloud Run, ECS): use the included `Dockerfile` directly.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. Install dependencies
2. Lint with `flake8`
3. Run the test suite with `pytest`
4. Build the Docker image (verifies the container actually builds)

## Technologies

Python · Pandas · Scikit-learn · NLTK (stemming) · Streamlit · Docker · GitHub Actions

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 SMS messages labeled `ham` or `spam`.

## Next steps (ideas, not implemented)

- Swap TF-IDF for a small transformer embedding (e.g. `sentence-transformers`) and compare
- Add a `/predict` FastAPI endpoint alongside the Streamlit UI for programmatic access
- Track experiments with MLflow instead of a flat `results.json`
