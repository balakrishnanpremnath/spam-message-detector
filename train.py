"""
Upgraded training script for the Spam Message Detector.

Improvements over v1:
- Text cleaning + stemming (preprocessing.py) before TF-IDF
- GridSearchCV hyperparameter tuning per model (5-fold CV, scored on F1)
- LinearSVC wrapped in CalibratedClassifierCV so the app can show a
  confidence score, not just a hard label
- Single sklearn Pipeline (clean -> TF-IDF -> classifier) saved as one
  artifact, so the app doesn't need to duplicate preprocessing logic
- Saves a metrics report (models/results.json) used by the app's
  "model info" panel
"""

import json
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text

# ---------------------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------------------
df = pd.read_csv("data/spam.csv")
df = df[["label", "message"]].copy().dropna().drop_duplicates()
df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
print(f"Dataset shape after cleaning: {df.shape}")

X = df["message"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 2. Build pipelines (cleaning happens inside the vectorizer)
# ---------------------------------------------------------------------


def make_vectorizer():
    return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)


param_grids = {
    "Naive Bayes": (
        Pipeline([("tfidf", make_vectorizer()), ("clf", MultinomialNB())]),
        {
            "tfidf__max_features": [3000, 5000],
            "clf__alpha": [0.1, 0.5, 1.0],
        },
    ),
    "Logistic Regression": (
        Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        {
            "tfidf__max_features": [3000, 5000],
            "clf__C": [1, 5, 10],
        },
    ),
    "Linear SVM": (
        Pipeline(
            [
                ("tfidf", make_vectorizer()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", dual="auto"), cv=3
                    ),
                ),
            ]
        ),
        {
            "tfidf__max_features": [3000, 5000],
            "clf__estimator__C": [0.5, 1, 5],
        },
    ),
}

# ---------------------------------------------------------------------
# 3. Grid search each model
# ---------------------------------------------------------------------


def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }


results = []
fitted = {}
best_params_log = {}

for name, (pipe, grid) in param_grids.items():
    t0 = time.time()
    search = GridSearchCV(pipe, grid, scoring="f1", cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_pipe = search.best_estimator_
    pred = best_pipe.predict(X_test)

    row = evaluate(name, y_test, pred)
    results.append(row)
    fitted[name] = best_pipe
    best_params_log[name] = search.best_params_

    print(f"\n{name} — best params: {search.best_params_} (tuned in {elapsed:.1f}s)")
    print(classification_report(y_test, pred, target_names=["Ham", "Spam"]))

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------
# 4. Pick best model, save it as ONE pipeline artifact
# ---------------------------------------------------------------------
best_row = results_df.loc[results_df["F1-Score"].idxmax()]
best_name = best_row["Model"]
best_pipeline = fitted[best_name]

cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
print(f"\nBest model: {best_name} (F1-Score: {best_row['F1-Score']:.4f})")
print(f"Confusion matrix:\n{cm}")

joblib.dump(best_pipeline, "models/spam_pipeline.pkl")

with open("models/results.json", "w") as f:
    json.dump(
        {
            "results": results_df.to_dict(orient="records"),
            "best_model": best_name,
            "best_params": best_params_log[best_name],
            "confusion_matrix": cm.tolist(),
            "dataset_size": len(df),
            "test_size": len(y_test),
        },
        f,
        indent=2,
    )

print("\nSaved models/spam_pipeline.pkl (cleaning + TF-IDF + classifier, single artifact)")
