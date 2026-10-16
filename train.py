"""Train and evaluate the SMS spam classifier.

Select the model using training cross-validation.
Evaluate the selected model on the existing test split.
Save artifacts compatible with the Streamlit application.
"""

import json
import time
from pathlib import Path

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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_DIR = BASE_DIR / "models"
RANDOM_STATE = 42


def make_vectorizer():
    """Keep text preprocessing inside the fitted pipeline."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def load_data():
    """Load the expected CSV and validate its labels."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"label", "message"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "data/spam.csv must contain 'label' and 'message' columns."
        )

    df = (
        df[["label", "message"]]
        .copy()
        .dropna()
        .drop_duplicates()
    )

    invalid_labels = set(df["label"]) - {"ham", "spam"}
    if invalid_labels:
        raise ValueError(
            f"Unexpected labels: {sorted(invalid_labels)}"
        )

    if set(df["label"]) != {"ham", "spam"}:
        raise ValueError("The dataset must contain both ham and spam.")

    if df["message"].astype(str).str.strip().eq("").any():
        raise ValueError(
            "The dataset contains blank messages. Review them before training."
        )

    conflicting = df.groupby("message")["label"].nunique()
    if conflicting.gt(1).any():
        raise ValueError(
            "Some identical messages have conflicting labels. "
            "Review these records before training."
        )

    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_candidates():
    """Create the candidate models and their search spaces."""
    calibration_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Calibrate the entire SVM pipeline so TF-IDF is fitted
    # separately within each calibration training fold.
    svm_pipeline = Pipeline(
        [
            ("tfidf", make_vectorizer()),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    dual="auto",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Naive Bayes": (
            Pipeline(
                [
                    ("tfidf", make_vectorizer()),
                    ("clf", MultinomialNB()),
                ]
            ),
            {
                "tfidf__max_features": [3000, 5000],
                "clf__alpha": [0.1, 0.5, 1.0],
            },
        ),
        "Logistic Regression": (
            Pipeline(
                [
                    ("tfidf", make_vectorizer()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "tfidf__max_features": [3000, 5000],
                "clf__C": [1, 5, 10],
            },
        ),
        "Linear SVM": (
            CalibratedClassifierCV(
                estimator=svm_pipeline,
                cv=calibration_cv,
            ),
            {
                "estimator__tfidf__max_features": [3000, 5000],
                "estimator__clf__C": [0.5, 1, 5],
            },
        ),
    }


def evaluate(name, y_true, y_pred):
    """Calculate metrics with spam as the positive class."""
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "F1-Score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label_encoded"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label_encoded"],
    )

    print(f"Dataset records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")

    selection_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fitted = {}
    cv_scores = {}
    best_params = {}

    # No test predictions are made during candidate selection.
    for name, (estimator, param_grid) in build_candidates().items():
        print(f"\nTraining {name}...")
        started = time.perf_counter()

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1",
            cv=selection_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        fitted[name] = search.best_estimator_
        cv_scores[name] = float(search.best_score_)
        best_params[name] = search.best_params_

        elapsed = time.perf_counter() - started
        print(f"Mean CV F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Elapsed: {elapsed:.1f} seconds")

    print("\nTraining cross-validation comparison:")
    for name, score in sorted(
        cv_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name}: {score:.4f}")

    # Choose the winner using training CV results only.
    selected_name = max(cv_scores, key=cv_scores.get)
    selected_model = fitted[selected_name]

    print(f"\nSelected model: {selected_name}")

    # Evaluate the selected model after the selection is complete.
    predictions = selected_model.predict(X_test)
    test_metrics = evaluate(selected_name, y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\nSelected model test results:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0,
        )
    )
    print(f"Test F1: {test_metrics['F1-Score']:.4f}")
    print(f"Confusion matrix:\n{cm}")

    results = {
        "results": [test_metrics],
        "best_model": selected_name,
        "best_params": best_params[selected_name],
        "selection_metric": "mean training cross-validation F1",
        "cv_scores": cv_scores,
        "confusion_matrix": cm.tolist(),
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
        "evaluation_note": (
            "This test split was used in earlier experiments. "
            "It is not a newly untouched final holdout."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        selected_model,
        MODEL_DIR / "spam_pipeline.pkl",
    )

    with (MODEL_DIR / "results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nSaved models/spam_pipeline.pkl")
    print("Saved models/results.json")
    print("Update README metrics using these new results.")


if __name__ == "__main__":
    main()
