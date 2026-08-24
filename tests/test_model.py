import joblib
import pytest

from preprocessing import clean_text  # noqa: F401  (needed to unpickle the pipeline)

MODEL_PATH = "models/spam_pipeline.pkl"


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


def test_pipeline_loads(pipeline):
    assert pipeline is not None


def test_predicts_obvious_ham_as_ham(pipeline):
    pred = pipeline.predict(["Hey, are you free for lunch tomorrow?"])[0]
    assert pred == 0


def test_predicts_obvious_spam_as_spam(pipeline):
    pred = pipeline.predict(
        ["WINNER! You have been selected to receive a FREE $1000 CASH prize. Call now!"]
    )[0]
    assert pred == 1


def test_predict_proba_shape(pipeline):
    proba = pipeline.predict_proba(["test message"])
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0][1] <= 1.0


def test_batch_prediction(pipeline):
    messages = ["Call me later", "URGENT you have won a prize, click here now!"]
    preds = pipeline.predict(messages)
    assert len(preds) == 2
