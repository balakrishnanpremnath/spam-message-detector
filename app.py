import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_text  # noqa: F401  (needed so joblib can unpickle the pipeline)

MODEL_PATH = Path("models/spam_pipeline.pkl")
RESULTS_PATH = Path("models/results.json")

st.set_page_config(page_title="Spam Message Detector", page_icon="📩", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return None


pipeline = load_pipeline()
results = load_results()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📩 Spam Message Detector")
st.write(
    "Enter an SMS or email message and the model will predict whether it is "
    "Spam or Not Spam, with a confidence score."
)

tab_single, tab_batch, tab_about = st.tabs(["Check a message", "Batch check (CSV)", "Model info"])

# ---------------------------------------------------------------------
# Tab 1: single message
# ---------------------------------------------------------------------
with tab_single:

    if "single_message" not in st.session_state:
        st.session_state.single_message = ""

    def set_example(example):
        st.session_state.single_message = example

    message = st.text_area(
        "Enter your message:",
        height=120,
        key="single_message"
    )

    examples = [
        "Hey, are you coming to class tomorrow?",
        "Congratulations! You have won a $1000 prize. Click now!",
        "Can you send me the assignment?",
        "URGENT! Claim your FREE reward today!",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        col.button(
            ex[:18] + "…",
            help=ex,
            use_container_width=True,
            on_click=set_example,
            args=(ex,),
        )

    if st.button("Analyze Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message.")
        else:
            proba = pipeline.predict_proba([message])[0]
            spam_prob = proba[1]
            prediction = int(spam_prob >= 0.5)

            if prediction == 1:
                st.error(f"🚨 SPAM MESSAGE — {spam_prob:.0%} confidence")
            else:
                st.success(f"✅ NOT SPAM — {1 - spam_prob:.0%} confidence")

            st.progress(float(spam_prob), text=f"Spam probability: {spam_prob:.0%}")

            st.session_state.history.insert(
                0,
                {
                    "message": message,
                    "prediction": "SPAM" if prediction else "HAM",
                    "spam_probability": round(float(spam_prob), 3),
                },
            )

    if st.session_state.history:
        st.divider()
        st.caption("Recent checks (this session)")
        hist_df = pd.DataFrame(st.session_state.history[:10])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------
# Tab 2: batch prediction from CSV
# ---------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with a `message` column to classify many messages at once.")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            batch_df = None

        if batch_df is not None:
            if "message" not in batch_df.columns:
                st.error("CSV must contain a column named `message`.")
            else:
                probs = pipeline.predict_proba(batch_df["message"].astype(str))[:, 1]
                batch_df["spam_probability"] = probs.round(3)
                batch_df["prediction"] = ["SPAM" if p >= 0.5 else "HAM" for p in probs]

                st.dataframe(batch_df, use_container_width=True, hide_index=True)

                spam_count = (batch_df["prediction"] == "SPAM").sum()
                st.caption(f"{spam_count} of {len(batch_df)} messages flagged as spam.")

                st.download_button(
                    "Download results as CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------------
# Tab 3: model info
# ---------------------------------------------------------------------
with tab_about:
    if results:
        st.subheader(f"Best model: {results['best_model']}")
        st.write(
            f"Trained on {results['dataset_size']} messages, "
            f"tested on {results['test_size']}."
        )

        res_df = pd.DataFrame(results["results"]).set_index("Model")
        st.dataframe(res_df.style.format("{:.3f}"), use_container_width=True)

        cm = results["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm, index=["Actual Ham", "Actual Spam"], columns=["Predicted Ham", "Predicted Spam"]
        )
        st.write("Confusion matrix (best model):")
        st.dataframe(cm_df, use_container_width=True)

        st.caption(f"Best hyperparameters: {results['best_params']}")
    else:
        st.info("Run `python train.py` to generate model metrics.")
