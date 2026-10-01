# Spam Message Detector

An NLP application that classifies SMS messages as **Spam** or **Ham**
using text preprocessing, TF-IDF, and machine learning.

The project includes model comparison, a Streamlit interface,
CSV batch prediction, automated tests, and a GitHub Actions workflow.

## Project Overview

Spam detection helps identify unwanted messages while reducing the
chance of incorrectly flagging legitimate communication.

This project compares three classification approaches:

- Multinomial Naive Bayes
- Logistic Regression
- Calibrated Linear SVM

**Spam** means an unwanted message.  
**Ham** means a legitimate message.

## Application Features

- Classify an individual message.
- Display a predicted spam probability.
- Upload a CSV file for batch predictions.
- Download batch prediction results.
- View model metrics and a confusion matrix.
- Retrain models using the provided training script.

## Dataset

**Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

| Detail | Value |
|---|---|
| Original dataset | 5,572 SMS messages |
| Records after cleaning in the saved experiment | 5,169 |
| Training records | 4,135 |
| Test records | 1,034 |
| Target labels | Ham and Spam |
| Split | 80% training, 20% test |
| Sampling | Stratified by label |
| Split random state | 42 |

The training script removes missing values and duplicate
label-message pairs before splitting the data.

The reported evaluation applies to SMS messages. Performance on
email messages has not been established.

## NLP Pipeline

1. Clean and normalize message text.
2. Apply stemming during preprocessing.
3. Convert text into TF-IDF features.
4. Train and tune classification models.
5. Generate predictions and evaluation metrics.
6. Save the selected pipeline for use in the application.

Preprocessing is applied through the TF-IDF vectorizer inside a
Scikit-learn pipeline, keeping training and prediction consistent.

### Feature Extraction

The TF-IDF vectorizer uses:

- Unigrams and bigrams
- Sublinear term frequency
- A tuned maximum vocabulary size

### Model Tuning

Each model uses `GridSearchCV` with five-fold cross-validation
and F1 scoring on the training data.

The Linear SVM is wrapped in `CalibratedClassifierCV` to provide
probability estimates.

## Saved Experiment Results

The following values are recorded in `models/results.json`.

Precision, recall, and F1 treat **Spam** as the positive class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.985 | 0.968 | 0.916 | 0.941 |
| Logistic Regression | 0.988 | 0.969 | 0.939 | 0.953 |
| Linear SVM | 0.989 | 0.976 | 0.939 | 0.957 |

The saved experiment selected Linear SVM, with:

- `C = 1`
- `max_features = 5000`

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 900 | 3 |
| Spam | 8 | 123 |

In this experiment:

- 900 legitimate messages were correctly classified.
- 123 spam messages were correctly detected.
- 3 legitimate messages were incorrectly flagged as spam.
- 8 spam messages were missed.

### Evaluation Note

The current training script tunes each model using training
cross-validation, then selects the final model using test-set F1.

Because the test set influences final model selection, these results
should be treated as development results rather than an independent
final estimate.

A planned improvement is to select the winning model using training
cross-validation and evaluate it on a genuinely untouched holdout.

## Repository Guide

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `train.py` | Training, tuning, evaluation, and model saving |
| `preprocessing.py` | Shared text preprocessing |
| `data/spam.csv` | Training dataset |
| `models/spam_pipeline.pkl` | Saved prediction pipeline |
| `models/results.json` | Saved metrics and model information |
| `notebooks/spam_detection.ipynb` | Notebook walkthrough |
| `tests/` | Preprocessing and model tests |
| `requirements.txt` | Application dependencies |
| `requirements-dev.txt` | Development dependencies |
| `Dockerfile` | Container configuration |
| `.github/workflows/ci.yml` | Automated validation workflow |

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/balakrishnanpremnath/spam-message-detector.git
cd spam-message-detector
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

Run commands from the repository root so relative file paths resolve
correctly.

## Using the Application

### Single Message

Enter a message in the single-message tab to view its predicted
label and confidence.

Confidence reflects the model's estimated probability. It does not
guarantee that the prediction is correct.

### CSV Batch Prediction

Prepare a CSV containing a `message` column:

```csv
message
"Are we meeting tomorrow?"
"Congratulations! Claim your prize now!"
"Please call me when you arrive."
```

Upload the file in the batch tab, review the predictions, and
download the results.

### Model Information

The model-information tab displays saved evaluation results from
`models/results.json`.

## Retrain the Models

```bash
python train.py
```

The script:

1. Loads and cleans the dataset.
2. Creates a stratified training/test split.
3. Tunes the three candidate models.
4. Calculates evaluation metrics.
5. Selects and saves a pipeline.
6. Writes the results to JSON.

Retraining overwrites:

- `models/spam_pipeline.pkl`
- `models/results.json`

After retraining, update the README metrics to match the new
saved results.

## Run Tests

Install application and development dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run the tests:

```bash
python -m pytest tests/ -v
```

Run the lint check used by CI:

```bash
flake8 . --max-line-length=100 --extend-exclude=notebooks --exclude=.github
```

## Run with Docker

With Docker installed and running:

```bash
docker build -t spam-message-detector .
docker run --rm -p 8501:8501 spam-message-detector
```

Open:

```text
http://localhost:8501
```

## Continuous Integration

The GitHub Actions workflow is configured for pushes and pull
requests targeting `main`.

It performs:

1. Dependency installation
2. Lint checks
3. Automated tests
4. Docker image build after successful tests

Check the repository's **Actions** tab for the latest execution status.

## Tools and Technologies

- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF
- Streamlit
- Pytest
- Docker
- GitHub Actions

## Limitations

- Evaluation uses a single SMS dataset and one train/test split.
- Final model selection currently uses test-set performance.
- Unfamiliar languages, wording, or spam patterns may reduce accuracy.
- Some legitimate messages may be flagged, and some spam may be missed.
- Calibrated probabilities can still be inaccurate on new types of data.
- The application is a portfolio demonstration, not a validated
  production filtering service.

## Planned Improvements

- Select the final model using training cross-validation.
- Evaluate on an untouched holdout dataset.
- Review false positives and false negatives.
- Check for near-duplicate messages across evaluation splits.
- Add application screenshots and a verified demonstration link.
- Compare TF-IDF with alternative text representations.
- Add experiment tracking and a prediction API.

## Author

**Balakrishnan Premnath**  
BSc (Hons) in Data Science — Sri Lanka Technology Campus

[GitHub](https://github.com/balakrishnanpremnath) |
[LinkedIn](https://www.linkedin.com/in/balakrishnan-premnath)
