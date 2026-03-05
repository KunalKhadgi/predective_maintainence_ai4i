## Predictive Maintenance using Time-Series ML (AI4I 2020)

This project implements an end-to-end predictive maintenance pipeline using the **AI4I 2020 Predictive Maintenance** dataset. The goal is to predict machine failures from multivariate sensor readings so that maintenance can be scheduled proactively instead of reactively.

### Dataset

- **Source**: AI4I 2020 Predictive Maintenance Dataset
- **Features** (examples):
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Torque [Nm]`
  - `Rotational speed [rpm]`
  - `Tool wear [min]`
- **Target**:
  - `Machine failure` (binary label, highly imbalanced)
- There is no explicit timestamp; rows are treated as a sequence in their given order.

The raw CSV is stored under `data/raw/ai4i2020.csv`.

### Project structure

```text
project_root/
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
├── notebooks/
│   └── eda_ai4i.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── lstm_model.py
│   ├── transformer_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── api/
│   └── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

### Approach

- **Data preprocessing**
  - Drop identifier columns (`UDI`, `Product ID`).
  - Encode categorical `Type` as integers (`L` → 0, `M` → 1, `H` → 2).
  - Standardize all features with `StandardScaler`.
- **Sequence construction**
  - Use a sliding window (`SequenceDataset`) to build sequences of length `seq_len` from the tabular data.
  - The label for each sequence is the `Machine failure` value at the end of the window.
- **Models**
  - `LSTMModel`: single-layer LSTM with a linear head and sigmoid output.
  - `TransformerModel`: linear embedding → Transformer encoder → mean pooling → linear + sigmoid.
  - The default training script uses the **Transformer**.
- **Evaluation**
  - Training script reports **precision–recall AUC** on the full dataset.
  - Dedicated evaluation script computes accuracy, precision, recall, F1, and confusion matrix.

### Tech stack

- Python, PyTorch
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- FastAPI, Uvicorn

---

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Ensure the dataset CSV is available at `data/raw/ai4i2020.csv`.

---

## Training

Train the model (Transformer by default) and save the trained artifact to `models/model.joblib`:

```bash
python src/train.py
```

This will:

- Load and preprocess the dataset using `src/data_preprocessing.py`.
- Construct time-series sequences using `src/dataset.py`.
- Train the Transformer model.
- Report PR-AUC on the full dataset.
- Save a model artifact containing:
  - Trained model weights
  - Feature names and scaler
  - Basic training metadata

---

## Evaluation

To compute classification metrics and a confusion matrix using the saved model:

```bash
python src/evaluate.py
```

This script:

- Reloads the data and preprocessing pipeline.
- Uses the saved model to generate predictions on the full dataset.
- Prints:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix
  - Full `classification_report`

---

## CLI prediction

You can run batch predictions from the command line using `src/predict.py`.

1. Create a JSON file with one or more records containing the required features. Example (`sample_input.json`):

```json
[
  {
    "Type": "L",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100.0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0
  }
]
```

2. Run:

```bash
python src/predict.py --input sample_input.json
```

The script returns a JSON list with `probability` and binary `prediction` for each record.

---

## FastAPI service

The project exposes a minimal FastAPI service under `api/app.py` with a `POST /predict` endpoint.

### Run the API locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Make sure you have trained the model first so `models/model.joblib` exists.

### Request format

Endpoint: `POST /predict`

Body (JSON):

```json
{
  "records": [
    {
      "Type": "L",
      "Air temperature [K]": 300.0,
      "Process temperature [K]": 310.0,
      "Rotational speed [rpm]": 1500.0,
      "Torque [Nm]": 40.0,
      "Tool wear [min]": 100.0,
      "TWF": 0,
      "HDF": 0,
      "PWF": 0,
      "OSF": 0,
      "RNF": 0
    }
  ]
}
```

Response:

```json
{
  "results": [
    {
      "probability": 0.12,
      "prediction": 0
    }
  ]
}
```

The API:

- Validates that all required features are present.
- Applies the same preprocessing pipeline used during training.
- Returns failure probabilities and binary predictions.

---

## Docker usage

You can run the API in a container using the provided `Dockerfile`.

1. **Build the image** (after training and saving `models/model.joblib`):

```bash
docker build -t predictive-maintenance-api .
```

2. **Run the container**:

```bash
docker run -p 8000:8000 predictive-maintenance-api
```

3. Send requests to `http://localhost:8000/predict` as described above.

---

## EDA and exploration

Exploratory analysis of the AI4I dataset is available in the notebook:

- `notebooks/eda_ai4i.ipynb`

It includes:

- Class distribution plots for `Machine failure`.
- Sensor trend visualizations before failure events.

