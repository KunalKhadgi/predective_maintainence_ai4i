import argparse
import json
from typing import List

import torch

from data_preprocessing import preprocess_input_records
from model_utils import load_trained_model


def run_inference(records: List[dict]):
    model, artifact = load_trained_model()
    features = artifact["features"]
    scaler = artifact["scaler"]

    X = preprocess_input_records(records, features=features, scaler=scaler)

    # Treat each record as a sequence of length 1 for inference
    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        probs = model(x_tensor).squeeze().numpy()

    # Ensure probs is a 1D array
    if probs.ndim == 0:
        probs = probs.reshape(1)

    preds = (probs >= 0.5).astype(int)

    results = []
    for p, y_hat in zip(probs, preds):
        results.append({"probability": float(p), "prediction": int(y_hat)})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using the trained predictive maintenance model."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON file containing a list of records.",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = [data]
    else:
        records = list(data)

    results = run_inference(records)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

