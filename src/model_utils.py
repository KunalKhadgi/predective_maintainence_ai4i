import os
from typing import Tuple

import joblib
import torch

from lstm_model import LSTMModel
from transformer_model import TransformerModel


DEFAULT_MODEL_PATH = os.path.join("models", "model.joblib")


def load_model_artifact(path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Load a trained model artifact saved by src/train.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train the model first with 'python src/train.py'."
        )
    return joblib.load(path)


def build_model_from_artifact(artifact: dict) -> torch.nn.Module:
    """
    Recreate the model instance from a saved artifact dictionary.
    """
    model_type = artifact.get("model_type", "transformer")
    input_dim = artifact["input_dim"]

    if model_type == "transformer":
        model = TransformerModel(input_dim=input_dim)
    elif model_type == "lstm":
        model = LSTMModel(input_dim=input_dim)
    else:
        raise ValueError(f"Unsupported model_type in artifact: {model_type}")

    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model


def load_trained_model(
    path: str = DEFAULT_MODEL_PATH,
) -> Tuple[torch.nn.Module, dict]:
    """
    Convenience helper to load the model and associated metadata.
    """
    artifact = load_model_artifact(path)
    model = build_model_from_artifact(artifact)
    return model, artifact

