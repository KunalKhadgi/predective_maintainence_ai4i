# Predictive Maintenance using Time-Series ML (AI4I 2020)

This project implements an end-to-end predictive maintenance pipeline using the AI4I 2020 dataset. The objective is to predict machine failures using multivariate sensor time-series data, enabling preventive maintenance decisions in manufacturing environments.

## Dataset

- AI4I 2020 Predictive Maintenance Dataset
- Sensors: temperature, torque, rotational speed, tool wear
- Highly imbalanced failure labels
- No explicit timestamps (treated as sequential data)

## Approach

- Data cleaning and feature scaling
- Sliding window sequence construction
- LSTM and Transformer-based time-series models
- Time-aware training without random shuffling

## Challenges

- Severe class imbalance
- Weak temporal signals
- No true time index
- Trade-off between interpretability and model complexity

## Evaluation

- Precision-recall metrics
- Recall on failure events
- Training stability across epochs

## Tech Stack

- Python, PyTorch
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## How to Run

```bash
pip install -r requirements.txt
cd src
python train.py
```
