# Reproducibility

## Environment

Tested with Python 3.13, NumPy 2.3.5, pandas 2.3.3, and PyTorch 2.14 (CUDA
available during training). A CPU-only PyTorch installation can run inference and
the fusion step, but full retraining is slow.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the public archive:

<https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip>

Unzip to `data/jena_climate_2009_2016.csv`. The loaders also download it
automatically on first use:

```bash
python code/common/data.py
```

## Reproduce the fusion result

The repository includes the final Transformer/LSTM validation and test predictions
and the final checkpoints. The fusion step therefore runs without retraining:

```bash
python code/fusion/fusion.py
```

This reads `outputs/v3/preds/` and `outputs/preds/`, and writes
`outputs/v3/fusion/fusion_report.json` plus the fused prediction arrays. The
reported test values are MAE 1.714 and RMSE 2.290 °C for equal-weight fusion.

## Re-export model predictions

```bash
python code/transformer/export_preds.py --model enc
python code/lstm/export_preds.py --tag lstm_time_res
```

## Retrain from scratch

```bash
# Transformer, final d128-L3 encoder
python code/transformer/train_enc.py --final

# Selected LSTM variant
python code/lstm/train.py --tag lstm_time_res --time-features --residual
```

The Transformer entry point downloads and prepares the data if needed. The LSTM
entry point uses `code/lstm_common/data.py` with the same data protocol.

## What is not in the repository

- the 42 MB raw CSV (public download; excluded by `.gitignore`);
- per-epoch experiment caches under `outputs/v3/experiments/`;
- intermediate ablation checkpoints and prediction dumps;
- presentation slides and course submission documents.

The included final checkpoints and selected prediction arrays are sufficient to
reproduce the fusion table without retraining.
