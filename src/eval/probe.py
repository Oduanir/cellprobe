"""Train + evaluate an MLP probe on frozen Geneformer 10M embeddings.

Mirrors NVIDIA's official benchmark in
`bionemo.geneformer.scripts.celltype_classification_bench`: a 3-layer MLP
classifier trained on frozen cell embeddings. For our task the head is a
binary classifier (disease vs healthy).

For each disease:
  1. Load train/val/test embeddings from `results/<disease>/embeddings/<split>/predictions__rank_0.pt`
  2. Load the matching `obs['label']` from `data/<disease>/splits/<split>/<disease>.h5ad`
  3. Standardize → MLPClassifier(hidden_layer_sizes=(128,), early stopping)
  4. Report accuracy, macro-F1, ROC-AUC, precision, recall, confusion matrix
     on val and test
  5. Save the fitted pipeline + metrics for downstream perturbation analysis

    python -u -m src.eval.probe --config configs/diseases.yaml \\
        --data-root data/ --results-root results/

The fitted MLP is small enough to ship with the repo (~100 KB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from anndata import read_h5ad
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SPLITS = ("train", "val", "test")
SEED = 1337  # matches bionemo.geneformer.scripts.celltype_classification_bench


def load_split(
    disease_key: str,
    split: str,
    data_root: Path,
    results_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for one (disease, split): embeddings and integer labels."""
    pt_path = results_root / disease_key / "embeddings" / split / "predictions__rank_0.pt"
    h5ad_path = data_root / disease_key / "splits" / split / f"{disease_key}.h5ad"

    if not pt_path.exists():
        raise FileNotFoundError(f"missing embeddings: {pt_path}")
    if not h5ad_path.exists():
        raise FileNotFoundError(f"missing h5ad: {h5ad_path}")

    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    X = d["embeddings"].float().cpu().numpy()

    adata = read_h5ad(h5ad_path)
    if "label" not in adata.obs.columns:
        raise KeyError(f"obs['label'] missing in {h5ad_path}")
    y = adata.obs["label"].to_numpy().astype(np.int64)

    if len(X) != len(y):
        raise ValueError(
            f"length mismatch for {disease_key}/{split}: "
            f"{len(X)} embeddings vs {len(y)} labels"
        )
    return X, y


def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute accuracy / F1 / AUC / precision / recall / confusion matrix."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    metrics = {
        "n": int(len(y)),
        "n_disease": int(y.sum()),
        "n_healthy": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    print(f"  [{label}] n={metrics['n']:,}  acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1_macro']:.3f}  auc={metrics['roc_auc']:.3f}")
    return metrics


def probe_disease(
    disease_key: str,
    disease_cfg: dict,
    data_root: Path,
    results_root: Path,
) -> dict:
    """Train the probe on `train`, evaluate on `val` and `test`."""
    print(f"[{disease_key}] loading splits")
    X_train, y_train = load_split(disease_key, "train", data_root, results_root)
    X_val, y_val = load_split(disease_key, "val", data_root, results_root)
    X_test, y_test = load_split(disease_key, "test", data_root, results_root)
    print(f"[{disease_key}] sizes: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}  "
          f"feat={X_train.shape[1]}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            verbose=False,
        )),
    ])

    print(f"[{disease_key}] training MLP probe")
    pipeline.fit(X_train, y_train)

    print(f"[{disease_key}] evaluating")
    metrics = {
        "train": evaluate(pipeline, X_train, y_train, "train"),
        "val":   evaluate(pipeline, X_val,   y_val,   "val"),
        "test":  evaluate(pipeline, X_test,  y_test,  "test"),
    }

    out_dir = results_root / disease_key
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "probe.joblib"
    joblib.dump(pipeline, model_path)
    metrics_path = out_dir / "probe_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[{disease_key}] saved probe → {model_path}  metrics → {metrics_path}")

    return {"disease": disease_key, "status": "done",
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "n_features": int(X_train.shape[1]),
            "test_metrics": metrics["test"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    diseases = cfg["diseases"]
    keys = args.only if args.only else list(diseases.keys())

    summary = []
    for key in keys:
        try:
            summary.append(probe_disease(key, diseases[key], args.data_root, args.results_root))
        except Exception as exc:
            print(f"[{key}] FAILED: {exc}")
            summary.append({"disease": key, "status": "failed", "error": str(exc)})

    summary_path = args.results_root / "probe_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {summary_path}")

    n_fail = sum(1 for s in summary if s["status"] == "failed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
