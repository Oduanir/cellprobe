"""QC + stratified train/val/test split for one disease config.

Reads `data/<disease>/raw/<dataset_id>.h5ad` (produced by `src.data.download`),
applies standard scRNA-seq QC, splits cells into train/val/test, and writes
one h5ad per split under `data/<disease>/splits/{train,val,test}/<disease>.h5ad`.

SCDL conversion (h5ad → BioNeMo memmap format used by Geneformer) is a separate
step that must run inside the BioNeMo container — see `scripts/scdl_convert.sh`.
The split here is what BioNeMo's `convert_h5ad_to_scdl` consumes one directory
at a time.

Run inside the BioNeMo container (scanpy + anndata + sklearn are pre-installed):

    python -u -m src.data.preprocess --config configs/diseases.yaml --out data/

Per-disease QC params can be overridden in `configs/diseases.yaml` under the
`qc:` key — defaults below are conservative scRNA-seq standards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import yaml
from sklearn.model_selection import train_test_split

DEFAULT_QC = {
    "min_genes_per_cell": 200,
    "min_cells_per_gene": 10,
    "max_pct_mt": 20.0,
}
DEFAULT_SPLIT = {"val_frac": 0.15, "test_frac": 0.15, "seed": 42}


def _flag_mito_genes(adata: ad.AnnData) -> str | None:
    """Add `var['mt']` boolean column if mitochondrial genes can be identified.

    CELLxGENE Census uses Ensembl IDs in `var.index` and typically exposes
    HGNC symbols in `var['feature_name']`. Mitochondrial genes start with
    `MT-` (or `mt-` for mouse). Returns the column name used, or None if
    we could not identify mito genes (in which case the mito filter is skipped).
    """
    if "feature_name" in adata.var.columns:
        symbols = adata.var["feature_name"].astype(str)
        adata.var["mt"] = symbols.str.upper().str.startswith("MT-")
        return "feature_name"
    if adata.var.index.astype(str).str.startswith(("MT-", "mt-")).any():
        adata.var["mt"] = adata.var.index.astype(str).str.upper().str.startswith("MT-")
        return "index"
    adata.var["mt"] = False
    return None


def qc_filter(adata: ad.AnnData, qc: dict) -> tuple[ad.AnnData, dict]:
    """Apply standard scRNA-seq QC filters, return filtered AnnData and stats.

    - Drop cells with fewer than `min_genes_per_cell` expressed genes.
    - Drop cells with mitochondrial fraction above `max_pct_mt` (if mito genes identifiable).
    - Drop genes expressed in fewer than `min_cells_per_gene` cells.
    """
    n0_obs, n0_var = adata.n_obs, adata.n_vars
    mito_source = _flag_mito_genes(adata)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    cell_keep = adata.obs["n_genes_by_counts"] >= qc["min_genes_per_cell"]
    if mito_source is not None and adata.var["mt"].any():
        cell_keep &= adata.obs["pct_counts_mt"] < qc["max_pct_mt"]
    adata = adata[cell_keep].copy()

    gene_keep = (adata.X > 0).sum(axis=0)
    gene_keep = np.asarray(gene_keep).ravel() >= qc["min_cells_per_gene"]
    adata = adata[:, gene_keep].copy()

    stats = {
        "n_cells_before": int(n0_obs),
        "n_cells_after": int(adata.n_obs),
        "n_genes_before": int(n0_var),
        "n_genes_after": int(adata.n_vars),
        "mito_source": mito_source,
        "qc_params": dict(qc),
    }
    return adata, stats


def stratified_split(
    adata: ad.AnnData,
    label_col: str,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, ad.AnnData]:
    """Stratified train/val/test split of cells by `label_col`."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    idx = np.arange(adata.n_obs)
    labels = adata.obs[label_col].to_numpy()
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed
    )
    val_size_relative = val_frac / (1 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_relative,
        stratify=labels[idx_trainval],
        random_state=seed,
    )
    return {
        "train": adata[idx_train].copy(),
        "val": adata[idx_val].copy(),
        "test": adata[idx_test].copy(),
    }


def preprocess_disease(
    disease_key: str,
    disease_cfg: dict,
    data_root: Path,
    qc_overrides: dict | None = None,
    split_overrides: dict | None = None,
    force: bool = False,
) -> dict:
    """Run the full preprocess on one disease, return a summary dict."""
    dataset_id = disease_cfg["dataset_id"]
    disease_label = disease_cfg["disease_label"]
    healthy_label = disease_cfg["healthy_label"]

    raw_path = data_root / disease_key / "raw" / f"{dataset_id}.h5ad"
    splits_dir = data_root / disease_key / "splits"

    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw h5ad: {raw_path}")

    expected = [splits_dir / s / f"{disease_key}.h5ad" for s in ("train", "val", "test")]
    if all(p.exists() for p in expected) and not force:
        print(f"[{disease_key}] all splits already present, skipping (--force to overwrite)")
        return {"disease": disease_key, "status": "skipped"}

    print(f"[{disease_key}] reading {raw_path}")
    adata = ad.read_h5ad(raw_path)

    adata.obs["label"] = (adata.obs["disease"].astype(str) == disease_label).astype(np.int8)
    n_disease = int((adata.obs["label"] == 1).sum())
    n_healthy = int((adata.obs["label"] == 0).sum())
    if n_disease == 0 or n_healthy == 0:
        raise ValueError(
            f"[{disease_key}] one class is empty: {n_disease} disease / {n_healthy} healthy"
        )
    print(f"[{disease_key}] loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes "
          f"({n_disease:,} {disease_label} / {n_healthy:,} {healthy_label})")

    qc = {**DEFAULT_QC, **(qc_overrides or {})}
    adata, qc_stats = qc_filter(adata, qc)
    print(f"[{disease_key}] post-QC: {adata.n_obs:,} cells, {adata.n_vars:,} genes "
          f"(mito_source={qc_stats['mito_source']})")

    split_params = {**DEFAULT_SPLIT, **(split_overrides or {})}
    split_params["seed"] = disease_cfg.get("seed", split_params["seed"])
    splits = stratified_split(adata, label_col="label", **split_params)

    summary = {
        "disease": disease_key,
        "raw_path": str(raw_path),
        "qc": qc_stats,
        "split_params": split_params,
        "splits": {},
        "status": "done",
    }
    for split_name, split_adata in splits.items():
        out_dir = splits_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{disease_key}.h5ad"
        split_adata.write_h5ad(out_path, compression="gzip")
        size_mb = out_path.stat().st_size / 1e6
        n_d = int((split_adata.obs["label"] == 1).sum())
        n_h = int((split_adata.obs["label"] == 0).sum())
        print(f"[{disease_key}] {split_name}: {split_adata.n_obs:,} cells "
              f"({n_d:,} disease / {n_h:,} healthy), {size_mb:,.1f} MB → {out_path}")
        summary["splits"][split_name] = {
            "path": str(out_path),
            "n_cells": split_adata.n_obs,
            "n_disease": n_d,
            "n_healthy": n_h,
            "size_mb": round(size_mb, 1),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="overwrite existing splits")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to a subset of disease keys (default: all)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    qc_overrides = cfg.get("qc")
    split_overrides = cfg.get("split")
    diseases = cfg["diseases"]
    keys = args.only if args.only else list(diseases.keys())

    summary = []
    for key in keys:
        try:
            summary.append(preprocess_disease(
                key, diseases[key], args.out,
                qc_overrides=qc_overrides,
                split_overrides=split_overrides,
                force=args.force,
            ))
        except Exception as exc:
            print(f"[{key}] FAILED: {exc}")
            summary.append({"disease": key, "status": "failed", "error": str(exc)})

    summary_path = args.out / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {summary_path}")


if __name__ == "__main__":
    main()
