"""Download disease scRNA-seq datasets from CELLxGENE Census as AnnData h5ad files.

Reads `configs/diseases.yaml`, downloads one dataset per disease, optionally subsampling
the healthy/disease class to a target size, and writes
`data/<disease>/raw/<dataset_id>.h5ad`.

Run inside the BioNeMo container (cellxgene_census + anndata are pre-installed):

    python -u -m src.data.download --config configs/diseases.yaml --out data/

Config knobs per disease:
    dataset_id, disease_label, healthy_label : required
    max_disease_cells, max_healthy_cells     : optional caps on class sizes
    seed                                     : RNG seed for subsampling (default 42)

If the h5ad file already exists, the download is skipped — re-run with `--force` to overwrite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cellxgene_census as cc
import numpy as np
import yaml


def download_disease(
    census,
    disease_key: str,
    disease_cfg: dict,
    out_dir: Path,
    force: bool = False,
) -> dict:
    """Fetch one disease dataset from the census and save as h5ad.

    If `max_disease_cells` or `max_healthy_cells` are set in the config, the
    respective class is uniformly subsampled before materializing the AnnData.
    """
    dataset_id = disease_cfg["dataset_id"]
    disease_label = disease_cfg["disease_label"]
    healthy_label = disease_cfg["healthy_label"]
    max_disease = disease_cfg.get("max_disease_cells")
    max_healthy = disease_cfg.get("max_healthy_cells")
    seed = disease_cfg.get("seed", 42)

    target_dir = out_dir / disease_key / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = target_dir / f"{dataset_id}.h5ad"

    if h5ad_path.exists() and not force:
        print(f"[{disease_key}] already present at {h5ad_path}, skipping (use --force to overwrite)")
        return {"disease": disease_key, "h5ad": str(h5ad_path), "status": "skipped"}

    print(f"[{disease_key}] querying class sizes ({dataset_id})")
    disease_obs = cc.get_obs(
        census,
        organism="Homo sapiens",
        value_filter=f'dataset_id == "{dataset_id}" and disease == "{disease_label}"',
        column_names=["soma_joinid"],
    )
    healthy_obs = cc.get_obs(
        census,
        organism="Homo sapiens",
        value_filter=f'dataset_id == "{dataset_id}" and disease == "{healthy_label}"',
        column_names=["soma_joinid"],
    )
    n_disease_full = len(disease_obs)
    n_healthy_full = len(healthy_obs)
    print(
        f"[{disease_key}] full counts: {n_disease_full:,} {disease_label} / "
        f"{n_healthy_full:,} {healthy_label}"
    )

    rng = np.random.default_rng(seed)
    disease_ids = disease_obs["soma_joinid"].to_numpy()
    healthy_ids = healthy_obs["soma_joinid"].to_numpy()

    if max_disease and n_disease_full > max_disease:
        print(f"[{disease_key}] subsampling {disease_label} {n_disease_full:,} -> {max_disease:,}")
        disease_ids = rng.choice(disease_ids, size=max_disease, replace=False)
    if max_healthy and n_healthy_full > max_healthy:
        print(f"[{disease_key}] subsampling {healthy_label} {n_healthy_full:,} -> {max_healthy:,}")
        healthy_ids = rng.choice(healthy_ids, size=max_healthy, replace=False)

    target_ids = np.concatenate([disease_ids, healthy_ids])
    print(f"[{disease_key}] fetching {len(target_ids):,} cells")
    adata = cc.get_anndata(
        census,
        organism="Homo sapiens",
        obs_coords=target_ids,
    )

    n_disease = int((adata.obs["disease"] == disease_label).sum())
    n_healthy = int((adata.obs["disease"] == healthy_label).sum())
    print(
        f"[{disease_key}] materialized: {adata.n_obs:,} cells "
        f"({n_disease:,} {disease_label} / {n_healthy:,} {healthy_label}), "
        f"genes: {adata.n_vars:,}"
    )

    print(f"[{disease_key}] writing {h5ad_path}")
    adata.write_h5ad(h5ad_path, compression="gzip")
    size_mb = h5ad_path.stat().st_size / 1e6
    print(f"[{disease_key}] wrote {size_mb:,.1f} MB")

    return {
        "disease": disease_key,
        "h5ad": str(h5ad_path),
        "n_cells": adata.n_obs,
        "n_disease": n_disease,
        "n_healthy": n_healthy,
        "n_disease_full": n_disease_full,
        "n_healthy_full": n_healthy_full,
        "n_genes": adata.n_vars,
        "size_mb": round(size_mb, 1),
        "status": "downloaded",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="overwrite existing h5ad")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="restrict to a subset of disease keys (default: all)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    census = cc.open_soma(census_version=cfg.get("census_version", "latest"))

    diseases = cfg["diseases"]
    keys = args.only if args.only else list(diseases.keys())

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for key in keys:
        summary.append(download_disease(census, key, diseases[key], args.out, force=args.force))

    summary_path = args.out / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {summary_path}")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
