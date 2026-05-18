"""Run Geneformer 10M inference on each disease/split to extract cell embeddings.

For each (disease, split) under `data/<disease>/scdl/<split>/`, invokes
`infer_geneformer` (via our `scripts/infer_wrapper.py` to apply the
BERTMLMLossWithReductionNoForward → BERTMLMLossWithReduction alias for
BioNeMo 2.7.1) and writes the per-cell embeddings as
`results/<disease>/embeddings/<split>/predictions__rank_0.pt`.

The pretrained Geneformer 10M from NGC produces (n_cells, 256) bf16
cell-level embeddings — these feed the MLP probe in `src.eval.probe`.

Must run inside the BioNeMo container with the GPU. The infer step is the
expensive one (a few minutes per split on an L4); the probe downstream is
sklearn-only.

    python -u -m src.eval.extract_embeddings --config configs/diseases.yaml \\
        --data-root data/ --results-root results/ \\
        --checkpoint /root/.cache/bionemo/<...>/geneformer_10M_*.untar
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")
DEFAULT_INFER_WRAPPER = Path(__file__).parent.parent.parent / "scripts" / "infer_wrapper.py"


def find_checkpoint(default: Path | None = None) -> Path:
    """Locate the Geneformer 10M checkpoint in the bionemo cache."""
    if default and default.exists():
        return default
    cache = Path(os.environ.get("BIONEMO_CACHE", "/root/.cache/bionemo"))
    candidates = sorted(cache.glob("*geneformer_10M*.untar"))
    if not candidates:
        raise FileNotFoundError(
            f"could not find geneformer_10M checkpoint under {cache}; "
            "run `download_bionemo_data geneformer/10M_241113:2.0 --source ngc` first"
        )
    return candidates[0]


def extract_split(
    disease_key: str,
    split: str,
    data_root: Path,
    results_root: Path,
    checkpoint: Path,
    infer_wrapper: Path,
    micro_batch_size: int,
    seq_length: int,
    force: bool,
) -> dict:
    """Run inference on one disease/split, return summary dict."""
    in_dir = data_root / disease_key / "scdl" / split
    out_dir = results_root / disease_key / "embeddings" / split

    if not in_dir.exists():
        return {"disease": disease_key, "split": split, "status": "failed",
                "error": f"missing SCDL dir: {in_dir}"}

    out_file = out_dir / "predictions__rank_0.pt"
    if out_file.exists() and not force:
        size_mb = out_file.stat().st_size / 1e6
        print(f"[{disease_key}/{split}] already extracted ({size_mb:.1f} MB), skipping (--force to overwrite)")
        return {"disease": disease_key, "split": split, "status": "skipped",
                "predictions": str(out_file), "size_mb": round(size_mb, 1)}

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(infer_wrapper),
        "--data-dir", str(in_dir),
        "--checkpoint-path", str(checkpoint),
        "--results-path", str(out_dir),
        "--precision", "bf16-mixed",
        "--micro-batch-size", str(micro_batch_size),
        "--seq-length", str(seq_length),
        "--num-gpus", "1",
    ]
    print(f"[{disease_key}/{split}] running infer (bs={micro_batch_size}, seq={seq_length})")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not out_file.exists():
        return {"disease": disease_key, "split": split, "status": "failed",
                "returncode": completed.returncode,
                "stderr": completed.stderr[-2000:]}

    size_mb = out_file.stat().st_size / 1e6
    print(f"[{disease_key}/{split}] wrote {out_file.name} ({size_mb:,.1f} MB)")
    return {"disease": disease_key, "split": split, "status": "done",
            "predictions": str(out_file), "size_mb": round(size_mb, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Geneformer checkpoint dir (default: auto-find in BIONEMO_CACHE)")
    parser.add_argument("--infer-wrapper", type=Path, default=DEFAULT_INFER_WRAPPER)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to a subset of disease keys")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS),
                        help=f"restrict to splits (default: {list(SPLITS)})")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    diseases = cfg["diseases"]
    keys = args.only if args.only else list(diseases.keys())

    checkpoint = find_checkpoint(args.checkpoint)
    print(f"checkpoint: {checkpoint}")

    summary = []
    for key in keys:
        for split in args.splits:
            summary.append(extract_split(
                key, split, args.data_root, args.results_root,
                checkpoint, args.infer_wrapper,
                args.micro_batch_size, args.seq_length, args.force,
            ))

    summary_path = args.results_root / "extract_embeddings_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {summary_path}")

    n_done = sum(1 for s in summary if s["status"] == "done")
    n_skip = sum(1 for s in summary if s["status"] == "skipped")
    n_fail = sum(1 for s in summary if s["status"] == "failed")
    print(f"done={n_done}, skipped={n_skip}, failed={n_fail}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
