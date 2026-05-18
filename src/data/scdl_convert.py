"""Convert each disease/split h5ad to BioNeMo SCDL memmap format.

Loops over `data/<disease>/splits/{train,val,test}/<disease>.h5ad` (produced by
`src.data.preprocess`) and invokes the BioNeMo CLI `convert_h5ad_to_scdl` for
each. Output lands at `data/<disease>/scdl/{train,val,test}/`.

This module is a thin wrapper around the CLI — the heavy lifting (rank-based
gene ordering for Geneformer, sparse memmap layout) is done by BioNeMo itself.

Must run inside the BioNeMo container (the `convert_h5ad_to_scdl` console script
is shipped by `bionemo-scdl` and not pip-installable standalone in 2.7.1).

    python -u -m src.data.scdl_convert --config configs/diseases.yaml --out data/

CELLxGENE Census stores raw counts in `.X` (not `.raw.X`), so we always pass
`--use-X-not-raw`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


def convert_split(disease_key: str, split: str, data_root: Path, force: bool = False) -> dict:
    """Run convert_h5ad_to_scdl on one disease/split."""
    in_dir = data_root / disease_key / "splits" / split
    out_dir = data_root / disease_key / "scdl" / split

    h5ad_files = list(in_dir.glob("*.h5ad"))
    if not h5ad_files:
        return {"disease": disease_key, "split": split, "status": "failed",
                "error": f"no h5ad in {in_dir}"}

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[{disease_key}/{split}] already converted, skipping (--force to overwrite)")
        return {"disease": disease_key, "split": split, "status": "skipped",
                "out_dir": str(out_dir)}

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "convert_h5ad_to_scdl",
        "--data-path", str(in_dir),
        "--save-path", str(out_dir),
        "--use-X-not-raw",
    ]
    print(f"[{disease_key}/{split}] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"disease": disease_key, "split": split, "status": "failed",
                "returncode": completed.returncode,
                "stderr": completed.stderr[-2000:]}

    produced = sorted(p.name for p in out_dir.rglob("*") if p.is_file())
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"[{disease_key}/{split}] wrote {len(produced)} files, {size_mb:,.1f} MB → {out_dir}")
    return {
        "disease": disease_key, "split": split, "status": "done",
        "out_dir": str(out_dir),
        "n_files": len(produced),
        "size_mb": round(size_mb, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="overwrite existing SCDL output")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to a subset of disease keys (default: all)")
    parser.add_argument("--splits", nargs="*", default=list(SPLITS),
                        help=f"restrict to a subset of splits (default: {list(SPLITS)})")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    diseases = cfg["diseases"]
    keys = args.only if args.only else list(diseases.keys())

    summary = []
    for key in keys:
        for split in args.splits:
            summary.append(convert_split(key, split, args.out, force=args.force))

    summary_path = args.out / "scdl_convert_summary.json"
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
