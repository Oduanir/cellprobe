"""In silico gene perturbation, candidate selection driven by Geneformer's own token use.

Why this design: we initially picked candidate genes by mean expression in the
disease cells. That produced zero embedding shift because Geneformer's tokenizer
normalizes by a global per-gene median dictionary — a constitutively highly-expressed
gene like IGKC (mean 275 in our UC subset) is normalized to ~1 and not in the
top-2048 token sequence, so removing it from .X changes nothing downstream.

The corrected approach:
  1. Run one baseline inference on the disease-confident subset, asking
     `infer_geneformer` to return `input_ids` alongside embeddings.
  2. The candidate set becomes the genes whose Ensembl ID actually appears as
     a frequent token across these cells — those are guaranteed to be in
     Geneformer's input and hence guaranteed to have a non-zero effect when
     perturbed (caveat: shifting one of 2048 tokens may still be small, but
     it's at least non-trivially measurable, which is the prerequisite to
     ranking).
  3. For each candidate gene g:
       - Write a perturbed h5ad with g's column in `.X` set to 0
       - convert_h5ad_to_scdl → infer → embeddings
       - Compute cosine_distance(baseline_emb, perturbed_emb) per cell
       - Score = mean cosine shift across the cells where g actually appeared
         in the baseline tokens (others can't be affected)
       - Also report mean delta P(disease) through the trained MLP probe.

This is the Theodoris 2023 methodology, adapted to BioNeMo by going through
its own data converter and inference CLI rather than mucking with internal
binary memmaps.

    python -u -m src.perturb.perturb \\
        --disease uc \\
        --n-cells 100 --top-tokens 200 --max-genes 5   # smoke test
    python -u -m src.perturb.perturb \\
        --disease uc \\
        --n-cells 500 --top-tokens 200                # full panel

Approximate cost on g6.xlarge L4 24GB (~45 s per gene):
  200 candidates × 3 diseases ≈ 7.5 h compute.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import urllib.request

import anndata as ad
import joblib
import numpy as np
import torch
import yaml
from scipy.sparse import csr_matrix

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
OT_QUERY = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    name
    associatedTargets(page: {index: 0, size: $size}) {
      rows { score target { id approvedSymbol } }
    }
  }
}
"""


def fetch_opentargets_candidates(efo_id: str, size: int = 300) -> list[tuple[str, str, float]]:
    """Return [(ensembl, symbol, score)] from OpenTargets for a given disease EFO."""
    body = json.dumps({"query": OT_QUERY,
                       "variables": {"efoId": efo_id, "size": size}}).encode()
    req = urllib.request.Request(OT_GRAPHQL_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    rows = data["data"]["disease"]["associatedTargets"]["rows"]
    return [(r["target"]["id"], r["target"]["approvedSymbol"], r["score"]) for r in rows]

REPO_ROOT = Path(__file__).resolve().parents[2]
INFER_WRAPPER = REPO_ROOT / "scripts" / "infer_wrapper.py"

SPECIAL_TOKENS = {0, 1, 2, 3, 4}   # [CLS] [MASK] [PAD] [SEP] [UKW]


def find_checkpoint() -> Path:
    cache = Path(os.environ.get("BIONEMO_CACHE", "/cache"))
    candidates = sorted(cache.glob("*geneformer_10M*.untar"))
    if not candidates:
        raise FileNotFoundError(f"no geneformer_10M checkpoint under {cache}")
    return candidates[0]


def load_geneformer_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Return (ensembl_to_token, token_to_ensembl) from the BioNeMo cache.

    The vocab is a JSON file `{"vocab": {"[CLS]": 0, ..., "ENSG...": 5, ...}, ...}`.
    """
    cache = Path(os.environ.get("BIONEMO_CACHE", "/cache"))
    for path in cache.rglob("geneformer.vocab"):
        with open(path) as f:
            payload = json.load(f)
        vocab = payload.get("vocab") if isinstance(payload, dict) else payload
        if not isinstance(vocab, dict):
            continue
        ens_to_tok = {k: v for k, v in vocab.items() if k.startswith("ENSG")}
        tok_to_ens = {v: k for k, v in ens_to_tok.items()}
        return ens_to_tok, tok_to_ens
    raise FileNotFoundError("geneformer.vocab not found in BIONEMO_CACHE")


def run_h5ad_through_pipeline(
    adata: ad.AnnData,
    scratch_dir: Path,
    name: str,
    checkpoint: Path,
    micro_batch_size: int = 16,
    include_input_ids: bool = False,
) -> dict:
    """Write adata → SCDL → infer; return the loaded predictions dict."""
    sd = scratch_dir / name
    h5ad_in = sd / "h5ad"
    scdl_out = sd / "scdl"
    infer_out = sd / "infer"
    if sd.exists():
        shutil.rmtree(sd)
    h5ad_in.mkdir(parents=True)
    adata.write_h5ad(h5ad_in / "x.h5ad")

    completed = subprocess.run(
        ["convert_h5ad_to_scdl", "--data-path", str(h5ad_in), "--save-path", str(scdl_out)],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"convert_h5ad_to_scdl failed: {completed.stderr[-1500:]}")

    infer_out.mkdir(parents=True)
    cmd = ["python", str(INFER_WRAPPER),
           "--data-dir", str(scdl_out),
           "--checkpoint-path", str(checkpoint),
           "--results-path", str(infer_out),
           "--precision", "bf16-mixed",
           "--micro-batch-size", str(micro_batch_size),
           "--seq-length", "2048",
           "--num-gpus", "1"]
    if include_input_ids:
        cmd.append("--include-input-ids")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"infer_geneformer failed: {completed.stderr[-1500:]}")
    pt = torch.load(infer_out / "predictions__rank_0.pt", map_location="cpu", weights_only=False)
    return pt


def select_token_candidates(
    input_ids: np.ndarray,
    top_tokens: int,
) -> list[tuple[int, int]]:
    """Return [(token_id, n_cells_containing)] for the top-K most common tokens
    that appear across cells, excluding special tokens.
    """
    counter: Counter[int] = Counter()
    n_cells = input_ids.shape[0]
    for c in range(n_cells):
        for t in np.unique(input_ids[c]):
            t = int(t)
            if t in SPECIAL_TOKENS:
                continue
            counter[t] += 1
    return counter.most_common(top_tokens)


def perturb_disease(
    disease_key: str,
    data_root: Path,
    results_root: Path,
    n_cells: int,
    top_tokens: int,
    micro_batch_size: int,
    max_genes: int | None,
    opentargets_efo: str | None = None,
) -> dict:
    test_h5ad = data_root / disease_key / "splits" / "test" / f"{disease_key}.h5ad"
    test_pt = results_root / disease_key / "embeddings" / "test" / "predictions__rank_0.pt"
    probe_path = results_root / disease_key / "probe.joblib"

    print(f"[{disease_key}] loading data + probe")
    adata = ad.read_h5ad(test_h5ad)
    emb_all = torch.load(test_pt, map_location="cpu", weights_only=False)["embeddings"].float().numpy()
    probe = joblib.load(probe_path)
    proba_all = probe.predict_proba(emb_all)[:, 1]

    is_disease = adata.obs["label"].to_numpy() == 1
    disease_idx = np.where(is_disease)[0]
    if len(disease_idx) < n_cells:
        chosen = disease_idx
    else:
        chosen = disease_idx[np.argsort(-proba_all[disease_idx])][:n_cells]
    sub = adata[chosen].copy()
    proba_baseline = proba_all[chosen]
    print(f"[{disease_key}] picked {len(sub)} disease-confident cells (mean P={proba_baseline.mean():.3f})")

    ens_to_tok, tok_to_ens = load_geneformer_vocab()
    print(f"[{disease_key}] vocab: {len(ens_to_tok)} Ensembl tokens")

    # Map var.feature_id (Ensembl) → column index in our adata
    ensembl_to_var_idx: dict[str, int] = {}
    if "feature_id" in sub.var.columns:
        for i, ens in enumerate(sub.var["feature_id"].astype(str)):
            ensembl_to_var_idx[ens] = i
    else:
        raise SystemExit(f"[{disease_key}] sub.var lacks 'feature_id' column — cannot map genes")

    checkpoint = find_checkpoint()
    scratch = Path(tempfile.mkdtemp(prefix=f"perturb_{disease_key}_"))
    print(f"[{disease_key}] scratch: {scratch}")

    # 1. Baseline inference, including input_ids
    print(f"[{disease_key}] baseline infer (with input_ids)")
    p0 = run_h5ad_through_pipeline(
        sub, scratch, "baseline", checkpoint, micro_batch_size, include_input_ids=True)
    input_ids = p0["input_ids"].numpy()
    emb_baseline = p0["embeddings"].float().numpy()
    print(f"[{disease_key}] baseline embeddings shape={emb_baseline.shape}")

    # 2. Build candidate list
    #    - If opentargets_efo is provided: query OT, filter to genes in var, find their token IDs
    #    - Otherwise (default): top-N most-frequent tokens used by Geneformer in baseline
    candidates: list[dict] = []
    if opentargets_efo:
        ot = fetch_opentargets_candidates(opentargets_efo, size=top_tokens * 2)
        print(f"[{disease_key}] OpenTargets {opentargets_efo}: {len(ot)} candidates")
        for ens, sym, ot_score in ot:
            if ens not in ensembl_to_var_idx or ens not in ens_to_tok:
                continue
            token_id = ens_to_tok[ens]
            cells_with_token = (input_ids == token_id).any(axis=1)
            if cells_with_token.sum() == 0:
                continue
            candidates.append({
                "token_id": int(token_id),
                "ensembl_id": ens,
                "symbol": sym,
                "opentargets_score": float(ot_score),
                "gene_idx_in_adata": int(ensembl_to_var_idx[ens]),
                "n_cells_with_token": int(cells_with_token.sum()),
                "cells_with_token_mask": cells_with_token,
            })
        print(f"[{disease_key}] {len(candidates)} OT candidates kept (in var + appearing in tokens)")
    else:
        top_token_freq = select_token_candidates(input_ids, top_tokens)
        for token_id, n_in in top_token_freq:
            ens = tok_to_ens.get(token_id)
            if ens is None or ens not in ensembl_to_var_idx:
                continue
            cells_with_token = (input_ids == token_id).any(axis=1)
            candidates.append({
                "token_id": int(token_id),
                "ensembl_id": ens,
                "gene_idx_in_adata": int(ensembl_to_var_idx[ens]),
                "n_cells_with_token": int(cells_with_token.sum()),
                "cells_with_token_mask": cells_with_token,
            })
        print(f"[{disease_key}] {len(candidates)} candidate tokens mapped to genes")
    if max_genes is not None:
        candidates = candidates[:max_genes]

    # 3. Perturb each candidate
    out_dir = results_root / disease_key / "perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()
    for i, c in enumerate(candidates):
        gi = c["gene_idx_in_adata"]
        sub_p = sub.copy()
        X = sub_p.X.tolil() if hasattr(sub_p.X, "tolil") else sub_p.X
        X[:, gi] = 0
        sub_p.X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)

        try:
            pt = run_h5ad_through_pipeline(
                sub_p, scratch, "pert", checkpoint, micro_batch_size, include_input_ids=False)
            emb_pert = pt["embeddings"].float().numpy()
        except Exception as exc:
            rows.append({"rank_in_panel": i, "ensembl_id": c["ensembl_id"],
                         "token_id": c["token_id"], "status": "failed",
                         "error": str(exc)[:300]})
            print(f"[{disease_key}] {c['ensembl_id']} FAILED: {exc}")
            continue

        # cosine distance only meaningful for cells that actually had this token in their baseline
        mask = c["cells_with_token_mask"]
        if mask.sum() == 0:
            row = {"rank_in_panel": i, "ensembl_id": c["ensembl_id"],
                   "token_id": c["token_id"], "status": "no_cells",
                   "n_cells_affected": 0}
        else:
            a = emb_baseline[mask]
            b = emb_pert[mask]
            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            cos_sim = (a_norm * b_norm).sum(axis=1)
            shifts = 1.0 - cos_sim
            proba_pert = probe.predict_proba(emb_pert)[:, 1]
            delta_p = (proba_pert[mask] - proba_baseline[mask]).mean()
            row = {
                "rank_in_panel": i,
                "ensembl_id": c["ensembl_id"],
                "symbol": c.get("symbol", ""),
                "opentargets_score": c.get("opentargets_score"),
                "token_id": c["token_id"],
                "n_cells_affected": int(mask.sum()),
                "mean_cosine_shift": float(shifts.mean()),
                "median_cosine_shift": float(np.median(shifts)),
                "max_cosine_shift": float(shifts.max()),
                "delta_p_disease": float(delta_p),
                "status": "done",
            }
        rows.append(row)

        if (i + 1) % 5 == 0 or i == len(candidates) - 1:
            total = time.time() - t_start
            eta = total / (i + 1) * (len(candidates) - i - 1)
            print(f"[{disease_key}] {i+1}/{len(candidates)}  shift={row.get('mean_cosine_shift', 'NA')}  "
                  f"ΔP={row.get('delta_p_disease', 'NA')}  elapsed={total/60:.1f}min  eta={eta/60:.1f}min")

    shutil.rmtree(scratch, ignore_errors=True)

    # Rank by mean cosine shift descending
    done = [r for r in rows if r["status"] == "done"]
    done.sort(key=lambda r: -r["mean_cosine_shift"])
    for rk, r in enumerate(done):
        r["rank_by_shift"] = rk

    out_json = out_dir / "perturbation.json"
    payload = {
        "disease": disease_key,
        "n_cells": int(len(sub)),
        "n_candidates": len(candidates),
        "n_done": len(done),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[{disease_key}] wrote {out_json}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--disease", required=True)
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--top-tokens", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=None,
                        help="cap on the number of genes actually perturbed (for smoke testing)")
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--opentargets-efo", default=None,
                        help="If set (e.g. EFO_0000729), use OpenTargets disease-target associations "
                             "as the candidate gene list instead of top-frequency tokens.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.disease not in cfg["diseases"]:
        raise SystemExit(f"disease {args.disease} not in config")

    perturb_disease(
        disease_key=args.disease,
        data_root=args.data_root,
        results_root=args.results_root,
        n_cells=args.n_cells,
        top_tokens=args.top_tokens,
        micro_batch_size=args.micro_batch_size,
        max_genes=args.max_genes,
        opentargets_efo=args.opentargets_efo,
    )


if __name__ == "__main__":
    main()
