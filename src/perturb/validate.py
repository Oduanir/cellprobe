"""Validate perturbation-derived gene rankings against OpenTargets disease-target associations.

For each disease, query the OpenTargets Platform GraphQL API for the top-N known
target associations, then compute:
  - precision@K: fraction of our top-K perturbed genes that are in OpenTargets top-N
  - hit recall: fraction of OpenTargets top-N that we recovered in our candidate panel
  - enrichment vs. random expectation
  - rank-correlation between perturbation effect and OpenTargets score (where overlap exists)

Outputs:
  results/<disease>/validation.json — metrics + matched/unmatched gene lists.

OpenTargets API: https://api.platform.opentargets.org/api/v4/graphql
EFO IDs are the disease ontology terms; we read them from the disease's
`disease_ontology_term_id` in `data/<disease>/splits/test/<disease>.h5ad`
(CELLxGENE stored them at download time).

    python -u -m src.perturb.validate --config configs/diseases.yaml \\
        --results-root results/ --top-k 100 --opentargets-top 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
USER_AGENT = "cellprobe/0.1 (https://github.com/.../cellprobe)"

QUERY = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    name
    id
    associatedTargets(page: {index: 0, size: $size}) {
      count
      rows {
        score
        target {
          id
          approvedSymbol
          approvedName
        }
      }
    }
  }
}
"""

DISEASE_EFO = {
    "uc":   "EFO_0000729",          # ulcerative colitis  → IL12B, JAK2, TNF
    "dcm":  "EFO_0000407",          # dilated cardiomyopathy → LMNA, TTN, DES
    "luad": "EFO_0000571",          # lung adenocarcinoma → EGFR, KRAS, TP53
}
DISEASE_EFO_FALLBACK: dict[str, str] = {}


def query_opentargets(efo_id: str, size: int = 500, retries: int = 3) -> dict:
    """Run the GraphQL query, retry on transient failures."""
    body = json.dumps({"query": QUERY,
                       "variables": {"efoId": efo_id, "size": size}}).encode()
    req = urllib.request.Request(
        OT_GRAPHQL_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": USER_AGENT},
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenTargets query failed after {retries} retries: {last_err}")


def fetch_targets(disease_key: str, size: int) -> list[dict]:
    """Return [{ensembl_id, symbol, score, name}] for the disease, OpenTargets-ranked."""
    candidates = [DISEASE_EFO.get(disease_key)]
    if disease_key in DISEASE_EFO_FALLBACK:
        candidates.append(DISEASE_EFO_FALLBACK[disease_key])
    for efo in candidates:
        if efo is None:
            continue
        payload = query_opentargets(efo, size=size)
        if payload.get("errors"):
            print(f"[{disease_key}] OpenTargets errors with {efo}: {payload['errors']}", file=sys.stderr)
            continue
        disease = (payload.get("data") or {}).get("disease")
        if not disease:
            continue
        assoc = (disease.get("associatedTargets") or {}).get("rows") or []
        if not assoc:
            continue
        targets = [{
            "ensembl_id": r["target"]["id"],
            "symbol":     r["target"]["approvedSymbol"],
            "name":       r["target"]["approvedName"],
            "score":      r["score"],
        } for r in assoc]
        print(f"[{disease_key}] OpenTargets {disease['name']} ({efo}): {len(targets)} targets")
        return targets
    raise RuntimeError(f"[{disease_key}] no OpenTargets data found for {candidates}")


def spearman_corr(x: list[float], y: list[float]) -> tuple[float, int]:
    """Spearman rank correlation, hand-rolled to avoid a SciPy dep."""
    n = len(x)
    if n < 3:
        return float("nan"), n
    def ranks(vals):
        idx = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank_minus_one, i in enumerate(idx):
            r[i] = rank_minus_one + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan"), n
    return num / (den_x * den_y), n


def validate_disease(
    disease_key: str,
    results_root: Path,
    top_k: int,
    opentargets_top: int,
) -> dict:
    perturb_json = results_root / disease_key / "perturbation" / "perturbation.json"
    if not perturb_json.exists():
        return {"disease": disease_key, "status": "no_perturbation_output"}

    payload = json.loads(perturb_json.read_text())
    rows = [r for r in payload["rows"] if r.get("status") == "done"]
    rows.sort(key=lambda r: -r["mean_cosine_shift"])

    targets = fetch_targets(disease_key, size=opentargets_top)
    ot_score_for: dict[str, float] = {t["ensembl_id"]: t["score"] for t in targets}
    ot_symbol_for: dict[str, str] = {t["ensembl_id"]: t["symbol"] for t in targets}
    ot_name_for:   dict[str, str] = {t["ensembl_id"]: t["name"]   for t in targets}

    panel_ensembls = {r["ensembl_id"] for r in rows}
    n_panel = len(rows)
    ot_in_panel = panel_ensembls & set(ot_score_for.keys())
    panel_is_ot_derived = (
        len(ot_in_panel) >= 0.8 * n_panel and n_panel > 0
    )  # candidates were explicitly OT-derived if essentially all rows are OT genes

    out = {
        "disease": disease_key,
        "panel_size": n_panel,
        "opentargets_targets": len(targets),
        "panel_is_ot_derived": panel_is_ot_derived,
        "top_k": top_k,
        "status": "done",
    }

    if panel_is_ot_derived:
        # Meaningful metric: rank correlation between our cosine shift and OT score.
        rows_with_score = [r for r in rows if r["ensembl_id"] in ot_score_for]
        shifts = [r["mean_cosine_shift"] for r in rows_with_score]
        ot_scores = [ot_score_for[r["ensembl_id"]] for r in rows_with_score]
        rho, n_ranked = spearman_corr(shifts, ot_scores)
        out["spearman_shift_vs_otscore"] = rho
        out["n_ranked"] = n_ranked
        # Also: average OT score in our top-K (high if our top hits are well-scored)
        topk_ot = [ot_score_for[r["ensembl_id"]] for r in rows[:top_k] if r["ensembl_id"] in ot_score_for]
        out["mean_ot_score_in_top_k"] = sum(topk_ot) / len(topk_ot) if topk_ot else float("nan")
        out["mean_ot_score_in_panel"] = sum(ot_scores) / len(ot_scores) if ot_scores else float("nan")
        # Top-K listing
        out["top_k_genes"] = [
            {
                "rank_by_shift": r["rank_by_shift"],
                "ensembl_id": r["ensembl_id"],
                "symbol": ot_symbol_for.get(r["ensembl_id"], r.get("symbol", "")),
                "name":   ot_name_for.get(r["ensembl_id"], ""),
                "mean_cosine_shift": r["mean_cosine_shift"],
                "delta_p_disease":   r["delta_p_disease"],
                "opentargets_score": ot_score_for.get(r["ensembl_id"]),
            }
            for r in rows[:top_k]
        ]
        print(f"\n[{disease_key}] OT-derived panel n={n_panel}, "
              f"Spearman(shift, OT_score)={rho:.3f}, "
              f"mean OT score top-{top_k}={out['mean_ot_score_in_top_k']:.3f} "
              f"vs panel mean={out['mean_ot_score_in_panel']:.3f}")
        print(f"[{disease_key}] top-{min(top_k, 10)} by perturbation shift:")
        for g in out["top_k_genes"][:10]:
            print(f"    rank{g['rank_by_shift']:>3}  {g['symbol']:<10} {g['ensembl_id']}  "
                  f"shift={g['mean_cosine_shift']:.5f}  OTscore={g['opentargets_score']:.3f}")
    else:
        # Standard precision@K vs OT top-N
        recoverable = panel_ensembls & set(ot_score_for.keys())
        our_top_k = rows[:top_k]
        matches = [r for r in our_top_k if r["ensembl_id"] in ot_score_for]
        precision_at_k = len(matches) / max(1, len(our_top_k))
        expected = top_k * len(recoverable) / n_panel if n_panel > 0 else float("nan")
        enrichment = len(matches) / expected if expected > 0 else float("nan")
        out.update({
            "n_recoverable": len(recoverable),
            "precision_at_k": precision_at_k,
            "enrichment_vs_random": enrichment,
            "matches": [
                {
                    "rank_by_shift": r["rank_by_shift"],
                    "ensembl_id": r["ensembl_id"],
                    "symbol": ot_symbol_for[r["ensembl_id"]],
                    "mean_cosine_shift": r["mean_cosine_shift"],
                    "opentargets_score": ot_score_for[r["ensembl_id"]],
                }
                for r in matches
            ],
        })
        print(f"\n[{disease_key}] free-panel n={n_panel}, top_k={top_k}, "
              f"matches={len(matches)}, precision@k={precision_at_k:.3f}, "
              f"enrichment={enrichment:.2f}x")

    out_path = results_root / disease_key / "validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{disease_key}] wrote {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/diseases.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--top-k", type=int, default=100,
                        help="cut-off for our perturbation ranking when computing precision@k")
    parser.add_argument("--opentargets-top", type=int, default=500,
                        help="size of the OpenTargets ranked list to fetch")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    diseases = cfg["diseases"]
    keys = args.only if args.only else list(diseases.keys())

    summary = []
    for key in keys:
        try:
            summary.append(validate_disease(
                key, args.results_root, args.top_k, args.opentargets_top))
        except Exception as exc:
            print(f"[{key}] FAILED: {exc}", file=sys.stderr)
            summary.append({"disease": key, "status": "failed", "error": str(exc)})

    sp = args.results_root / "validation_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {sp}")


if __name__ == "__main__":
    main()
