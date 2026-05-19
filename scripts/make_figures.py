"""Generate the comparative figures for the CellProbe project.

Reads from `results/` (committed JSONs) and writes PNG + PDF to
`results/figures/`. Pure matplotlib + seaborn — no GPU or BioNeMo needed.

Figures produced:
  - fig1_probe_performance.png — confusion matrices + headline metrics
  - fig2_top10_targets.png     — perturbation top-10 per disease (bar chart)
  - fig3_perturbation_summary.png — shift distributions + Spearman vs OT

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# disease ordering and consistent colours used in every figure
DISEASES = ["uc", "dcm", "luad"]
NAMES = {
    "uc":   "Ulcerative colitis",
    "dcm":  "Dilated cardiomyopathy",
    "luad": "Lung adenocarcinoma",
}
COLOURS = {
    "uc":   "#1f77b4",   # blue (immune/IBD)
    "dcm":  "#d62728",   # red (cardio)
    "luad": "#2ca02c",   # green (onco)
}

# Common style
sns.set_theme(context="notebook", style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.titleweight": "bold",
    "axes.labelweight": "regular",
})


def load_probe_metrics(disease: str) -> dict:
    return json.loads((RESULTS / disease / "probe_metrics.json").read_text())


def load_validation(disease: str) -> dict:
    return json.loads((RESULTS / disease / "validation.json").read_text())


def load_perturbation(disease: str) -> dict:
    return json.loads((RESULTS / disease / "perturbation" / "perturbation.json").read_text())


# ---------------------------------------------------------------------------
# Figure 1 — Probe performance
# ---------------------------------------------------------------------------
def figure_probe_performance():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, d in zip(axes, DISEASES):
        m = load_probe_metrics(d)["test"]
        cm = np.array(m["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=cm, fmt=",d", cbar=False,
            xticklabels=["healthy", "disease"],
            yticklabels=["healthy", "disease"],
            cmap="Blues", vmin=0, vmax=1,
            ax=ax, annot_kws={"size": 11},
        )
        ax.set_title(
            f"{NAMES[d]}\n"
            f"acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}  AUC={m['roc_auc']:.3f}",
            color=COLOURS[d],
        )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if d == DISEASES[0] else "")

    fig.suptitle(
        "Frozen Geneformer + MLP probe — test-set classification",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_probe_performance.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig1_probe_performance.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 2 — Top-10 perturbation candidates per disease
# ---------------------------------------------------------------------------
def figure_top10_targets():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, d in zip(axes, DISEASES):
        v = load_validation(d)
        top = v["top_k_genes"][:10][::-1]   # reverse for horizontal bar (top at top)
        symbols = [g["symbol"] for g in top]
        shifts  = [g["mean_cosine_shift"] * 1e4 for g in top]   # scale 1e-4 → 1
        ot_scores = [g["opentargets_score"] for g in top]

        # bar coloured by OT score
        norm = plt.Normalize(vmin=0.3, vmax=0.9)
        cmap = plt.cm.viridis
        bar_colors = [cmap(norm(s)) for s in ot_scores]
        bars = ax.barh(symbols, shifts, color=bar_colors, edgecolor="black", linewidth=0.4)
        # annotate the OT score next to each bar
        for bar, s in zip(bars, ot_scores):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"OT={s:.2f}", va="center", fontsize=9, color="dimgray")

        rho = v.get("spearman_shift_vs_otscore")
        ax.set_title(
            f"{NAMES[d]}\nSpearman(shift, OT) = {rho:+.2f}" if rho is not None else NAMES[d],
            color=COLOURS[d],
        )
        ax.set_xlabel("cosine shift (×10⁻⁴)")
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(left=0, right=max(shifts) * 1.35)

    # shared colour bar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.012, aspect=40)
    cbar.set_label("OpenTargets evidence score", rotation=270, labelpad=14)

    fig.suptitle(
        "Top-10 perturbation candidates per disease (OpenTargets-curated panel)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_top10_targets.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig2_top10_targets.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 3 — Perturbation summary: shift distribution + Spearman bar
# ---------------------------------------------------------------------------
def figure_perturbation_summary():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: strip plot of all shifts per disease
    ax = axes[0]
    by_disease = []
    for d in DISEASES:
        p = load_perturbation(d)
        for row in p["rows"]:
            if row.get("status") == "done":
                by_disease.append((NAMES[d], row["mean_cosine_shift"] * 1e4, COLOURS[d]))
    diseases_col = [x[0] for x in by_disease]
    shifts_col = [x[1] for x in by_disease]

    sns.stripplot(
        x=diseases_col, y=shifts_col, ax=ax,
        palette={NAMES[d]: COLOURS[d] for d in DISEASES},
        size=6, alpha=0.7, jitter=0.25, edgecolor="black", linewidth=0.3,
        hue=diseases_col, legend=False,
    )
    # overlay median
    for i, d in enumerate(DISEASES):
        vals = [x[1] for x in by_disease if x[0] == NAMES[d]]
        ax.hlines(np.median(vals), i - 0.3, i + 0.3, colors="black", linewidth=2)
    ax.set_ylabel("cosine shift  (×10⁻⁴)")
    ax.set_xlabel("")
    ax.set_title("Per-gene perturbation effect distribution (50 OT genes / disease)")

    # Right: Spearman bar chart
    ax = axes[1]
    rhos = []
    for d in DISEASES:
        v = load_validation(d)
        rhos.append(v.get("spearman_shift_vs_otscore") or 0.0)
    bars = ax.bar(
        [NAMES[d].split()[-1] if d != "luad" else "Lung adeno." for d in DISEASES],
        rhos,
        color=[COLOURS[d] for d in DISEASES],
        edgecolor="black", linewidth=0.4,
    )
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rho:+.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(rhos) * 1.3 + 0.05)
    ax.set_ylabel("Spearman ρ (shift vs OT)")
    ax.set_title("Ranking alignment\nwith OpenTargets")

    fig.suptitle(
        "Perturbation effect signal across diseases",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_perturbation_summary.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig3_perturbation_summary.{{png,pdf}}")


def main() -> None:
    figure_probe_performance()
    figure_top10_targets()
    figure_perturbation_summary()


if __name__ == "__main__":
    main()
