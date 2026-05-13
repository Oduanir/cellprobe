"""Diagnostic: search CELLxGENE Census for UC / IBD / Crohn cells under any disease label.

Run inside the BioNeMo container."""
import cellxgene_census as cc
import pandas as pd

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)
census = cc.open_soma(census_version="latest")

labels = [
    "ulcerative colitis",
    "Crohn disease",
    "Crohn's disease",
    "inflammatory bowel disease",
    "colitis",
    "ileitis",
]

for label in labels:
    try:
        obs = cc.get_obs(
            census,
            organism="Homo sapiens",
            value_filter=f'disease == "{label}"',
            column_names=["dataset_id", "disease"],
        )
        n = len(obs)
        print(f"{label!r}: {n} cells")
        if n:
            for dsid, c in obs.groupby("dataset_id").size().sort_values(ascending=False).head(3).items():
                print(f"    {dsid}  {c} cells")
    except Exception as e:
        print(f"{label!r}: error: {e}")

print()
print("=== Datasets with tissue_general == large intestine + any disease ===")
obs = cc.get_obs(
    census,
    organism="Homo sapiens",
    value_filter='tissue_general == "large intestine"',
    column_names=["dataset_id", "disease"],
)
print(f"Total large intestine cells: {len(obs):,}")
print(obs.groupby("disease").size().sort_values(ascending=False).head(15))
