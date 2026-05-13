"""Find which CELLxGENE Census datasets contain UC cells (and matching healthy controls)."""
import cellxgene_census as cc
import pandas as pd

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 220)
census = cc.open_soma(census_version="latest")

# Datasets with UC cells
print("=== Datasets containing UC cells ===")
obs = cc.get_obs(
    census,
    organism="Homo sapiens",
    value_filter='disease == "ulcerative colitis"',
    column_names=["dataset_id", "tissue_general"],
)
uc_per_ds = obs.groupby("dataset_id").size().sort_values(ascending=False)
print(uc_per_ds)
print()

# For top datasets, get titles + matching healthy counts
ds = census["census_info"]["datasets"].read().concat().to_pandas()
top_uc_dsids = uc_per_ds.head(5).index.tolist()
print("=== Titles + matching healthy counts ===")
for dsid in top_uc_dsids:
    row = ds[ds["dataset_id"] == dsid]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    # matching healthy
    healthy = cc.get_obs(
        census,
        organism="Homo sapiens",
        value_filter=f'dataset_id == "{dsid}" and disease == "normal"',
        column_names=["dataset_id"],
    )
    print(f"DS {dsid}")
    print(f"  title : {r['dataset_title']}")
    print(f"  collection: {r['collection_name']}")
    print(f"  DOI   : {r['collection_doi']}")
    print(f"  UC    : {uc_per_ds[dsid]:,} cells   healthy in same DS: {len(healthy):,} cells")
    print()
