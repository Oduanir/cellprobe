# Results

Disease-vs-healthy classification using **frozen Geneformer 10M embeddings + sklearn MLP probe**, following the methodology of `bionemo.geneformer.scripts.celltype_classification_bench`.

## Test-set performance

| Disease | n test | Accuracy | F1 (macro) | ROC AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| **Ulcerative colitis** | 11,564 | **0.971** | **0.971** | **0.996** | 0.972 | 0.971 |
| **Dilated cardiomyopathy** | 30,000 | 0.900 | 0.900 | 0.965 | 0.900 | 0.900 |
| **Lung adenocarcinoma** | 30,000 | **0.979** | **0.979** | **0.998** | 0.979 | 0.979 |
| **Mean** | — | **0.950** | **0.950** | **0.986** | — | — |

The frozen pretrained model already encodes disease state strongly across all 3 indications. DCM trails the other two — consistent with the biological intuition that cardiomyopathic remodeling produces subtler per-cell expression shifts than active inflammation (UC) or oncogenic transformation (LUAD).

Train/test gap is 1–3 accuracy points across the three — no overfitting.

## Confusion matrices

(rows = true label, columns = predicted: healthy / disease)

**UC:**
```
              pred_healthy   pred_disease
healthy            6390           154
disease            181          4839
```

**DCM:**
```
              pred_healthy   pred_disease
healthy           13282          1718
disease            1283         13717
```

**LUAD:**
```
              pred_healthy   pred_disease
healthy           14688           312
disease             317         14683
```

## How to reproduce

```bash
# On a g6.xlarge in eu-west-3 with the BioNeMo container and SCDL data already prepared
./scripts/run_probe_pipeline.sh                # extract embeddings + train+eval probes for all 3 diseases
./scripts/run_probe_pipeline.sh --probe-only   # re-run only the sklearn MLP (no GPU needed)
```

See `docs/SETUP_AWS.md` for the full environment + data-prep procedure.

## Methodological notes

- **Why frozen + probe instead of full fine-tuning?** This is the methodology NVIDIA themselves use to benchmark Geneformer (see `celltype_classification_bench/bench.py` shipped with the framework). It's faster (~30 min/disease vs hours), more reproducible (sklearn is deterministic given a seed), and the resulting classifier is sufficient for downstream in-silico perturbation analysis — perturbing a gene's token in the input and observing the cell-embedding shift through the MLP is the same operation regardless of whether the encoder was fine-tuned.
- **Embedding dimension**: 256 (Geneformer 10M hidden size).
- **Probe architecture**: `StandardScaler → MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, n_iter_no_change=50, max_iter=500, random_state=1337)`.
- **Splits**: stratified 70/15/15 by `disease_label` per disease; reproducible with `seed=42`.

## Raw artifacts

- Per-disease metrics: `results/<disease>/probe_metrics.json`
- Cross-disease summary: `results/probe_summary.json`
- Embedding extraction log: `results/extract_embeddings_summary.json`
- Trained probes (not committed; ~100 KB each, regenerable from the script): `results/<disease>/probe.joblib`
