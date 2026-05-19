# Design notes

This document captures the engineering rationale behind CellProbe: which technical choices were made, which alternatives were considered, and which pivots happened during the build. The public-facing project description is in the top-level [`README.md`](../README.md); headline results are in [`RESULTS.md`](../RESULTS.md).

## Why Geneformer on BioNeMo

**Geneformer** (Theodoris *et al.* 2023, *Nature*) is a transformer foundation model trained on the rank-ordered expression of single cells from ~30 M human cells in the CELLxGENE Census. It encodes each cell as a vector that captures gene-regulatory state, and supports two downstream patterns: fine-tuning a head for classification, and perturbing input tokens to interrogate causality. Both patterns are directly relevant to **target identification** — the canonical first step of drug discovery.

**NVIDIA BioNeMo** ships Geneformer (10 M and 106 M variants) as a first-class model: a pinned NGC container, CLI entrypoints (`infer_geneformer`, `convert_h5ad_to_scdl`, `bionemo-geneformer-train`), and a single-cell data loader (`SCDL`) optimised for memory-mapped sparse matrices. Building on BioNeMo rather than re-implementing keeps the project anchored to a reproducible, externally maintained reference and makes it directly relevant for biopharma teams evaluating the stack.

## Alternatives considered

| Alternative | Why interesting | Why not chosen |
|---|---|---|
| **ESM-2 fine-tuning** (protein language model) | Well-trodden path with abundant tutorials, lighter compute. | Less coupled to BioNeMo's strengths; demonstrates HuggingFace skills more than NVIDIA-stack skills. Kept as a fallback if Geneformer setup proved intractable. |
| **MolMIM / DiffDock** (molecular generation, docking) | Strong NVIDIA showcase across small-molecule discovery. | Further from single-cell genomics; would require domain ramp-up unrelated to the project's core. |
| **Agentic LLM for pharma literature** | On-trend topic. | Generic — would not exercise BioNeMo's biology-specific infrastructure (SCDL, Geneformer, model serving). |

Geneformer + BioNeMo wins on the intersection of **NVIDIA-native tooling** and **single-cell biology that has its own canonical use case** (in-silico perturbation per Theodoris 2023).

## Three indications, one pipeline

A single-disease demo would be a benchmark. Three diseases on the **same code with three configs** demonstrates a *reusable methodology*, which is the actual deliverable. Coverage chosen to span the major drug-discovery verticals:

| Indication | Vertical | Dataset chosen | Why this deposit |
|---|---|---|---|
| **Dilated cardiomyopathy (DCM)** | Cardio | Reichart *et al.* 2022, *Science* | Chaffin 2022 (used in the Theodoris benchmark) is not in the curated Census deposit. Reichart 2022 is a richer DCM atlas with healthy controls in the same study. |
| **Ulcerative colitis (UC)** | Immuno / IBD | Oliver *et al.* 2024, *Nature* | Smillie 2019 was the first pick but its Census deposit contains only the healthy subset. Oliver 2024 is a 2024 IBD atlas with both classes properly labelled. |
| **Lung adenocarcinoma (LUAD)** | Onco | Salcher *et al.* 2022, *Cancer Cell* (LuCA core atlas) | LuCA is a meta-atlas integrating multiple LUAD studies (including Kim 2020) with shared healthy lung controls — better than any single primary study. |

The dataset selection script reads `configs/diseases.yaml`, which exposes the dataset IDs, disease/healthy labels, and per-disease sampling caps. Swapping a dataset is a one-line edit.

## Methodology pivot: fine-tuning → frozen probe

The original plan called for **end-to-end fine-tuning** of Geneformer on a disease-vs-healthy classification head per indication, following the Theodoris paper. After auditing what BioNeMo 2.7.1 actually ships, the picture changed:

- BioNeMo does not provide a turn-key fine-tuning CLI for sequence-level classification. End-to-end fine-tuning would require either custom YAML configs against `train_geneformer` (Pydantic-validated, undocumented for binary classification) or HuggingFace-based code that abandons the BioNeMo stack.
- BioNeMo *does* ship a benchmark — `bionemo.geneformer.scripts.celltype_classification_bench` — that runs inference on a pretrained checkpoint to extract cell embeddings, then trains a 3-layer scikit-learn MLP on top. **This is the methodology NVIDIA themselves benchmark Geneformer with.**

The project pivoted to this **frozen-encoder + linear probe** pattern. It is faster (no GPU training instability, ~30 min per disease vs hours), deterministic, and aligned with the official benchmark — the right engineering choice for a 2-week proof-of-concept. The pipeline is otherwise unchanged: SCDL conversion, Geneformer inference, downstream classifier, perturbation.

Trade-off documented: a fully fine-tuned classifier would likely close the residual gap on DCM (90 % accuracy vs 97-98 % on UC/LUAD), and would let the perturbation analysis use the classifier's gradient rather than a frozen encoder's embedding shift. Both are natural next steps.

## Compute decisions

| Decision | Choice | Rationale |
|---|---|---|
| Container vs. pip install | Pinned NGC container `nvcr.io/nvidia/clara/bionemo-framework:2.7.1` | Reproducibility — pinned tag rather than `:latest` or `:nightly` so any future reproduction lands on the same artefact. |
| Cloud provider | AWS EC2 in `eu-west-3` (Paris) | EMEA data residency is a real customer concern even when the data is public, and Paris keeps the latency low from the developer's location. |
| Instance type | `g6.xlarge` (1× NVIDIA L4 24 GB, Ada Lovelace) — ~$0.805/h | `g5.xlarge` (A10G) is **not offered in `eu-west-3`** as of 2026-05; only `g4dn` (T4) and `g6` (L4) are. L4 is the newer architecture, cheaper than A10G would have been, and ships fast FP8 / INT8 tensor cores. |
| EBS volume | 200 GB gp3 | BioNeMo container is ~55 GB, the three SCDL-converted disease panels add ~5 GB. 100 GB is too tight after the container is pulled. |
| Stop-when-idle | Routine `aws ec2 stop-instances` between sessions | Compute pause; only EBS storage continues (~$0.30/day). Total project compute spend stayed under $2. |

A local-GPU development path on a WSL2 + RTX 4070 SUPER setup is preserved in [`SETUP.md`](SETUP.md) for reference. The local path validated the BioNeMo container before the AWS pivot, but is not the active workflow.

## In-silico perturbation: candidate selection

The first naive implementation drew candidate genes by **mean expression** in the disease cell subset (top-K most expressed genes). The cosine-shift effect of perturbing those candidates was empirically zero. Diagnosed root cause: Geneformer's tokeniser normalises each gene's expression by a **global per-gene median** before ranking; constitutively highly-expressed genes (e.g. IGKC, with mean expression 275 in the UC subset) get normalised to ~1 and do not make the top-2048 token sequence — so zeroing them in the input doesn't change anything downstream.

The corrected design draws candidates from **OpenTargets' disease–target catalogue**, filtered to (a) genes present in the AnnData `var` and (b) genes whose Ensembl ID appears as a token in the baseline inference output. This guarantees that every perturbation can have an effect, and ties the candidate panel to clinically-relevant biology from the start. The Spearman correlation between our cosine-shift ranking and OpenTargets' evidence score (+0.11 to +0.29 across indications) is positive but well below 1.0 — the desired regime, where the model proposes a *complementary* ranking rather than a recapitulation of the catalogue.

## Risk register and fallback

The pre-build risk register and the fallback path (pivot to ESM-2 fine-tuning if BioNeMo blocked after 3-4 days) are kept in the source for reference but were never triggered. The methodology pivot above (fine-tuning → probe) was an *opportunistic* simplification, not a fallback under duress.

## What is parked for follow-up

See [`../FUTURE_PROJECTS.md`](../FUTURE_PROJECTS.md):

- Responder prediction (ICB / immune-checkpoint blockade)
- Perturb-seq response prediction
- Drug-resistance trajectories

These are larger problems that share the data and methodology backbone of CellProbe and would naturally extend it.
