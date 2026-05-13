# RTX-CODE — Fine-tuning Geneformer with NVIDIA BioNeMo

## Context

This project exists to support an application to **NVIDIA — Solutions Architect, AI for Drug Discovery (EMEA)**.

The job is customer-facing technical advisory for biopharma clients adopting NVIDIA's AI stack (BioNeMo, Clara, Parabricks, foundation models, agentic AI). The candidate profile is strong on the biology/genomics side (PhD biostats, Head of DS at Enterome, NGS / RNA-seq / microbiome / drug development at Pharnext) but lighter on:

- Hands-on fine-tuning of transformer foundation models
- NVIDIA-specific tooling (BioNeMo, Triton, TensorRT, CUDA)
- Foundation-model-era ML engineering at scale

A public GitHub project demonstrating fine-tuning a biological foundation model **using NVIDIA's own framework**, with a clean write-up, would close that gap visibly and demonstrate ability to be productive day-1 in the role.

## Why this specific project

**Geneformer + BioNeMo** is the strategic sweet spot:

- **Geneformer** is a transformer foundation model for single-cell genomics, pretrained on ~30M human single cells. It learns gene network representations and can be fine-tuned for downstream tasks (cell type classification, perturbation prediction, disease state).
- **BioNeMo** is NVIDIA's drug discovery framework. It ships Geneformer as a supported model with optimized training recipes, GPU acceleration, and Triton deployment paths.
- **Single-cell genomics** is exactly the domain expertise the candidate already has (Enterome NGS / microbiome / RNA-seq, Pharnext translational). No domain ramp-up needed — focus stays on the ML/NVIDIA tooling.
- **Coverage of role keywords**: foundation model fine-tuning, transformer training, healthcare/life sciences AI, NVIDIA platform, inference deployment, communication of complex AI to non-experts.

Alternatives considered:
- **ESM-2 fine-tuning** (protein language model) — easier, lighter compute, well-trodden path. Fallback if Geneformer setup proves too painful.
- **MolMIM / DiffDock** (molecular generation, docking) — strong NVIDIA showcase but further from the candidate's bio expertise; would dilute the "I am the customer profile" angle.
- **Agentic LLM for pharma literature** — on-trend but generic; doesn't show bio depth.

Geneformer wins because it forces engagement with BioNeMo specifically AND leverages existing genomics expertise.

## Goal

Ship, in 10–14 days of focused work:

1. A public GitHub repo with a clean, **pipeline-first** Geneformer fine-tuning framework applied to **three drug-discovery-relevant indications** (cardio, immuno, onco). One pipeline, three configs, three result sets — the reusable methodology is the deliverable, not a single benchmark.
2. A blog post (LinkedIn primary + Medium cross-post) walking through the project — *"Fine-tuning Geneformer with NVIDIA BioNeMo for target identification across three drug-discovery indications"*. The blog post is the differentiator: NVIDIA Solutions Architects communicate constantly (whitepapers, hackathons, demos).
3. Bonus if time allows: inference deployment via Triton Inference Server, with a latency/throughput benchmark.

The goal is **not** scientific novelty. It is a credible, well-documented technical demonstration that can be linked from the CV / LinkedIn / cover letter — and that signals "I built something a customer could adopt".

## Technical scope

### Diseases

Three indications, all sourced from CELLxGENE Census (BioNeMo's `bionemo-geneformer` package depends on `cellxgene_census` — supported path). Final dataset selections after auditing what's actually deposited in the curated Census:

| Disease | Dataset (CELLxGENE Census) | Cells downloaded | Disease / healthy split | Why |
|---|---|---|---|---|
| **Dilated cardiomyopathy (DCM)** | Reichart et al. 2022, *Science* — "DCM/ACM heart cell atlas: All cells" (DOI 10.1126/science.abo1984) | 764,953 | 482,581 / 282,372 | Chaffin 2022 (the Theodoris/Geneformer benchmark) is not in the Census; Reichart 2022 is a richer DCM atlas with healthy controls in the same study. Heart failure has extensive OpenTargets coverage. |
| **Ulcerative colitis (UC)** | Oliver et al. 2024, *Nature* — "Extended - All genes" (DOI 10.1038/s41586-024-07571-1) | 85,891 (healthy subsampled at download from 1.07M to 50k for class balance) | 35,891 / 50,000 | Smillie 2019 was the initial pick but its Census deposit contains only the healthy subset (no UC cells). Oliver 2024 is a more recent IBD atlas with both classes properly labeled. Immuno indication, Enterome-aligned. |
| **Lung adenocarcinoma (LUAD)** | Salcher et al. 2022, *Cancer Cell* — "Single-cell lung cancer atlas (LuCA) -- core atlas" (DOI 10.1016/j.ccell.2022.10.008) | 623,816 | 410,927 / 212,889 | LuCA core is a meta-atlas integrating multiple LUAD studies (including Kim 2020) with shared healthy lung controls — better than any single primary study. Oncology, Enterome-aligned. |

Dataset selection script: `src/data/download.py` (parameterized by `configs/diseases.yaml`).

### Task

**Target identification via in silico perturbation** — the Geneformer paper's flagship use case applied at scale across three indications:

1. Fine-tune Geneformer on disease-vs-healthy binary classification per indication.
2. Perturb gene embeddings (gene by gene) and rank candidates by their effect on shifting the disease cell state toward healthy.
3. Validate the ranked list against known drug targets in **OpenTargets** / **DGIdb** for the disease. Metric: enrichment of known targets in top-K vs random.
4. Compare across the 3 indications — does the same methodology generalize? Where does it fail?

Parked for follow-up (see `FUTURE_PROJECTS.md`): responder prediction, Perturb-seq response, resistance.

### Pipeline

1. **Environment setup**
   - BioNeMo Framework (NVIDIA NGC container or pip install)
   - PyTorch + HuggingFace as needed
   - GPU access (see compute section)
2. **Data prep**
   - Download dataset
   - QC + filtering (standard scRNA-seq preprocessing)
   - Tokenization for Geneformer (ranked gene representation)
3. **Fine-tuning**
   - Load Geneformer pretrained checkpoint from BioNeMo
   - Fine-tune on the chosen task using BioNeMo's recipes
   - Track training (loss curves, val metrics)
4. **Evaluation**
   - Test set metrics (accuracy, macro-F1, confusion matrix)
   - Comparison vs a non-foundation baseline (logistic regression on highly-variable genes, or a simple MLP)
   - Few-shot evaluation if interesting (does Geneformer beat baseline with very little labeled data?)
5. **(Bonus) Inference deployment**
   - Export model
   - Deploy via Triton Inference Server
   - Benchmark latency / throughput on GPU

### Deliverables

- `README.md` — project overview, results, how to reproduce
- `notebooks/` — exploratory and reporting notebooks
- `src/` — clean Python modules for data prep, training, eval
- `configs/` — training configs (YAML)
- `results/` — metrics, figures
- `blog/` — draft of the blog post (Markdown), to be cross-posted on Medium / LinkedIn

## Compute

Two-tier compute matching what an SA would recommend to a customer: consumer GPU for pipeline validation, A100 for production.

| | Local dev | Production runs |
|---|---|---|
| Hardware | RTX 4070 SUPER (12 GB) — pipeline validation, smoke tests | **Lambda Labs A100 40 GB** (~$1.10/h) |
| Access | `ssh gpu` over Tailscale to WSL2 host | Direct SSH from Mac |
| Workload | Tokenization sanity, 1-epoch smoke run, in silico perturbation testing | 3 full fine-tuning runs, in silico perturbation production sweep |

Budget envelope: **$50-100**. Realistic A100 expectation: 10-20 h × $1.10 ≈ $11-22 across the three diseases. See `docs/SETUP.md` for the full Day 1-2 procedure.

## Timeline (10–14 days, focused)

| Day | Milestone | Status |
|---|---|---|
| 1–2 | WSL2 + Docker + NVIDIA Container Toolkit; NGC login; BioNeMo 2.7.1 pull; Geneformer 10M smoke test on RTX 4070 SUPER | ✅ done 2026-05-13 |
| 3 | Source DCM/UC/LUAD scRNA-seq from CELLxGENE Census; data download script | |
| 4–5 | Parameterized data prep pipeline (QC + Geneformer tokenization + SCDL conversion); same code, 3 configs | |
| 6 | Smoke fine-tuning runs on RTX 4070 SUPER (1 epoch, subset) to validate the training loop | |
| 7–8 | Production fine-tuning on Lambda Labs A100 — 3 disease classifiers | |
| 9 | In silico perturbation analysis (gene embedding perturbation, candidate target ranking) | |
| 10 | OpenTargets / DGIdb validation — enrichment of known targets in top-K vs random | |
| 11 | Cross-disease comparison figures, classifier metrics table, baseline LogReg/MLP comparison | |
| 12–13 | Blog post draft (LinkedIn + Medium) — narrative around methodology generalizability | |
| 14 | Polish, publish, link from LinkedIn / CV / cover letter | |

## Decisions locked

| Question | Choice | Date |
|---|---|---|
| Biological problem | **A — Target ID via in silico perturbation** (Geneformer flagship use case) | 2026-05-12 |
| Diseases | **DCM + UC + LUAD** (cardio + immuno + onco) — pipeline-first, three configs | 2026-05-13 |
| Task | Disease-vs-healthy binary classification + gene-embedding perturbation ranking | 2026-05-13 |
| BioNeMo install | **NGC container** `nvcr.io/nvidia/clara/bionemo-framework:2.7.1` (pinned) | 2026-05-12 |
| GPU rental | **Lambda Labs A100 40GB** (~$1.10/h) | 2026-05-12 |
| Blog target | **LinkedIn primary + Medium cross-post** | 2026-05-12 |

## Success criteria

- Repo is public, well-documented, reproducible from scratch (one pipeline, three configs, three result sets)
- Blog post is published before the NVIDIA application is submitted
- Application cover letter and LinkedIn explicitly reference the project
- Fine-tuned Geneformer beats a non-foundation baseline (LogReg / MLP on HVG) on at least 2 of the 3 indications
- In silico perturbation recovers a non-trivial fraction of OpenTargets known drug targets in the top-K predictions, on at least DCM (the paper-anchored baseline)
- Bonus: a recruiter or hiring manager comments on the blog post

## Fallback plan

If BioNeMo setup or Geneformer fine-tuning hits a wall after 3–4 days, pivot to **ESM-2 fine-tuning for a protein function task** (HuggingFace + PyTorch, plenty of tutorials, lighter compute). Worse positioning but ships reliably. Better to ship a smaller project than fail to ship the ambitious one.

## Links and references

- BioNeMo Framework: https://docs.nvidia.com/bionemo-framework/
- Geneformer paper: Theodoris et al., Nature 2023
- CELLxGENE Discover: https://cellxgene.cziscience.com/
- NVIDIA Launchpad (free GPU trial): https://www.nvidia.com/en-us/launchpad/
- Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/

## Job posting reference

NVIDIA — Solutions Architect, AI for Drug Discovery (EMEA) — JR2017335
