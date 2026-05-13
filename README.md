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

1. A public GitHub repo with a clean, reproducible fine-tuning pipeline for Geneformer on a chosen biological task.
2. A blog post (Medium or LinkedIn) walking through the project — "Fine-tuning Geneformer with NVIDIA BioNeMo for [task]". The blog post is the differentiator: NVIDIA Solutions Architects communicate constantly (whitepapers, hackathons, demos).
3. Bonus if time allows: inference deployment via Triton Inference Server, with a latency/throughput benchmark.

The goal is **not** scientific novelty. It is a credible, well-documented technical demonstration that can be linked from the CV / LinkedIn / cover letter.

## Technical scope

### Dataset (to be decided)

Options to evaluate:
- **CELLxGENE Discover** — curated public scRNA-seq datasets, easy to filter.
- A **cancer scRNA-seq** dataset (e.g., pan-cancer atlas subset) — strong narrative for drug discovery audience.
- A **disease vs control** scRNA-seq dataset — clean binary classification task, easy to benchmark.
- Something **microbiome / immuno-related** if a suitable scRNA-seq dataset exists — capitalizes on Enterome experience.

Decision criteria:
- Dataset size manageable for fine-tuning on a single A100 (<= ~100k cells ideal)
- Public, well-annotated labels
- A baseline non-foundation method exists in the literature for comparison

### Task

Pick the **simplest task that demonstrates fine-tuning value**:
- Cell type classification (multi-class)
- Disease state classification (binary)
- Perturbation response prediction (more ambitious)

Start with cell type classification on a labeled dataset. Move to disease state if time permits.

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

Geneformer fine-tuning requires more VRAM than the candidate's RTX 4070 Super (12GB). Options:

- **Local dev** on RTX 4070 Super (remote access via `ssh gpu`) — use a small subset for pipeline validation
- **Final runs on rented A100**:
  - Lambda Labs, RunPod, or Vast.ai — A100 40GB at ~$1–2 / hour
  - Budget envelope: ~$50–100 for the full project
- **Free credit alternatives**: NVIDIA Launchpad (free trial), Google Colab Pro+ with A100 (~$50/month)

Recommended: develop locally on the 4070 Super with a tiny dataset slice; do the final training + eval runs on a rented A100.

## Timeline (10–14 days, focused)

| Day | Milestone |
|-----|-----------|
| 1–2 | Environment setup, BioNeMo install, Geneformer checkpoint loading, GPU access confirmed |
| 3   | Dataset chosen and downloaded; preprocessing pipeline drafted |
| 4–5 | Tokenization + data loaders working end-to-end on a tiny subset |
| 6–9 | Fine-tuning runs; iterate on hyperparameters; full-dataset training on A100 |
| 10–11 | Evaluation, baseline comparison, figures |
| 12–13 | Write-up: README + blog post draft |
| 14  | Polish, publish blog post, link from LinkedIn / CV |

## Open decisions (to make in next session)

1. **Final dataset choice** — cancer scRNA-seq vs disease-state vs immune/microbiome. Pick one and stick to it.
2. **Final task** — cell type classification (safer) vs disease state (better narrative).
3. **BioNeMo install path** — NGC container (closer to NVIDIA's production setup, more impressive) vs pip install (faster to iterate).
4. **GPU rental provider** — Lambda Labs vs RunPod vs Vast.ai. Lambda Labs tends to be the cleanest.
5. **Blog post target** — Medium (broader reach) vs LinkedIn article (better signal to NVIDIA recruiters specifically). Could do both.

## Success criteria

- Repo is public, well-documented, reproducible from scratch
- Blog post is published before the NVIDIA application is submitted
- Application cover letter and LinkedIn explicitly reference the project
- Fine-tuned Geneformer beats the non-foundation baseline on the chosen task (even by a small margin)
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
