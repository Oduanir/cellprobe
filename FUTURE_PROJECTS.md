# Future work — natural extensions of CellProbe

The CellProbe pipeline (frozen Geneformer + linear probe + in-silico perturbation, three diseases, OpenTargets validation) is a deliberately scoped 2-week demonstration. Three larger problems share the same backbone — same data sources, same BioNeMo container, same SCDL pipeline — and are natural follow-ups.

## 1. Responder prediction for immune-checkpoint blockade (ICB)

**Biological question.** From the baseline tumour transcriptome at single-cell resolution, can we predict which patients will respond to anti-PD1 / anti-CTLA4 therapy?

**ML task.** Binary classification (responder vs. non-responder), cell- or patient-level (with pseudo-bulk aggregation).

**Candidate datasets.**
- Sade-Feldman *et al.* 2018 (melanoma, anti-PD1) — gold standard, ~16 k cells
- Jerby-Arnon *et al.* 2018 (melanoma)
- Bassez *et al.* 2021 (breast cancer, pre/on-treatment)

**Baseline.** Logistic regression or a small MLP on immune-tumour signatures (IFN-γ, T-cell exhaustion) or on highly-variable genes.

**Metric.** ROC-AUC, balanced accuracy, macro-F1. Few-shot performance is especially informative here.

**Why it's a natural follow-up.** The CellProbe disease-vs-healthy classifier is one swap away: replace the binary label, keep the entire upstream pipeline. The perturbation analysis then identifies which genes are most predictive of treatment response — a directly actionable biomarker question.

## 2. Perturbation response prediction (Perturb-seq)

**Biological question.** Given a genetic knock-out (CRISPR) or a drug treatment, can we predict the resulting transcriptomic signature of a cell *without* running the wet experiment?

**ML task.** Regression / generation — predict a transcriptome (log fold-change vector) conditioned on a perturbation identifier. More complex than classification.

**Candidate datasets.**
- Norman *et al.* 2019 (Perturb-seq, K562, ~280 perturbations) — used in BioNeMo's own tutorial
- Replogle *et al.* 2022 (genome-scale Perturb-seq)
- sci-Plex (drug perturbations)

**Baseline.** Per-perturbation mean response, or a linear model conditioned on the perturbation embedding.

**Metric.** Pearson correlation on log fold-change vectors; classification accuracy of perturbation identity from the predicted transcriptome.

**Why it's a natural follow-up.** This is the use case BioNeMo communicates most actively. It exercises the BioNeMo training stack (not just inference) and lands directly in the in-silico drug-screening narrative.

## 3. Therapeutic resistance

**Biological question.** Which cell populations escape treatment, and what is their transcriptomic signature? Can resistant subpopulations be predicted from the pre-treatment tumour?

**ML task.** Binary classification (resistant vs. sensitive) at cell level, or identification of populations emerging post-treatment.

**Candidate datasets.** Pre/post-treatment scRNA-seq from cancer studies — fragmented across publications, requires data wrangling to assemble.

**Baseline.** Classical resistance signature score.

**Why it's parked for now.** The data-wrangling cost is high relative to a 2-week first project: there is no single curated resistance atlas comparable to Reichart, Oliver, or LuCA for CellProbe's three diseases. To unlock this problem, the right move is either to wait for such an atlas to be deposited in the Census, or to dedicate a separate effort to building one.

---

These three extensions, plus the methodological improvements parked in [`docs/PROJECT_HISTORY.md`](docs/PROJECT_HISTORY.md) (end-to-end fine-tuning instead of frozen probe, gradient-based perturbation), are the priority follow-ups if CellProbe were to grow beyond a proof-of-concept.
