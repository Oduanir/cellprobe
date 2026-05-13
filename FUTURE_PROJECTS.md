# Future projects — backlog Geneformer / BioNeMo

Problèmes drug-discovery non retenus pour le premier projet (A : target ID via in silico perturbation). Conservés ici comme suite logique une fois le premier projet shippé.

## B. Patient stratification / responder prediction en immunothérapie

**Question bio** : À partir de la composition cellulaire et de l'état transcriptomique de la tumeur baseline, peut-on prédire quels patients vont répondre à un checkpoint inhibitor (anti-PD1 / anti-CTLA4) ?

**Étape pharma** : biomarker clinique, translation, patient stratification.

**Tâche ML** : binary classification (responder vs non-responder) au niveau patient ou cellule. Pseudo-bulk possible.

**Datasets candidats** :
- Sade-Feldman et al. 2018 (melanoma, anti-PD1) — gold standard, ~16k cellules
- Jerby-Arnon et al. 2018 (melanoma)
- Bassez et al. 2021 (breast cancer, pre/on-treatment)

**Baseline non-foundation** : LogReg ou MLP sur signatures immuno-tumorales (IFN-γ, T-cell exhaustion, etc.) ou sur HVG.

**Métrique** : AUC, balanced accuracy, F1 macro. Few-shot eval comme angle différenciant.

**Narrative** : clean classification, alignement Enterome immuno/microbiome, hot topic pharma. Plus clinique qu'upstream drug discovery, mais ship probability élevée.

**Suite logique du projet A** : une fois le classifier disease vs healthy validé, on remplace par responder vs non-responder. Pipeline réutilisable.

---

## C. Perturbation response prediction (Perturb-seq)

**Question bio** : Étant donné un KO génique (CRISPR) ou un traitement drug, peut-on prédire la signature transcriptomique de la cellule résultante sans faire l'expérience ?

**Étape pharma** : mechanism of action, hit-to-lead, drug screening in silico.

**Tâche ML** : régression / generation — prédire un transcriptome (vecteur logFC) conditionné sur une perturbation. Plus complexe que classification.

**Datasets candidats** :
- Norman et al. 2019 (Perturb-seq, K562, ~280 perturbations) — utilisé dans le tutoriel BioNeMo officiel
- Replogle et al. 2022 (genome-scale Perturb-seq)
- sci-Plex (drug perturbations)

**Baseline** : moyenne de réponse perturbation-spécifique, ou modèle linéaire conditionné.

**Métrique** : Pearson correlation sur logFC, classification accuracy de la perturbation à partir du transcriptome prédit.

**Narrative** : le plus NVIDIA-aligned (BioNeMo communique beaucoup là-dessus), drug screening in silico. Mais output moins lisible pour un lecteur non-spécialiste.

**Risque** : easy to ship a half-baked version. Évaluation moins parlante côté blog post.

**Suite logique** : projet ambitieux pour démontrer la maîtrise complète de BioNeMo. À envisager une fois le projet A consolidé.

---

## D. Résistance thérapeutique

**Question bio** : Quelles populations cellulaires échappent au traitement, et quelle est leur signature transcriptomique ? Peut-on les prédire à partir de la tumeur pré-traitement ?

**Étape pharma** : résistance acquise, combination therapy design.

**Tâche ML** : binary classification (résistant vs sensible) au niveau cellule, ou identification de populations émergentes post-traitement.

**Datasets candidats** : scRNA-seq pré/post-traitement cancer — fragmentés, à compiler manuellement.

**Baseline** : signature score classique de résistance.

**Narrative** : énorme unmet medical need, mais labels "résistant" débattus, datasets dispersés. Beaucoup de data wrangling pour un narrative qui chevauche B.

**Pourquoi reporté** : data wrangling cost trop élevé pour un premier projet 10-14 jours. À reprendre si un dataset propre devient disponible.
