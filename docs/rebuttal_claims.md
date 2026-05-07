# Rebuttal claims and evidence map

Working document for the response to Reviewer 1, comment 4 ("RDKit descriptors
are deterministic — does MTL learn deeper chemistry, or just feature
extraction?"). Each claim links to the result file that backs it. Pending
claims are listed with the experiment that will resolve them.

All numbers below are on the lipid task unless otherwise stated. Manuscript
reference: ChemBERTa-RT R²=0.820, MAE=52.8 — ChemBERTa+RDKit (MTL) R²=0.842,
MAE=45.1 (median across 15 epochs, paper Section 3.1).

---

## A. Claims now established by evidence on this branch

**A1. RDKit descriptors alone do not reproduce ChemBERTa-RT performance.**
The best classical model on the seven descriptors reaches R²=0.559 (RandomForest, 200 trees, 5 seeds), MAE=83.1 — well below the manuscript's ChemBERTa-RT (R²=0.820). So a non-trivial share of the lipid RT signal lives in SMILES information that the seven descriptors do not capture, and "ChemBERTa is just learning to recover the descriptors" is excluded as a complete explanation.
Evidence: `lipid_rdkit_only_baseline_metrics.csv`.

**A2. Vanilla (non-fine-tuned) ChemBERTa embeddings carry only modest RT signal.**
Closed-form linear fit on top of frozen ChemBERTa CLS embeddings (768-dim): R²=0.387, MAE=108.4 — strictly the best a 15-epoch AdamW-trained frozen variant could reach, since the convex MSE objective is solved exactly here. Gap to fine-tuned ChemBERTa-RT (R²=0.820) is +0.43 R², so end-to-end fine-tuning is doing the dominant share of representation learning, not the ZINC pre-training alone.
Evidence: `lipid_chemberta_frozen_metrics.csv` (model = `ChemBERTa-frozen-RT (CLS only) + OLS / Ridge1`).

**A3. Concatenating the seven RDKit descriptors with frozen ChemBERTa adds limited headroom.**
Frozen Concat best (Ridge α=1): R²=0.459, MAE=100.2. Compared to frozen-CLS-only at R²=0.387, that is +0.072 R² from descriptor concatenation — modest, and far from the +0.43 R² delta produced by fine-tuning.
Evidence: `lipid_chemberta_frozen_metrics.csv` (model = `ChemBERTa-Concat (frozen, closed-form) + Ridge1`).

**A4. A 7-feature RandomForest beats frozen ChemBERTa-Concat.**
RandomForest on the seven descriptors (R²=0.559) beats frozen ChemBERTa-Concat with optimal linear fitting on 768+7 dims (R²=0.459). In the no-fine-tuning regime, the choice of head matters more than whether ChemBERTa is involved at all. Fine-tuning is the lever that makes the encoder worth using; frozen + concat is not a competitive design point.
Evidence: cross-reference of `lipid_rdkit_only_baseline_metrics.csv` and `lipid_chemberta_frozen_metrics.csv`.

**A5. The descriptor set is internally heterogeneous: collinear in places, near-inert linearly in others, dominated by a feature that contributes only nonlinearly.**
On MinMax-scaled inputs:
- `mol_weight` (Ridge coef +1.19) and `heavy_atoms` (−0.90) are classic collinear size proxies — both correlate +ve with RT univariately, but the linear model uses one to cancel the other.
- `rotatable_bonds` is near-zero in the linear fit (coef +0.01) and univariate (r=+0.01) — contributes essentially nothing linear.
- `polar_surface_area` has univariate r=−0.02 yet RandomForest importance 0.32 (top of the seven) — a feature whose contribution is purely interaction-mediated.

Implication for the rebuttal: even if MTL "only" teaches the model to recover the descriptors, doing so via SMILES is itself a non-trivial nonlinear regression problem. The descriptor set is also redundant in places (mol_weight ↔ heavy_atoms) — defensible to discuss in a future-work or limitations note.
Evidence: `lipid_baseline_feature_importance.csv`, `scripts/baseline_feature_importance_lipid.py`.

**A6. Sklearn-baseline seed variance is negligible at this dataset size.**
Across 5 random seeds, RandomForest test R² varies by ~0.001 (0.5593 to 0.5605); Ridge and GradientBoosting are deterministic up to floating-point noise. The 63k-train / 7.8k-test split is large enough that single-model run-to-run variance is small — a useful prior for sizing the multi-seed ChemBERTa runs (a handful of seeds will give tight error bars on the headline R² numbers).
Evidence: `lipid_rdkit_only_baseline_metrics.csv` (per-seed columns).

---

## B. Claims contingent on pending experiments

**B1. (Central rebuttal claim) Fine-tuned ChemBERTa-Concat does or does not match ChemBERTa+RDKit (MTL) on lipid RT.**
This is the experiment Reviewer 1 explicitly requested. Decision rule (pre-committed):
- If R²_test ≥ 0.84 (≈ MTL): reviewer's null is supported. MTL is best framed as a feature-engineering convenience rather than as inducing fundamentally richer representations. Manuscript needs to be reframed accordingly.
- If R²_test ≈ 0.82 (≈ ChemBERTa-RT, no descriptors): concat in the fine-tuned regime adds nothing; the +0.02 R² that MTL delivers is genuinely from the auxiliary supervision objective, not from "having access to descriptor values." Strongest support for the paper's claim.
- If 0.82 < R²_test < 0.84: partial. Both signals contribute. Manuscript should sharpen the claim from "MTL learns deeper chemistry" to "MTL contributes incrementally on top of descriptor availability."
Pending: `chemberta_concat_baseline_lipid.ipynb` on Colab GPU, multi-seed.

**B2. Fine-tuned ChemBERTa-Concat behaves analogously on peptide RT.**
Same decision rule applied to the paper's peptide numbers (ChemBERTa-RT R²=0.743 vs ChemBERTa+RDKit R²=0.757). The paper's primary "MTL learns richer chemistry" claim is established on peptides, so this is where the reviewer's argument actually lands.
Pending: `chemberta_concat_baseline_peptide.ipynb` on Colab GPU, multi-seed.

**B3. The frozen-encoder upper bound is tight.**
The closed-form result (A2/A3) bounds the notebook's 15-epoch AdamW-trained frozen variant from above. If the AdamW frozen run lands close to the closed-form numbers (within a few hundredths of R²), the bound is tight and we can defensibly cite the closed-form as representative; if there is a meaningful gap, AdamW + early stopping is acting as additional regularization beyond Ridge.
Pending: frozen branch of `chemberta_concat_baseline_lipid.ipynb` on Colab.

**B4. The descriptor-only floor and feature-importance pattern hold on peptide RT.**
A1, A5, A6 are lipid-specific so far. The peptide domain has different chemistry and a different descriptor distribution (mostly-linear amino-acid polymers); the descriptor set might be more or less redundant there. Determines whether the rebuttal's "even worst-case, MTL is doing nontrivial nonlinear work" argument generalizes or needs to be lipid-scoped.
Pending: `chemberta_concat_baseline_peptide.ipynb` Section 1 (sklearn baseline; runs on CPU once peptide CSVs are accessible) + an analogous `scripts/baseline_feature_importance_peptide.py`.

---

## C. Adjacent claims (not yet scoped)

**C1. MTL pre-training transfers to lipid better than concat pre-training.**
The paper's headline claim is about transfer, and the reviewer's argument bites differently here: peptide-time concatenation cannot shape the encoder (descriptors flow around it), while MTL gradients explicitly do. This is the comparison most likely to favor MTL even if MTL ≈ Concat in direct training. Would require a new transfer-experiment notebook that mirrors `chemberta_TransferLearning_peptide_RTrdkit_Lipidmetlin_250k_*` but with a peptide-RT-only and a peptide-RT-concat variant as additional source-task baselines.

**C2. The descriptor set could be pruned without loss.**
A5 suggests `mol_weight` ↔ `heavy_atoms` collinearity and a near-inert `rotatable_bonds`. An MTL run with a reduced auxiliary-target set (e.g. drop `heavy_atoms`, drop `rotatable_bonds`) would test whether the auxiliary supervision is robust to descriptor pruning. Worth mentioning as a limitation / future work even if not run.

---

## Status snapshot

| Claim | Status | Evidence file |
|---|---|---|
| A1 | ✅ established | `lipid_rdkit_only_baseline_metrics.csv` |
| A2 | ✅ established | `lipid_chemberta_frozen_metrics.csv` |
| A3 | ✅ established | `lipid_chemberta_frozen_metrics.csv` |
| A4 | ✅ established | both CSVs above |
| A5 | ✅ established | `lipid_baseline_feature_importance.csv` |
| A6 | ✅ established | `lipid_rdkit_only_baseline_metrics.csv` |
| B1 | ⏳ pending Colab | `chemberta_concat_baseline_lipid.ipynb` |
| B2 | ⏳ pending Colab + peptide CSVs | `chemberta_concat_baseline_peptide.ipynb` |
| B3 | ⏳ pending Colab | frozen branch of lipid notebook |
| B4 | ⏳ pending peptide CSVs | peptide notebook + new feature-importance script |
| C1 | not yet scoped | needs new transfer notebook |
| C2 | not yet scoped | optional ablation |
