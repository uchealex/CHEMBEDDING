"""5-fold cross-validation for the lipid RT baselines.

Pools the manuscript's fixed train/test partition (63,163 + 7,798 molecules)
into a single set of 70,961 molecules, then runs 5-fold CV (shuffled,
random_state=42) for the model conditions that can be evaluated on local
hardware:

  - RDKit-only baselines: Ridge, RandomForest, GradientBoosting
  - ChemBERTa-frozen-RT (CLS only): closed-form Ridge on cached embeddings
  - ChemBERTa-Concat (frozen): closed-form Ridge on cached [CLS || RDKit]

The fine-tuned variants are out of scope here -- under CV they multiply Colab
GPU-hours by k. The frozen closed-form fit is strictly an upper bound on the
notebook's 15-epoch AdamW frozen training, so the conclusions transfer.

Output: lipid_cv_baselines_metrics.csv, long-format with columns
(model, fold, split, metric, value).
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

DESC_COLS = [
    "mol_weight",
    "polar_surface_area",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "aromatic_rings",
    "heavy_atoms",
]
CACHE_DIR = ".cache_embeddings"
N_SPLITS = 5
KFOLD_SEED = 42


def get_pooled_embeddings():
    """Pool the cached train/test CLS embeddings. If the cache is missing,
    instruct the caller to run scripts/frozen_concat_lipid_local.py first."""
    train_path = os.path.join(CACHE_DIR, "lipid_train_cls.npy")
    test_path = os.path.join(CACHE_DIR, "lipid_test_cls.npy")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Missing cached embeddings under {CACHE_DIR}/. "
            "Run scripts/frozen_concat_lipid_local.py first to populate the cache."
        )
    return np.concatenate([np.load(train_path), np.load(test_path)], axis=0)


def main():
    train_df = pd.read_csv("METLIN_RT_lipid_train_with_rdkit.csv")
    test_df = pd.read_csv("METLIN_RT_lipid_test_with_rdkit.csv")
    pool_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"pooled: {len(pool_df)} molecules ({len(train_df)} train + {len(test_df)} test from manuscript split)")

    pool_emb = get_pooled_embeddings()
    assert len(pool_emb) == len(pool_df), (
        f"embedding count {len(pool_emb)} != pool size {len(pool_df)} -- cache stale?"
    )
    print(f"pool_emb: {pool_emb.shape}")

    pool_desc = pool_df[DESC_COLS].values.astype(float)
    pool_y = pool_df["rt"].values.astype(float)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=KFOLD_SEED)
    rows = []

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(pool_y)):
        t0 = time.time()

        # Per-fold scalers (no info leakage from test rows)
        desc_scaler = MinMaxScaler().fit(pool_desc[tr_idx])
        desc_tr = desc_scaler.transform(pool_desc[tr_idx]).astype(np.float32)
        desc_te = desc_scaler.transform(pool_desc[te_idx]).astype(np.float32)

        y_scaler = MinMaxScaler()
        y_tr_s = y_scaler.fit_transform(pool_y[tr_idx].reshape(-1, 1)).ravel()
        y_tr = pool_y[tr_idx]
        y_te = pool_y[te_idx]

        emb_tr = pool_emb[tr_idx]
        emb_te = pool_emb[te_idx]

        feature_views = [
            ("RDKit-only (Ridge)", desc_tr, desc_te, Ridge(alpha=1.0)),
            (
                "RDKit-only (RandomForest)",
                desc_tr,
                desc_te,
                RandomForestRegressor(n_estimators=200, random_state=fold_idx, n_jobs=-1),
            ),
            (
                "RDKit-only (GradientBoosting)",
                desc_tr,
                desc_te,
                GradientBoostingRegressor(n_estimators=200, random_state=fold_idx),
            ),
            ("ChemBERTa-frozen-RT (CLS only)", emb_tr, emb_te, Ridge(alpha=1.0)),
            (
                "ChemBERTa-Concat (frozen)",
                np.hstack([emb_tr, desc_tr]),
                np.hstack([emb_te, desc_te]),
                Ridge(alpha=1.0),
            ),
        ]

        for name, X_tr, X_te, mdl in feature_views:
            mdl.fit(X_tr, y_tr_s)
            for split_name, X, y_orig in [("Train", X_tr, y_tr), ("Test", X_te, y_te)]:
                y_pred_s = mdl.predict(X)
                y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                rows.append(
                    {
                        "model": name,
                        "fold": fold_idx,
                        "split": split_name,
                        "metric": "R2",
                        "value": float(r2_score(y_orig, y_pred)),
                    }
                )
                rows.append(
                    {
                        "model": name,
                        "fold": fold_idx,
                        "split": split_name,
                        "metric": "MAE",
                        "value": float(mean_absolute_error(y_orig, y_pred)),
                    }
                )

        print(f"  fold {fold_idx + 1}/{N_SPLITS} done in {time.time() - t0:.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv("lipid_cv_baselines_metrics.csv", index=False)
    print(f"\nwrote lipid_cv_baselines_metrics.csv ({len(out)} rows)")

    summary = (
        out[out["split"] == "Test"]
        .groupby(["model", "metric"])["value"]
        .agg(["median", "mean", "std", "min", "max"])
        .round(4)
    )
    print()
    print(summary)


if __name__ == "__main__":
    main()
