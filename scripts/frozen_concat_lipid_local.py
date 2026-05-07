"""Frozen ChemBERTa-Concat baseline run on the local METLIN lipid CSVs.

Closed-form analog of the 'ChemBERTa-Concat (frozen)' variant in
chemberta_concat_baseline_lipid.ipynb: forward-pass ChemBERTa once, then fit a
linear head in closed form. Strictly an upper bound on what the notebook's
15-epoch AdamW frozen training could converge to, since the convex MSE
objective is solved exactly here.

Runs on Apple Silicon MPS or CPU (no NVIDIA GPU required). Embedding extraction
is the bottleneck and is cached under .cache_embeddings/ so reruns are fast.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer

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
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def extract_cls_embeddings(smiles, tokenizer, bert, device, log_every_batches=50):
    bert.eval()
    embs = []
    for i in range(0, len(smiles), BATCH_SIZE):
        batch = smiles[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        out = bert(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embs.append(cls)
        bidx = i // BATCH_SIZE
        if bidx % log_every_batches == 0:
            print(f"    {i:>6d}/{len(smiles)}")
    return np.concatenate(embs, axis=0)


def get_or_compute_embeddings(train_smiles, test_smiles):
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_path = os.path.join(CACHE_DIR, "lipid_train_cls.npy")
    test_path = os.path.join(CACHE_DIR, "lipid_test_cls.npy")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("loading cached embeddings...")
        return np.load(train_path), np.load(test_path)

    device = get_device()
    print(f"device: {device}")
    print("loading ChemBERTa...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).to(device)

    t0 = time.time()
    print(f"extracting train CLS embeddings ({len(train_smiles)} molecules)...")
    train_emb = extract_cls_embeddings(train_smiles, tokenizer, bert, device)
    print(f"  done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print(f"extracting test CLS embeddings ({len(test_smiles)} molecules)...")
    test_emb = extract_cls_embeddings(test_smiles, tokenizer, bert, device)
    print(f"  done in {time.time() - t0:.1f}s")

    np.save(train_path, train_emb)
    np.save(test_path, test_emb)
    return train_emb, test_emb


def main():
    train_df = pd.read_csv("METLIN_RT_lipid_train_with_rdkit.csv")
    test_df = pd.read_csv("METLIN_RT_lipid_test_with_rdkit.csv")
    print(f"train: {len(train_df)}   test: {len(test_df)}")

    train_emb, test_emb = get_or_compute_embeddings(
        train_df["smile"].tolist(), test_df["smile"].tolist()
    )
    print(f"train_emb: {train_emb.shape}   test_emb: {test_emb.shape}")

    desc_scaler = MinMaxScaler().fit(train_df[DESC_COLS].values)
    desc_train = desc_scaler.transform(train_df[DESC_COLS].values).astype(np.float32)
    desc_test = desc_scaler.transform(test_df[DESC_COLS].values).astype(np.float32)

    y_scaler = MinMaxScaler()
    y_train_s = y_scaler.fit_transform(train_df[["rt"]].values).ravel()
    y_train = train_df["rt"].values
    y_test = test_df["rt"].values

    feature_sets = [
        ("ChemBERTa-frozen-RT (CLS only)", train_emb, test_emb),
        (
            "ChemBERTa-Concat (frozen, closed-form)",
            np.hstack([train_emb, desc_train]),
            np.hstack([test_emb, desc_test]),
        ),
    ]
    head_options = [
        ("OLS", LinearRegression),
        ("Ridge1", lambda: Ridge(alpha=1.0)),
        ("Ridge100", lambda: Ridge(alpha=100.0)),
    ]

    rows = []
    for feat_name, X_tr, X_te in feature_sets:
        for head_name, head_factory in head_options:
            mdl = head_factory()
            mdl.fit(X_tr, y_train_s)
            for split_name, X, y_orig in [("Train", X_tr, y_train), ("Test", X_te, y_test)]:
                y_pred_s = mdl.predict(X)
                y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                rows.append(
                    {
                        "model": f"{feat_name} + {head_name}",
                        "seed": 0,
                        "split": split_name,
                        "metric": "R2",
                        "value": r2_score(y_orig, y_pred),
                    }
                )
                rows.append(
                    {
                        "model": f"{feat_name} + {head_name}",
                        "seed": 0,
                        "split": split_name,
                        "metric": "MAE",
                        "value": mean_absolute_error(y_orig, y_pred),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv("lipid_chemberta_frozen_metrics.csv", index=False)

    summary = out.pivot_table(
        index="model", columns=["split", "metric"], values="value"
    ).round(4)
    print()
    print(summary)


if __name__ == "__main__":
    main()
