"""Feature importance analysis for the RDKit-only lipid RT baseline.

Fits LinearRegression, Ridge, and RandomForest on the seven RDKit descriptors,
then reports per-feature contributions:

  - Linear / Ridge coefficients on MinMax-scaled features. Because every
    descriptor lives in [0, 1] after scaling, the raw coefficient magnitude
    is directly comparable across features (it is the change in scaled RT
    that comes from sweeping that descriptor across its full observed range).
  - RandomForest feature_importances_ for a nonlinear reference.
  - Univariate Pearson correlation of each descriptor with RT, as a sanity
    anchor that does not depend on any model.

Prints a tidy table and writes lipid_baseline_feature_importance.csv.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
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


def main():
    train_df = pd.read_csv("METLIN_RT_lipid_train_with_rdkit.csv")
    test_df = pd.read_csv("METLIN_RT_lipid_test_with_rdkit.csv")

    X_train = train_df[DESC_COLS].values.astype(float)
    X_test = test_df[DESC_COLS].values.astype(float)
    y_train = train_df["rt"].values.astype(float)
    y_test = test_df["rt"].values.astype(float)

    x_scaler = MinMaxScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Univariate correlation reference (Pearson, on raw / unscaled values)
    corrs = {col: float(np.corrcoef(train_df[col], train_df["rt"])[0, 1]) for col in DESC_COLS}

    rows = []

    # OLS / Ridge — coefficients on scaled features
    for name, mdl in [("LinearRegression", LinearRegression()), ("Ridge(alpha=1.0)", Ridge(alpha=1.0))]:
        mdl.fit(X_train_s, y_train_s)
        y_pred_s_train = mdl.predict(X_train_s)
        y_pred_s_test = mdl.predict(X_test_s)
        y_pred_train = y_scaler.inverse_transform(y_pred_s_train.reshape(-1, 1)).ravel()
        y_pred_test = y_scaler.inverse_transform(y_pred_s_test.reshape(-1, 1)).ravel()
        print(f"\n=== {name} ===")
        print(
            f"  R2:  train={r2_score(y_train, y_pred_train):.4f}  test={r2_score(y_test, y_pred_test):.4f}"
        )
        print(
            f"  MAE: train={mean_absolute_error(y_train, y_pred_train):.2f}  "
            f"test={mean_absolute_error(y_test, y_pred_test):.2f}"
        )
        print(f"  intercept (scaled RT): {mdl.intercept_:.4f}")
        coef_table = (
            pd.DataFrame(
                {
                    "feature": DESC_COLS,
                    "coef": mdl.coef_,
                    "abs_coef": np.abs(mdl.coef_),
                    "pearson_r_with_RT": [corrs[c] for c in DESC_COLS],
                }
            )
            .sort_values("abs_coef", ascending=False)
            .reset_index(drop=True)
        )
        print(coef_table.round(4).to_string(index=False))
        for _, r in coef_table.iterrows():
            rows.append(
                {
                    "model": name,
                    "feature": r["feature"],
                    "score_kind": "coef",
                    "value": r["coef"],
                }
            )

    # RandomForest — built-in feature_importances_
    rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X_train_s, y_train_s)
    y_pred_test_rf = y_scaler.inverse_transform(rf.predict(X_test_s).reshape(-1, 1)).ravel()
    print("\n=== RandomForest(n_estimators=200, random_state=0) ===")
    print(f"  R2:  test={r2_score(y_test, y_pred_test_rf):.4f}")
    print(f"  MAE: test={mean_absolute_error(y_test, y_pred_test_rf):.2f}")
    rf_table = (
        pd.DataFrame(
            {
                "feature": DESC_COLS,
                "importance": rf.feature_importances_,
                "pearson_r_with_RT": [corrs[c] for c in DESC_COLS],
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(rf_table.round(4).to_string(index=False))
    for _, r in rf_table.iterrows():
        rows.append(
            {
                "model": "RandomForest",
                "feature": r["feature"],
                "score_kind": "importance",
                "value": r["importance"],
            }
        )

    # Univariate as an anchor row
    for col, r in corrs.items():
        rows.append(
            {
                "model": "Univariate",
                "feature": col,
                "score_kind": "pearson_r",
                "value": r,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv("lipid_baseline_feature_importance.csv", index=False)
    print("\nwrote lipid_baseline_feature_importance.csv")


if __name__ == "__main__":
    main()
