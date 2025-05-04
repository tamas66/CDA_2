import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import matplotlib.pyplot as plt


# ───────────────────────── PCR core helper ──────────────────────────
def run_pcr_from_df(
    df: pd.DataFrame,
    y_col: str,
    max_pcs: int | None = None,
    cv_splits: int = 5,
    variance_threshold: float | None = None,
):
    """
    Principal-Component Regression on an *already prepared* DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the response column `y_col` and numeric predictors.
    y_col : str
        Response column name.
    max_pcs : int or None
        Max number of PCs to test (grid search).  Ignored if
        `variance_threshold` is given.  Default: min(p, 15).
    cv_splits : int
        Folds for CV.
    variance_threshold : float or None
        If set (e.g. 0.90), keep enough PCs to reach that fraction of
        explained variance and skip grid search.
    """

    if y_col not in df.columns:
        raise KeyError(f"{y_col} not found in DataFrame.")

    X = df.select_dtypes(np.number).drop(columns=[y_col]).values
    y = df[y_col].values
    p = X.shape[1]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca",   PCA()),
        ("ols",   LinearRegression())
    ])

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # ---- choose n_components ---------------------------------------
    if variance_threshold is not None:
        pipe.set_params(pca__n_components=variance_threshold)
        cv_r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean()
        pipe.fit(X, y)
        n_opt = pipe.named_steps["pca"].n_components_
    else:
        max_pcs = min(p, 15) if max_pcs is None else min(p, max_pcs)
        param_grid = {"pca__n_components": list(range(3, max_pcs + 1))}
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", n_jobs=-1)
        search.fit(X, y)
        cv_r2 = search.best_score_
        pipe = search.best_estimator_
        n_opt = pipe.named_steps["pca"].n_components_

    # ---- back-project coefficients ---------------------------------
    L = pipe.named_steps["pca"].components_.T                # p × m
    beta_pc = pipe.named_steps["ols"].coef_                  # length m
    beta_orig = L @ beta_pc                                  # length p

    coef_series = pd.Series(beta_orig,
                            index=df.select_dtypes(np.number).drop(columns=[y_col]).columns
                           ).sort_values(key=np.abs, ascending=False)

    # ---- report ----------------------------------------------------
    print(f"\nPCR CV R²  : {cv_r2:.3f}")
    print(f"PCs kept   : {n_opt}")
    print("\nTop 10 coefficients (original variables):")
    print(coef_series.head(10).round(3))

    return cv_r2

# ────────────────── helper: safe PCR + score table ──────────────────
def try_pcr(df, y_col, var_thr=0.70):
    """Run PCR on y_col if it is numeric; return CV-R2 or NaN."""
    if not np.issubdtype(df[y_col].dtype, np.number):
        print(f"{y_col:<12s} skipped (non-numeric)")
        return np.nan
    try:
        X = df.select_dtypes(np.number).drop(columns=[y_col])
        if X.shape[1] < 3:
            print(f"{y_col:<12s} skipped (not enough predictors)")
            return np.nan
        print(f"\n====  {y_col}  ====")
        r2 = run_pcr_from_df(df, y_col=y_col, variance_threshold=var_thr)
        return r2
    except Exception as e:
        print(f"{y_col:<12s} failed ({e})")
        return np.nan


# ──────────────────────────── MAIN loop ─────────────────────────────
if __name__ == "__main__":
    from initial_data import load_data, preprocess_data

    df = preprocess_data(load_data("HR_data.csv"))

    target_cols = [
        "Individual","Puzzler","Frustrated",
        "upset","hostile","alert","ashamed","inspired","nervous",
        "attentive","afraid","active","determined"
    ]

    results = {}
    for y in target_cols:
        r2 = try_pcr(df, y, var_thr=0.70)   # keep 70 % variance
        results[y] = r2

    # -------- summary table sorted by CV-R² ----------
    res_series = pd.Series(results).dropna().sort_values(ascending=False)
    print("\n===  PCR cross-validated R² (70 % variance PCs)  ===")
    print(res_series.round(3).to_string())

    plt.plot(res_series, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("PCR cross-validated R² (70% variance PCs)")
    plt.xlabel("Response variable")
    plt.ylabel("CV-R²")
    plt.tight_layout()
    plt.show()

