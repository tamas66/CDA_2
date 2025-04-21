from initial_data import load_data, inspect_data, preprocess_data, visualize_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet


# ─────────────────────────── Varimax helper ────────────────────────────
def varimax(Phi, gamma=1, q=50, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T
            @ (
                Lambda**3
                - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))
            )
        )
        R = u @ vt
        d = s.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return Phi @ R


# ───────────────────────────── Main routine ────────────────────────────
def SPCA(
    df: pd.DataFrame,
    n_variance: float,
    threshold: float,
) -> None:
    """
    Dense PCA ➔ thresholded loadings ➔ varimax ➔ elastic‑net SPCA,
    with diagnostic variance & correlation plots.
    """

    # ── Dense PCA ──────────────────────────────────────────────────────
    if not (0 < n_variance <= 1):
        raise ValueError("n_variance must lie in (0, 1].")

    X_std = StandardScaler().fit_transform(df.select_dtypes(np.number))
    pca = PCA(n_components=n_variance).fit(X_std)
    k = pca.n_components_
    L_dense = pca.components_.T
    S_dense = pca.transform(X_std)
    var_dense = pca.explained_variance_

    print(f"Number of components selected: {k}")

    # ── Hard‑thresholding ──────────────────────────────────────────────
    L_thr = L_dense.copy()
    L_thr[np.abs(L_thr) < threshold] = 0
    norms = np.linalg.norm(L_thr, axis=0)
    norms[norms == 0] = 1.0
    L_thr /= norms

    S_thr = X_std @ L_thr
    var_thr = S_thr.var(axis=0, ddof=1)

    # ── Varimax rotation on thresholded loadings ───────────────────────
    L_var = varimax(L_thr)
    L_var /= np.linalg.norm(L_var, axis=0, keepdims=True)
    S_var = X_std @ L_var
    var_var = S_var.var(axis=0, ddof=1)

    # ── Elastic‑net SPCA ───────────────────────────────────────────────
    n_nonzero = 10           # target NNZ per component
    l1_ratio = 0.05
    alpha_grid = np.logspace(-4, 2, 30)

    def _choose_alpha(X, y):
        for a in alpha_grid:
            coef = ElasticNet(alpha=a, l1_ratio=l1_ratio,
                              fit_intercept=False, max_iter=10000).fit(X, y).coef_
            if (np.abs(coef) > 1e-6).sum() <= n_nonzero:
                return a
        return alpha_grid[-1]

    alphas = [_choose_alpha(X_std, S_dense[:, j]) for j in range(k)]
    L_en = np.zeros_like(L_dense)
    for j, a in enumerate(alphas):
        en = ElasticNet(alpha=a, l1_ratio=l1_ratio,
                        fit_intercept=False, max_iter=10000)
        coef = en.fit(X_std, S_dense[:, j]).coef_
        if np.all(coef == 0):
            continue
        L_en[:, j] = coef / np.linalg.norm(coef)

    S_en = X_std @ L_en
    var_en = S_en.var(axis=0, ddof=1)

    # ── Variance tables ────────────────────────────────────────────────
    def _print(tag, arr):
        print(f"\nVariance per PC (dense → {tag}):")
        for i in range(k):
            print(f"  PC{i+1}: {var_dense[i]:.3f} → {arr[i]:.3f}")

    _print("threshold",  var_thr)
    _print("varimax",    var_var)
    _print("elastic‑net", var_en)

    # ── Plot helper ────────────────────────────────────────────────────
    def _triple(title, arr, S_mat, L_mat):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].bar(range(1, k + 1), var_dense, alpha=0.5, label="dense")
        ax[0].bar(range(1, k + 1), arr,       alpha=0.8, label=title)
        ax[0].set_xlabel("Component"); ax[0].set_ylabel("Variance λ")
        ax[0].set_title(f"Variance per component ({title})")
        ax[0].legend()

        sns.heatmap(np.abs(np.corrcoef(S_mat.T)), cmap="viridis",
                    cbar=False, square=True, ax=ax[1])
        ax[1].set_title(f"|corr| {title} scores")

        sns.heatmap(np.abs(np.corrcoef(L_mat.T)), cmap="viridis",
                    cbar=False, square=True, ax=ax[2])
        ax[2].set_title(f"|corr| {title} loadings")

        plt.tight_layout(); plt.show()

    # ── Figures ────────────────────────────────────────────────────────
    _triple("threshold",   var_thr, S_thr, L_thr)
    _triple("varimax",     var_var, S_var, L_var)
    _triple("elastic‑net", var_en,  S_en,  L_en)



    

if __name__ == "__main__":
    # Load the data
    df = load_data("HR_data.csv")

    # Preprocess the data
    df = preprocess_data(df)

    # Perform Sparse PCA with thresholding and varimax rotation and Elastic net
    SPCA(df, n_variance=0.9, threshold=0.15)


