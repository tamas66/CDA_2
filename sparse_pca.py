import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ───────────────────────── 1.  HELPER FUNCTIONS ─────────────────────────
def varimax(Phi, gamma=1, q=50, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vt
        d = s.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return Phi @ R


def plot_cos2_heatmap(
    loadings_mat,
    feature_names,
    comp_names,
    max_pc: int = 6,
    variables_per_page: int = 25,
    cmap: str = "viridis",
    title_prefix: str = "",
):
    """Paginated Cos² heat-map."""
    if isinstance(loadings_mat, np.ndarray):
        L = pd.DataFrame(loadings_mat, index=feature_names, columns=comp_names)
    else:
        L = loadings_mat.copy()

    pcs = L.columns[:max_pc]
    cos2 = (L[pcs] ** 2).div((L[pcs] ** 2).sum(axis=1), axis=0)

    total_vars = cos2.shape[0]
    n_pages = math.ceil(total_vars / variables_per_page)
    n_cols = 2
    n_rows = math.ceil(n_pages / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i in range(n_pages):
        start, end = i * variables_per_page, min((i + 1) * variables_per_page, total_vars)
        sns.heatmap(
            cos2.iloc[start:end],
            annot=True, fmt=".2f", cmap=cmap, linewidths=0.5,
            cbar_kws={"label": "Cos²"}, ax=axes[i]
        )
        axes[i].set_title(f"{title_prefix} Cos²: variables {start + 1}–{end}")
        axes[i].set_xlabel("Components")
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=8)

    for j in range(n_pages, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


# ───────────────────────── 2.  SPARSE-PCA COMPUTATION ───────────────────
def compute_sparse_pca(
    df: pd.DataFrame,
    n_variance: float = 0.9,
    threshold: float = 0.15,
    n_nonzero: int = 10,
    l1_ratio: float = 0.05,
):
    """
    Returns dictionaries with loadings (L) and scores (S)
    for dense PCA, thresholded PCA, varimax and elastic-net SPCA.
    No plots are produced here.
    """
    if not (0 < n_variance <= 1):
        raise ValueError("n_variance must be in (0,1].")

    X_std = StandardScaler().fit_transform(df.select_dtypes(np.number))
    pca = PCA(n_components=n_variance).fit(X_std)
    k = pca.n_components_

    # dense
    L_dense = pca.components_.T
    S_dense = pca.transform(X_std)

    # threshold
    L_thr = L_dense.copy()
    L_thr[np.abs(L_thr) < threshold] = 0
    L_thr /= np.where(np.linalg.norm(L_thr, axis=0) == 0, 1, np.linalg.norm(L_thr, axis=0))
    S_thr = X_std @ L_thr

    # varimax
    L_var = varimax(L_thr)
    L_var /= np.linalg.norm(L_var, axis=0, keepdims=True)
    S_var = X_std @ L_var

    # elastic-net
    alpha_grid = np.logspace(-4, 2, 30)

    def choose_alpha(X, y):
        for a in alpha_grid:
            coef = ElasticNet(alpha=a, l1_ratio=l1_ratio,
                              fit_intercept=False, max_iter=10000).fit(X, y).coef_
            if (np.abs(coef) > 1e-6).sum() <= n_nonzero:
                return a
        return alpha_grid[-1]

    L_en = np.zeros_like(L_dense)
    for j in range(k):
        a = choose_alpha(X_std, S_dense[:, j])
        coef = ElasticNet(alpha=a, l1_ratio=l1_ratio,
                          fit_intercept=False, max_iter=10000).fit(X_std, S_dense[:, j]).coef_
        if np.any(coef):
            L_en[:, j] = coef / np.linalg.norm(coef)
    S_en = X_std @ L_en

    return {
        "dense": {"L": L_dense, "S": S_dense},
        "thr":   {"L": L_thr,   "S": S_thr},
        "var":   {"L": L_var,   "S": S_var},
        "en":    {"L": L_en,    "S": S_en},
        "k":     k,                           # keep for plotting utilities
    }


# ───────────────────────── 3.  DIAGNOSTIC PLOTS ─────────────────────────
def plot_sparse_pca_diagnostics(result_dict: dict, df: pd.DataFrame):
    """Variance bars, correlation heat-maps & Cos² heat-maps for each flavour."""
    k = result_dict["k"]
    var_dense = np.var(result_dict["dense"]["S"], axis=0, ddof=1)
    feat_names = df.select_dtypes(np.number).columns
    comp_names = [f"PC{i+1}" for i in range(k)]

    def triple(title, S_mat, L_mat):
        var = np.var(S_mat, axis=0, ddof=1)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].bar(range(1, k + 1), var_dense, alpha=0.5, label="dense")
        ax[0].bar(range(1, k + 1), var, alpha=0.8, label=title)
        ax[0].set_xlabel("Component"); ax[0].set_ylabel("Variance λ")
        ax[0].set_title(f"Variance per component ({title})"); ax[0].legend()

        sns.heatmap(np.abs(np.corrcoef(S_mat.T)), cmap="viridis",
                    cbar=False, square=True, ax=ax[1])
        ax[1].set_title(f"|corr| {title} scores")

        sns.heatmap(np.abs(np.corrcoef(L_mat.T)), cmap="viridis",
                    cbar=False, square=True, ax=ax[2])
        ax[2].set_title(f"|corr| {title} loadings")
        plt.tight_layout(); plt.show()

        plot_cos2_heatmap(L_mat, feat_names, comp_names,
                          max_pc=6, title_prefix=title)

    triple("threshold", result_dict["thr"]["S"], result_dict["thr"]["L"])
    triple("varimax",   result_dict["var"]["S"], result_dict["var"]["L"])
    triple("elastic-net", result_dict["en"]["S"], result_dict["en"]["L"])


# ───────────────────────── 4.  CLUSTERING UTILITY ───────────────────────
def cluster_sparse_scores(
    score_dict: dict,
    meta_labels: np.ndarray,
    k_range: range = range(2, 11),
    n_dims: int = 5,
):
    """K-means + silhouette (and ARI if labels provided) for each sparse flavour."""
    print("\n===  K-means clustering in sparse subspaces  ===")
    for tag in ["thr", "var", "en"]:
        S = score_dict[tag]["S"][:, :n_dims]
        best_k, best_sil = None, -1
        for k in k_range:
            km = KMeans(k, random_state=0, n_init="auto").fit(S)
            sil = silhouette_score(S, km.labels_)
            if sil > best_sil:
                best_k, best_sil, best_lab = k, sil, km.labels_
        msg = f"{tag:10s}  best k={best_k}  silhouette={best_sil: .3f}"
        if meta_labels is not None:
            ari = adjusted_rand_score(meta_labels, best_lab)
            msg += f"   ARI vs labels = {ari: .3f}"
        print(msg)

        if S.shape[1] >= 2:  # scatter of first two dims
            plt.figure(figsize=(16, 4))
            sns.scatterplot(x=S[:, 0], y=S[:, 1],
                            hue=best_lab, palette="tab10",
                            style=meta_labels if meta_labels is not None else None,
                            s=60)
            plt.title(f"{tag} – KMeans k={best_k}")
            plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.tight_layout(); plt.show()

# ─────────────────────────  ROLE-SUBSPACE ANALYSIS ──────────────────────
def role_subspace_similarity(
    df: pd.DataFrame,
    role_col: str = "Puzzler",       # 1 = puzzler, 0 = instructor
    flavour: str = "en",             # "thr" | "var" | "en"
    n_variance: float = 0.9,
    threshold: float = 0.15,
):
    """
    Compare sparse-PCA loadings for Puzzlers vs Instructors.

    Returns
    -------
    L_A : ndarray (p × kA)
    L_B : ndarray (p × kB)
    cos_mat : ndarray (kA × kB) absolute cosine similarities.
    """
    if flavour not in {"thr", "var", "en"}:
        raise ValueError("flavour must be 'thr', 'var' or 'en'.")

    # --- split --------------------------------------------------------
    df_A = df[df[role_col] == 0]     # instructors
    df_B = df[df[role_col] == 1]     # puzzlers

    res_A = compute_sparse_pca(df_A, n_variance, threshold)
    res_B = compute_sparse_pca(df_B, n_variance, threshold)

    L_A = res_A[flavour]["L"];  kA = L_A.shape[1]
    L_B = res_B[flavour]["L"];  kB = L_B.shape[1]

    # unit-norm (already orthonormal, but in case sparsity killed norms)
    L_A = L_A / np.linalg.norm(L_A, axis=0, keepdims=True)
    L_B = L_B / np.linalg.norm(L_B, axis=0, keepdims=True)

    # absolute cosine similarity matrix
    cos_mat = np.abs(L_A.T @ L_B)

    # --- print summary ----------------------------------------------
    print(f"\nSparse-PCA flavour: {flavour}")
    print(" | Instructor PCs | Puzzler PCs |  |cos|")
    for i in range(min(kA, kB)):
        print(f"   PC{i+1:<2d}            PC{i+1:<2d}        {cos_mat[i,i]:.3f}")

    # --- heat-map ----------------------------------------------------
    plt.figure(figsize=(16, 16))
    sns.heatmap(cos_mat, annot=True, fmt=".2f", cmap="magma",
                xticklabels=[f"P{i+1}" for i in range(kB)],
                yticklabels=[f"I{i+1}" for i in range(kA)])
    plt.title(f"|cosine| between Instructor and Puzzler loadings "
              f"({flavour})")
    plt.xlabel("Puzzler PCs"); plt.ylabel("Instructor PCs")
    plt.tight_layout(); plt.show()

    return L_A, L_B, cos_mat



# ───────────────────────── 5.  EXAMPLE USAGE ────────────────────────────
if __name__ == "__main__":
    from initial_data import load_data, preprocess_data

    df = load_data("HR_data.csv")
    df = preprocess_data(df)

    # 1) compute sparse-PCA flavours
    #spca_results = compute_sparse_pca(df, n_variance=0.9, threshold=0.15)

    # 2) variance bars of sparse pca method vs dense pca, correlation heat-maps & Cos² heat-maps
    #plot_sparse_pca_diagnostics(spca_results, df)

    # 3) K means clustering + silhouette score
    #phase_labels = df["phase"].values if "phase" in df else None
    #cluster_sparse_scores(spca_results, meta_labels=phase_labels)

    # Team role inspection using sparse PCA
    # compare the sparse PCA methods sub-spaces of the two roles
    # L_I, L_P, C = role_subspace_similarity(
    #     df,
    #     role_col="Puzzler",
    #     flavour="en",      #"thr", "var" or "en"
    #     n_variance=0.9,
    #     threshold=0.15,
    # )
    

