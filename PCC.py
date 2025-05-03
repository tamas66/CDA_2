import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype


# ───────────────────────── PCC core helper ──────────────────────────
def run_pcc_from_df(
    df: pd.DataFrame,
    y_col: str,
    max_pcs: int | None = None,
    cv_splits: int = 5,
    variance_threshold: float | None = None,
    scoring: str = "accuracy",
):
    """
    Principal-Component Classification on an *already prepared* DataFrame.
    """
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not found in DataFrame.")

    # Separate features and target
    X = df.drop(columns=[y_col]).select_dtypes(np.number).values
    y = df[y_col].values

    p = X.shape[1]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    if variance_threshold is not None:
        pipe.set_params(pca__n_components=variance_threshold)
        cv_score = cross_val_score(pipe, X, y, cv=cv, scoring=scoring).mean()
        pipe.fit(X, y)
        n_opt = pipe.named_steps["pca"].n_components_
    else:
        max_pcs = min(p, 15) if max_pcs is None else min(p, max_pcs)
        param_grid = {"pca__n_components": list(range(3, max_pcs + 1))}
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(X, y)
        cv_score = search.best_score_
        pipe = search.best_estimator_
        n_opt = pipe.named_steps["pca"].n_components_

    # Back-project coefficients
    pca = pipe.named_steps["pca"]
    clf = pipe.named_steps["clf"]
    L = pca.components_.T  # p × m
    beta_pc = clf.coef_    # (n_classes, m)

    # Handle binary and multi-class cases
    feature_names = df.drop(columns=[y_col]).select_dtypes(np.number).columns
    if len(clf.classes_) == 2:
        beta_pc = beta_pc.ravel()
        beta_orig = L @ beta_pc
        coef_series = pd.Series(beta_orig, index=feature_names
                           ).sort_values(key=np.abs, ascending=False)
    else:
        beta_orig = L @ beta_pc.T
        coef_series = pd.DataFrame(beta_orig, index=feature_names, columns=clf.classes_)

    # Report results
    print(f"\nPCC CV {scoring}: {cv_score:.3f}")
    print(f"Variance kept: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"PCs kept    : {n_opt} out of {p}")
    if cv_score > 0.75:
        if isinstance(coef_series, pd.Series):
            print("\nTop 10 coefficients (original variables):")
            print(coef_series.head(10).round(3))
        else:
            print("\nTop coefficients per class:")
            for cls in coef_series.columns:
                sorted_coef = coef_series[cls].sort_values(key=np.abs, ascending=False)
                print(f"\nClass {cls} top 10 coefficients:")
                print(sorted_coef.head(10).round(3))

    return cv_score


def try_pcc(df, y_col, var_thr=0.70, scoring="accuracy", min_samples=5):
    """Run PCC on y_col if suitable; return CV score or NaN."""
    df = df.copy()
    print(f"\n====  {y_col}  ====")
    # Handle missing values
    initial_count = len(df)
    df = df.dropna(subset=[y_col])
    if initial_count != len(df):
        print(f"{y_col:<12} | Dropped {initial_count - len(df)} missing values")

    # Handle numeric conversion
    if np.issubdtype(df[y_col].dtype, np.number):
        y_vals = df[y_col]
        min_val = int(y_vals.min())
        max_val = int(y_vals.max())
        
        # Preserve original scale for binary (0/1) variables
        if max_val == 1 and min_val == 0:
            scale_bounds = (0, 1)
        else:
            scale_bounds = (min_val, max_val)
        
        y_processed = np.clip(y_vals, *scale_bounds).astype(int)
        
        # Check if we have valid categories after processing
        unique_vals = np.unique(y_processed)
        if len(unique_vals) < 2:
            print(f"{y_col:<12} | Insufficient unique values after processing - skipping")
            return np.nan
            
        # Convert to ordered categorical
        df[y_col] = pd.Categorical(
            y_processed,
            categories=np.arange(scale_bounds[0], scale_bounds[1]+1),
            ordered=True
        )
        print(f"{y_col:<12} | Converted to {len(unique_vals)}-class ordered categorical")

    # Check class balance
    class_counts = df[y_col].value_counts()
    if len(class_counts) < 2:
        print(f"{y_col:<12} | Needs ≥2 classes - skipping")
        return np.nan
        
    if class_counts.min() < min_samples:
        print(f"{y_col:<12} | Small classes detected - applying class weights")
    
    try:

        return run_pcc_from_df(
            df,
            y_col=y_col,
            variance_threshold=var_thr,
            scoring=scoring
        )
    except Exception as e:
        print(f"{y_col:<12} | Failed ({str(e)}")
        return np.nan


# ──────────────────────────── MAIN loop ─────────────────────────────
if __name__ == "__main__":
    from initial_data import load_data, preprocess_data

    df = preprocess_data(load_data("HR_data.csv"))

    scoring = "accuracy"
    var_thr = 0.70

    target_cols = [
        "Individual","Puzzler","Frustrated",
        "upset","hostile","alert","ashamed","inspired","nervous",
        "attentive","afraid","active","determined"
    ]

    results = {}
    for y in target_cols:
        score = try_pcc(df, y, var_thr=var_thr, scoring=scoring)
        results[y] = score

    # Summary table
    res_series = pd.Series(results).dropna().sort_values(ascending=False)
    print("\n===  PCC cross-validated accuracy (70% variance PCs)  ===")
    print(res_series.round(3).to_string())
    # Visaulization
    plt.figure(figsize=(10, 6))
    plt.plot(res_series, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"PCC cross-validated {scoring} ({var_thr*100:.0f}% variance PCs)")
    plt.xlabel("Response variable")
    plt.ylabel(f"CV-{scoring}")
    plt.tight_layout()
    plt.show()