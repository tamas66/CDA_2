from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from initial_data import load_data

def run_pc_logr_from_df(
    df: pd.DataFrame,
    y_col: str,
    max_pcs: Optional[int] = None,
    cv_splits: int = 5,
    variance_threshold: Optional[float] = None,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    with_pca: bool = True
):    
    # ensure only numerical features for PCA
    physio_cols = [c for c in df.columns if c.startswith(("HR_", "EDA_", "TEMP_"))]
    X = df[physio_cols].values

    y = df[y_col]
    y = preprocess_target(y)
    
    # create the pipeline for Logistic Regression with PCA (if with_pca=True)
    if with_pca:
        pipe = Pipeline([
            ("scale", StandardScaler()),  # standardize the data
            ("pca", PCA()),  # PCA for dimensionality reduction
            ("logreg", LogisticRegression(max_iter=max_iter, solver=solver, multi_class='auto'))
        ])
    else:
        pipe = Pipeline([
            ("scale", StandardScaler()),  # standardize the data
            ("logreg", LogisticRegression(max_iter=max_iter, solver=solver, multi_class='auto'))
        ])

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # choose the number of components for PCA if needed
    if with_pca:
        if variance_threshold is not None:
            pipe.set_params(pca__n_components=variance_threshold)
            pipe.fit(X, y)
            n_opt = pipe.named_steps["pca"].n_components_
        else:
            max_pcs = min(X.shape[1], 15) if max_pcs is None else min(X.shape[1], max_pcs)
            param_grid = {"pca__n_components": list(range(3, max_pcs + 1))}
            search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
            search.fit(X, y)
            pipe = search.best_estimator_
            n_opt = pipe.named_steps["pca"].n_components_
        
        print(f"\nPCLogR CV accuracy with PCA : {pipe.score(X, y):.3f}")
        print(f"PCs kept with PCA           : {n_opt}")
    else:
        pipe.fit(X, y)
        print(f"\nLogistic Regression CV accuracy without PCA : {pipe.score(X, y):.3f}")

    if with_pca:
        # back‐project to original features
        L = pipe.named_steps["pca"].components_.T # (p × m)
        beta_pc = pipe.named_steps["logreg"].coef_ # (C × m) or (1 × m)
        
        if beta_pc.shape[0] == 1:
            # binary: one row only
            beta_orig = L @ beta_pc.ravel() # shape (p,)
            agg = pd.Series(np.abs(beta_orig), index=physio_cols)
        else:
            # multiclass: back‐project each class, then sum across classes
            beta_orig = L @ beta_pc.T # shape (p × C)
            df_imp = pd.DataFrame(beta_orig,
                                  index=physio_cols,
                                  columns=[f"class_{c}" for c in pipe.named_steps["logreg"].classes_])
            agg = df_imp.abs().sum(axis=1)

        # pick top 10
        top10 = agg.nlargest(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top10.values, y=top10.index, palette="coolwarm")
        plt.title(f"Top 10 contributing features")
        plt.xlabel('Sum of absolute back-projected coefficients')
        plt.ylabel('Feature')
        plt.show()

    else:
        betas = pipe.named_steps["logreg"].coef_[0]
        imp = pd.Series(np.abs(betas), index=physio_cols).nlargest(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=imp.values, y=imp.index, palette="coolwarm")
        plt.title("Top 10 contributing features (No PCA)")
        plt.xlabel('Absolute coefficient value')
        plt.ylabel('Feature')
        plt.show()
    return pipe.score(X, y)


def preprocess_data(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    target = df[target_col] if target_col else None
    df = df.drop(columns=[target_col]) if target_col else df
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    if target_col:
        df[target_col] = target

    return df

def preprocess_target(y):
    # check if the target has continuous values
    print(f"Unique values in target: {y.unique()}")   
    # if there is a continuous value (non-integer), round the values to the nearest integer
    y = y.round().astype(int)    
    # convert to categorical type if it isn't already
    if y.dtype != 'category':
        y = y.astype('category')
    
    print(f"After processing, unique values in target: {y.unique()}")
    return y

if __name__ == "__main__":
    file_path = 'HR_data_2.csv'
    df = load_data(file_path)
                    
    df = df.drop(columns=['Unnamed: 0'])
    # preprocess the data -> imputing missing values
    df = preprocess_data(df)

    target_cols = ["hostile", "Puzzler", "Frustrated",
                  "upset", "alert", "ashamed", "inspired", "nervous",
                   "attentive", "afraid", "active", "determined"]

    results = []  # will hold (emotion, acc_with_pca, acc_without_pca)

    for y_col in target_cols:
        if df[y_col].dtype == 'object' or df[y_col].nunique() <= 10:
            print(f"\n==== {y_col} ====")
            a = run_pc_logr_from_df(df, y_col, variance_threshold=0.70, with_pca=True)
            b = run_pc_logr_from_df(df, y_col, variance_threshold=0.70, with_pca=False)
            results.append((y_col, a, b))

    filtered = [(emo,a,b) for emo,a,b in results if a is not None and b is not None]

    if not filtered:
        print("No valid results to plot")
    else:
        emotions, accs_pca, accs_no = zip(*filtered)
        x = np.arange(len(emotions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(x - width/2, accs_pca, width, label='With PCA', color='skyblue')
        ax.bar(x + width/2, accs_no, width, label='Without PCA', color='salmon')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=90)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model accuracy for each emotion (With and without PCA)')
        ax.legend()
        plt.tight_layout()
        plt.show()


