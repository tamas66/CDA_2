import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from py_pcha import PCHA
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

def plot_pcha_sse(X_std, k_min=2, k_max=8, delta=0.1, figsize=(6,4)):
    sse = []
    for k in range(k_min, k_max):
        _, _, _, SSE_k, _ = PCHA(X_std.T, noc=k, delta=delta)
        sse.append((k, SSE_k))

    ks, errs = zip(*sse)

    plt.figure(figsize=figsize)
    plt.plot(ks, errs, marker='o')
    plt.xlabel('Number of archetypes (k)')
    plt.ylabel('Sum of squared errors (SSE)')
    plt.title(f'Elbow plot (delta={delta})')
    plt.xticks(ks)
    plt.tight_layout()
    plt.show()

    return list(ks), list(errs)

# determine number of components from CV -> suggested during lectures
# function to project new data onto archetypes (XC)
def project_to_archetypes(X_val, XC):
    alphas = np.zeros((X_val.shape[0], XC.shape[0]))
    for i in range(X_val.shape[0]):
        x = X_val[i]
        fun = lambda a: np.sum((x - a @ XC) ** 2)  # now XC is (k, n_features)
        cons = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1}, 
                {'type': 'ineq', 'fun': lambda a: a})
        a0 = np.ones(XC.shape[0]) / XC.shape[0]
        res = minimize(fun, a0, constraints=cons, method='SLSQP')
        alphas[i] = res.x
    return alphas

def cv_grid_search_archetypes(df, categorical_cols, k_values=[6,7,8,9],delta_values=[0.1,0.3,0.5],n_splits=5,random_state=42):
    """
    Performs KFold CV to choose (k, delta) for archetypal analysis.
    """
    X_raw = df.drop(columns=categorical_cols)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = []

    for k in k_values:
        for delta in delta_values:
            fold_errors = []
            for train_idx, val_idx in kf.split(X_raw):
                # split
                train_df = df.iloc[train_idx]
                val_df   = df.iloc[val_idx]

                # impute numericals
                num_cols = train_df.select_dtypes(include=np.number).columns
                imputer  = SimpleImputer(strategy='mean')
                X_tr_num = imputer.fit_transform(train_df[num_cols])
                X_va_num = imputer.transform(val_df[num_cols])

                # scale
                scaler     = StandardScaler()
                X_tr_std   = scaler.fit_transform(X_tr_num)
                X_va_std   = scaler.transform(X_va_num)

                # fit AA
                XC_raw, _, _, _, _ = PCHA(X_tr_std.T, noc=k, delta=delta)
                XC = np.asarray(XC_raw).T  # shape k × p

                # project and reconstruct
                alphas      = project_to_archetypes(X_va_std, XC)
                X_va_recon  = alphas @ XC

                # compute error
                fold_errors.append(np.mean((X_va_std - X_va_recon)**2))

            avg_err = np.mean(fold_errors)
            cv_results.append({'k': k, 'delta': delta, 'error': avg_err})
            print(f"[CV] k={k}, delta={delta:.2f}, error={avg_err:.4f}")

    best = min(cv_results, key=lambda x: x['error'])
    print(f"Best params: k={best['k']}, delta={best['delta']:.2f}, error={best['error']:.4f}")
    return cv_results, best


def emotion_weight_clustering(df, weights,
                              emotion_cols=None,
                              combined_k=3,
                              random_state=42):
    """
    Clusters by [archetype weights + emotions], shows PCA map & emotion profiles
    """
    if emotion_cols is None:
        emotion_cols = ['upset','hostile','alert','ashamed',
                        'inspired','nervous','attentive','afraid','active','determined']
    
    # combined weights + emotions 
    W = weights
    E = df[emotion_cols].astype(float).values
    
    # scale
    Ws = StandardScaler().fit_transform(W)
    Es = StandardScaler().fit_transform(E)
    
    # concat & cluster
    Xc = np.hstack([Ws, Es])
    kmc = KMeans(n_clusters=combined_k, random_state=random_state).fit(Xc)
    labels_c = kmc.labels_
    df['combined_cluster'] = labels_c
    
    # PCA 2D
    p2 = PCA(n_components=2).fit_transform(Xc)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=p2[:,0], y=p2[:,1], hue=labels_c, palette='tab10', s=50)
    plt.title('PCA of weights+emotions (combined clusters)')
    plt.tight_layout()
    plt.show()
    
    # mean emotion profile per combined cluster
    cep = df.groupby('combined_cluster')[emotion_cols].mean().reset_index().melt(
        id_vars='combined_cluster', var_name='emotion', value_name='mean_score'
    )
    plt.figure(figsize=(10,4))
    sns.barplot(data=cep, x='emotion', y='mean_score', hue='combined_cluster')
    plt.xticks(rotation=45)
    plt.title('Mean emotion ratings by combined cluster')
    plt.tight_layout()
    plt.show()
       
    return {
        'combined': {
            'labels': labels_c,
            'pca2d': p2,
            'cluster_emotions_profile': cep
        }
    }

def plot_archetype_weights_by_phase(weights, phases, component_prefix='A'):
    # overall distributions
    n_components = weights.shape[1]
    comp_names = [f"{component_prefix} {i+1}" for i in range(n_components)]
    df_all = pd.DataFrame(weights, columns=comp_names)
    
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df_all)
    plt.title('Distribution of archetype activation weights')
    plt.xlabel('Archetype')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # faceted by Phase
    df_phase = df_all.copy()
    df_phase['Phase'] = pd.Series(phases).astype(str).values
    melted = df_phase.melt(
        id_vars='Phase',
        var_name='Archetype',
        value_name='Weight'
    )
    
    g = sns.catplot(
        data=melted,
        x='Phase',
        y='Weight',
        col='Archetype',
        kind='box',
        col_wrap=3,
        sharey=False,
        height=4
    )
    g.fig.suptitle('Archetype activation by task phase', y=1.02)
    g.set_axis_labels("Phase", "Weight")
    plt.tight_layout()
    plt.show()

# track archetype evolution through rounds
# use S.T weights directly
# visualize how much each archetype is "activated" for a specific participant over time, 
# across different task phases and rounds.
def plot_archetype_dynamics(df, weights, participant=None, 
                            id_col='Individual', round_col='Round', phase_col='Phase',
                            prefix='Archetype', n_clusters = 3):
    """
    Plots time-series of archetype weights for one participant (if given),
    plus the mean dynamics across all participants. Also clusters the task
    phases based on archetype activation.
    """

    # ensure ID column is string for comparison
    df[id_col] = df[id_col].astype(str)
    n_comp = weights.shape[1]

    # individual dynamics
    if participant is not None:
        pid = str(participant)
        mask = df[id_col] == pid
        n_sel = mask.sum()
        if n_sel == 0:
            raise ValueError(f"No rows found for participant {pid}")
        print(f"Plotting {n_sel} segments for participant {pid}")

        df_sub = df.loc[mask].reset_index(drop=True)
        w_sub  = weights[mask.values]

        # sort by round & phase
        df_sub = df_sub.sort_values(by=[round_col, phase_col]).reset_index(drop=True)
        w_sub   = w_sub[df_sub.index.values]

        # build x‐axis labels
        labels = [f"R{r} {p}" for r, p in zip(df_sub[round_col], df_sub[phase_col])]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_sel)
        for i in range(n_comp):
            ax.plot(x, w_sub[:, i], marker='o', label=f'{prefix} {i+1}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Task (round + phase)')
        ax.set_ylabel('Activation weight')
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.title(f'{prefix} Dynamics for participant {pid}')
        plt.tight_layout()
        plt.show()

    # mean dynamics across participants
    # sort full df
    df_all = df.copy().reset_index(drop=True)
    df_all = df_all.sort_values(by=[round_col, phase_col]).reset_index(drop=True)
    # group and average
    grouped = df_all.groupby([round_col, phase_col])
    mean_weights = []
    labels = []
    for (r, p), grp in grouped:
        mean_weights.append(weights[grp.index].mean(axis=0))
        labels.append(f"R{r} {p}")
    mean_weights = np.vstack(mean_weights)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    for i in range(n_comp):
        ax.plot(x, mean_weights[:, i], marker='o', label=f'{prefix} {i+1}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Task (round + phase)')
    ax.set_ylabel('Mean activation weight')
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title(f'Mean {prefix} dynamics across participants')
    plt.tight_layout()
    plt.show()

    mean_weights = np.vstack(mean_weights)
    # run KMeans on mean activation weights
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(mean_weights)

    # create a DataFrame for plotting
    cluster_df = pd.DataFrame({
        'Task': labels,
        'Cluster': clusters
    })

    # extract Round and Phase from 'Task' string and convert to int
    cluster_df['Round'] = cluster_df['Task'].str.extract(r'round_(\d+)', expand=False).astype(int)
    cluster_df['Phase'] = cluster_df['Task'].str.extract(r'phase(\d+)', expand=False).astype(int)
    print(cluster_df.head())

    # visualize clusters by round/phase
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=cluster_df,
        x='Phase',
        y='Round',
        hue='Cluster',
        palette='tab10',
        s=200,
        legend='full'
    )
    plt.title('Clusters of task phases based on Archetype activation patterns')
    plt.gca().invert_yaxis()  
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    from initial_data import load_data, inspect_data, preprocess_data
    # load the data
    file_path = 'HR_data_2.csv'
    df = load_data(file_path)
            
    # inspect the data
    inspect_data(df)
    df = df.drop(columns=['Unnamed: 0'])
    inspect_data(df)

    # convert selected columns to object type to work with imputation in preprocessing from initial_data.py
    categorical = ['Round','Phase','Individual','Puzzler','original_ID','raw_data_path','Team_ID','Frustrated','Cohort',
        'upset','hostile','alert','ashamed','inspired','nervous',
        'attentive','afraid','active','determined']

    pure_cat = ['Round','Phase','raw_data_path','Team_ID', 'Cohort']

    df[categorical] = df[categorical].astype('object')

    # preprocess the data -> imputing missing values
    df = preprocess_data(df)

    X_df = df.drop(columns=categorical) # drop categorical values for lower dimensional representation

    # prepare different versions of the data to be given to different models
    X = X_df.values

    # standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)  # shape (312, 52)

    plot_pcha_sse(X_std, k_min=2, k_max=9, delta=0.1)
    #cv_results, best_params = cv_grid_search_archetypes(df=df,categorical_cols=categorical,k_values=[4,5,6,7,8,9,10,11],delta_values=[0.2,0.4,0.5,0.8],n_splits=5)
    # number of components from cross-validation
    n_components = 11 # 8
    delta = 0.8 # 0.9

    XC, S, C, SSE, varexpl = PCHA(X_std.T, noc=n_components, delta=delta)
    archetypes = XC.T

    archetype_df = pd.DataFrame(XC, 
                                index=X_df.columns, 
                                columns=[f"Archetype_{i+1}" for i in range(XC.shape[1])])

    # visualize the variable loadings for each archetype
    plt.figure(figsize=(14, 8))
    sns.heatmap(archetype_df, annot=False, cmap="coolwarm", center=0)
    plt.title("Variable Contributions to Each Archetype")
    plt.xlabel("Archetypes")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()


    weights = S.T  # shape (312, 5), sample weights
    emotion_weight_clustering(df, weights)

    plot_archetype_weights_by_phase(weights, df['Phase'], component_prefix='A')


    # look at reconstruction error for the two different methods
    pca = PCA(n_components).fit(X_std)

    print(f"AA Variance Explained: {varexpl:.2%}")
    print(f"PCA Variance Explained: {pca.explained_variance_ratio_.sum():.2%}")

    # calculate AA reconstruction error
    X_reconstructed = (XC @ S).T  # (52,5) @ (5,312) > (52,312) -> transpose to (312,52)
    aa_reconstruction_error = np.mean(np.square(X_std - X_reconstructed))

    print(f"AA Reconstruction Error: {aa_reconstruction_error:.2f}")
    print(f"PCA Reconstruction Error: {np.mean(np.square(X_std - pca.inverse_transform(pca.transform(X_std)))):.2f}")

    plot_archetype_dynamics(df, weights, participant='2', n_clusters = 2)


'''
###### some additional PCA stuff which does the same TS-plot as for the AA, maybe useful?
from sklearn.decomposition import PCA

n_components = 10  
pca = PCA(n_components=n_components)
pca_weights = pca.fit_transform(X_std)  

participant = '2'
df['Individual'] = df['Individual'].astype(str)
mask = df['Individual'] == participant

n_selected = mask.sum()
print(f"Selected {n_selected} rows for participant {participant}")
if n_selected == 0:
    raise ValueError(f"No rows found for participant {participant}")

df_sub = df.loc[mask].reset_index(drop=True)
weights_sub = pca_weights[mask.values, :]

# plot time series
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(weights_sub.shape[1]):
    ax.plot(np.arange(n_selected), weights_sub[:, i], marker='o', label=f'PC{i+1}')

df_sub_sorted = df_sub.sort_values(by=['Round', 'Phase'])
labels = [f"R{r} {p}" for r, p in zip(df_sub_sorted['Round'], df_sub_sorted['Phase'])]
ax.set_xticks(np.arange(n_selected))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('PCA Component Value')
ax.set_xlabel('Task (round + phase)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'PCA dynamics for participant {participant}')
plt.tight_layout()
plt.show()


df_sorted = df.sort_values(by=['Round', 'Phase']).reset_index(drop=True)
grouped = df_sorted.groupby(['Round', 'Phase'])

mean_weights = []
labels = []

for (round_, phase), group in grouped:
    group_weights = pca_weights[group.index, :]
    mean = group_weights.mean(axis=0)
    mean_weights.append(mean)
    labels.append(f"R{round_} {phase}")

mean_weights = np.array(mean_weights)


fig, ax = plt.subplots(figsize=(12, 6))
for i in range(mean_weights.shape[1]):
    ax.plot(np.arange(len(labels)), mean_weights[:, i], marker='o', label=f'PC{i+1}')

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Mean PCA Value')
ax.set_xlabel('Task (round + phase)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Mean PCA Dynamics across Tasks')
plt.tight_layout()
plt.show()



from sklearn.cluster import KMeans

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(mean_weights)

cluster_df = pd.DataFrame({
    'Task': labels,
    'Cluster': clusters
})

# Extract Round and Phase
cluster_df['Round'] = cluster_df['Task'].str.extract(r'round_(\d+)', expand=False).astype(int)
cluster_df['Phase'] = cluster_df['Task'].str.extract(r'phase(\d+)', expand=False).astype(int)


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=cluster_df,
    x='Phase',
    y='Round',
    hue='Cluster',
    palette='tab10',
    s=200,
    legend='full'
)
plt.title('Clusters of task phases based on PCA patterns')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

'''