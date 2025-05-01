import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_data import load_data, inspect_data, preprocess_data, visualize_data, scale_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, DictionaryLearning, FastICA
from sklearn.cluster import KMeans
from py_pcha import PCHA  
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load the data
file_path = 'HR_data_2.csv'
df = load_data(file_path)
        
# inspect the data
inspect_data(df)

# convert selected columns to object type to work with imputation in preprocessing from initial_data.py
categorical = ['Round','Phase','Individual','Puzzler','Frustrated','Cohort',
    'upset','hostile','alert','ashamed','inspired','nervous',
    'attentive','afraid','active','determined', 'raw_data_path', 'Team_ID']

df[categorical] = df[categorical].astype('object')

# preprocess the data -> imputing missing values
df = preprocess_data(df)

X_df = df.drop(columns=categorical) # drop categorical values for lower dimensional representation

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# prepare different versions of the data to be given to different models
X = X_scaled

XC, S, C, SSE, varexpl = PCHA(X.T, noc=10, delta=0.1)
X_trans = np.array(S).T           # S matrix: samples x archetypes

# FastICA
#FastICA_ = FastICA(n_components=10, random_state=0)
#X_trans = FastICA_.fit_transform(X)
#XC = FastICA_.components_.T  # components (archetypes) x features

# scatter of the first four dims
state = 'inspired'  # can pick any emotion or Puzzler etc
color_values = df[state].values

proj_df = pd.DataFrame(X_trans[:, :6], columns=[f"Comp_{i+1}" for i in range(6)])
proj_df[state] = df[state]
print(proj_df[state])

sns.pairplot(proj_df, hue=state, palette='tab10')  # 'hue' based on the emotion state
plt.suptitle(f"AA: Pairplot of first 4 components colored by '{state}'")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.heatmap(X_trans, cmap='viridis', cbar=True)
plt.title("Soft Weights: Subjects vs Archetypes")
plt.xlabel("Archetypes")
plt.ylabel("Subjects")
plt.tight_layout()
plt.show()


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

for arch in archetype_df.columns:
    print(f"Top variables for {arch}:")
    print(archetype_df[arch].abs().sort_values(ascending=False).head(5))
    print()

emotion_states = ['Frustrated',
    'upset','hostile','alert','ashamed','inspired','nervous',
    'attentive','afraid','active','determined'] 

# --- HARD ASSIGNMENT ---
# Find closest archetype
closest_archetype = np.argmax(X_trans, axis=1)

# Add assignment to dataframe
df['Closest_Archetype'] = closest_archetype

# Group by closest archetype
hard_emotion_means = df.groupby('Closest_Archetype')[emotion_states].mean()

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(hard_emotion_means, annot=True, cmap='coolwarm', center=0)
plt.title("Emotion Means per Archetype (Hard Assignment)")
plt.xlabel("Emotion")
plt.ylabel("Archetype")
plt.tight_layout()
plt.show()


# --- SOFT ASSIGNMENT ---

# Create a soft-weighted dataframe
# Create DataFrame to hold soft emotion means (components x emotions)
""" FOR ICA
soft_emotion_means = pd.DataFrame(0.0, index=[f"IC_{i+1}" for i in range(X_trans.shape[1])], columns=emotion_states)

for i in range(X_trans.shape[1]):
    weights = np.abs(X_trans[:, i])  # shape: (n_samples,)
    
    # Avoid divide-by-zero
    if np.isclose(weights.sum(), 0):
        continue
    
    # Compute weighted average of emotions using soft assignment
    weighted_sum = (df[emotion_states].T * weights).T.sum(axis=0)
    soft_emotion_means.iloc[i] = weighted_sum / weights.sum()

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(soft_emotion_means, annot=True, cmap='viridis', center=0)
plt.title("Emotion Means per ICA Component (Soft Assignment)")
plt.xlabel("Emotion")
plt.ylabel("Independent Component")
plt.tight_layout()
plt.show()
"""

soft_emotion_means = pd.DataFrame(0, index=range(X_trans.shape[1]), columns=emotion_states)  # (archetypes x emotions)

# Loop over archetypes
for arch in range(X_trans.shape[1]):
    # For each archetype, weight emotions by soft membership
    weights = X_trans[:, arch]
    weighted_emotions = df[emotion_states].multiply(weights, axis=0)
    soft_emotion_means.loc[arch] = weighted_emotions.sum(axis=0) / weights.sum()

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(soft_emotion_means, annot=True, cmap='viridis', center=0)
plt.title("Emotion Means per Archetype (Soft Assignment)")
plt.xlabel("Emotion")
plt.ylabel("Archetype")
plt.tight_layout()
plt.show()


"""
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

range_n_clusters = range(2, 12)  # Try 2 to 10 clusters

X_input = X_trans  # Use the ICA or AA projection

for n_clusters in range_n_clusters:
    fig, ax = plt.subplots(figsize=(7, 5))

    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_input)

    # Silhouette score
    silhouette_avg = silhouette_score(X_input, cluster_labels)
    print(f"For n_clusters = {n_clusters}, average silhouette_score = {silhouette_avg:.3f}")

    # Compute silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_input, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot for {n_clusters} Clusters")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    plt.tight_layout()
    plt.show()
"""

from sklearn.manifold import TSNE

# === CLUSTERING ===
kmeans = KMeans(n_clusters=9, random_state=0)  # choose your number of clusters
cluster_labels = kmeans.fit_predict(X_trans)

# === 2D PROJECTION ===
# Option 1: PCA (linear)
X_2d = PCA(n_components=2).fit_transform(X_trans)

# Option 2: t-SNE (nonlinear, slower)
# X_2d = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X_trans)

# === VISUALIZATION ===
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=cluster_labels, palette='tab10', s=60)
plt.title("Subject Clusters Based on Component Weights")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

df['Cluster'] = cluster_labels
cluster_emotions = df.groupby('Cluster')[emotion_states].mean()

plt.figure(figsize=(12, 5))
sns.heatmap(cluster_emotions, annot=True, cmap='coolwarm', center=0)
plt.title("Mean Emotion Ratings per Cluster")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()