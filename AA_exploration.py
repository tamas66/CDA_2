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