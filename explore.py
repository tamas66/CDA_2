import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_data import load_data, inspect_data, preprocess_data, visualize_data, scale_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, DictionaryLearning
from sklearn.cluster import KMeans
from py_pcha import PCHA  
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load the data
file_path = 'HR_data.csv'
df = load_data(file_path)
        
# inspect the data
inspect_data(df)

# convert selected columns to object type to work with imputation in preprocessing from initial_data.py
categorical = ['Round','Phase','Individual','Puzzler','Frustrated','Cohort',
    'upset','hostile','alert','ashamed','inspired','nervous',
    'attentive','afraid','active','determined']

df[categorical] = df[categorical].astype('object')

# preprocess the data -> imputing missing values
df = preprocess_data(df)

X_df = df.drop(columns=categorical) # drop categorical values for lower dimensional representation

# prepare different versions of the data to be given to different models
X = X_df.values
X_std = StandardScaler().fit_transform(X)  
X_nmf  = X - np.min(np.min(X))  

# define methods
n_components = 10  # change as desired, param to be tuned
methods = {
    'PCA': PCA(n_components=n_components),
    'NMF': NMF(n_components=n_components, init='random', random_state=0),
    'DictLearn': DictionaryLearning(n_components=n_components, alpha=1.0, max_iter=200),
    'KMeans': KMeans(n_clusters=n_components, random_state=0),
    'AA': None
}

# loop, fit, transform, and plot
for name, model in methods.items():
    # pick the right input
    if name=='NMF': 
        X_in = X_nmf  
    elif name=='AA':
        X_in = X
    else:
        X_in = X_std

    # do the decomposition/clustering
    if name == 'KMeans':
        model.fit(X_in)
        # use cluster center distances as a low-dim embedding
        X_trans = model.transform(X_in)
    elif name == 'AA':
        XC, S, C, SSE, varexpl = PCHA(X.T, noc=n_components, delta=0.1)
        X_trans = np.array(S).T   # put back to samples x archetypes
    else:
        X_trans = model.fit_transform(X_in)

    # scatter of the first four dims
    state = 'inspired'  # can pick any emotion or Puzzler etc
    color_values = df[state].values

    proj_df = pd.DataFrame(X_trans[:, :4], columns=[f"Comp_{i+1}" for i in range(4)])
    proj_df[state] = df[state]
    print(proj_df[state])
    #'hue' is based on the emotion state
    sns.pairplot(proj_df, hue=state, palette='tab10')
    plt.suptitle(f"{name}: Pairplot of first 4 components colored by '{state}'")
    plt.tight_layout()
    plt.show()

