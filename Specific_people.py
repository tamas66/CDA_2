import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_data import load_data, inspect_data, preprocess_data, visualize_data, scale_data
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

file_path = 'HR_data_2.csv'
df = load_data(file_path)
        
# inspect the data
inspect_data(df)

# convert selected columns to object type to work with imputation in preprocessing from initial_data.py
categorical = ['Round','Phase','Individual','Puzzler','Cohort', 'raw_data_path', 'Team_ID', 'original_ID', "Unnamed: 0"]

df[categorical] = df[categorical].astype('object')

# Pick out only columns with "Round" == "round_1"
# df = df[(df['Phase'] == 'phase2') | (df['Phase'] == 'phase1')]

# preprocess the data -> imputing missing values
df = preprocess_data(df)

X_df = df.drop(columns=categorical) # drop categorical values for lower dimensional representation

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_df)
X_scaled = X_df.values  # Use the raw values for PCA and other analyses

import numpy as np

group1 = X_scaled[np.r_[1:170, 241:310]]
group2 = X_scaled[171:241]

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[1:160, 0], X_pca[1:160, 1], label='Cohorts Dx_y (1–160)', alpha=0.5)
plt.scatter(X_pca[235:310, 0], X_pca[235:310, 1], label='Cohorts Dz_w (235–310)', alpha=0.5)
plt.scatter(X_pca[160:235, 0], X_pca[160:235, 1], label='Cohorts D1_2 (160–235)', alpha=0.5)
plt.legend()
plt.title('PCA with cohort D1_2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

ks_results = [ks_2samp(group1[:, i], group2[:, i]) for i in range(X_scaled.shape[1])]

feature_names = X_df.columns

for i, result in enumerate(ks_results):
    if result.pvalue < 0.05:  # Check if the p-value is less than 0.05
        print(f"Feature {feature_names[i]}: KS statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4f} (significant)")

# Pairwise barplot of each feature of the two groups
plt.figure(figsize=(12, 8))
plt.bar(range(X_scaled.shape[1]), np.mean(group1, axis=0), alpha=0.5, label='Cohorts (1–170 and 241–310)')
plt.bar(range(X_scaled.shape[1]), np.mean(group2, axis=0), alpha=0.5, label='Cohorts D1_2 (170–241)')
plt.xticks(range(X_scaled.shape[1]), feature_names, rotation=90)
plt.xlabel('Features')

plt.ylim(0, 100)

plt.ylabel('Mean Value')
plt.title('Mean Values of Features for Two Groups')
plt.legend()
plt.tight_layout()
plt.show()