import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    
    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found in the data.")
    
    return df

def scale_data(X: np.ndarray, only_center: bool = False) -> np.ndarray:
    """
    Scale the data using StandardScaler.
    
    Args:
        X (np.ndarray): The input data to scale.
        
    Returns:
        np.ndarray: The scaled data.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_centered = (X - mu)
    X_scaled = X_centered / sigma
    if only_center:
        return X_centered, mu, sigma

    return X_scaled, mu, sigma


def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspect the data by printing the first few rows and summary statistics.
    
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    print("First few rows of the data:")
    print(df.head())
    
    print("\nSummary statistics:")
    print(df.describe(include='all'))
    
    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values in each column:")
    print(df.isnull().sum())

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by filling missing values and converting data types.
    
    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Fill missing values with the mean of the column if numeric
    # or with the mode if categorical
    for col in df.columns:
        if df[col].dtype == 'object':
            # Fill categorical missing values with mode
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # Fill numeric missing values with mean
            df[col] = df[col].fillna(df[col].mean())
    
    # Convert categorical columns to category type
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    return df

def visualize_data(df: pd.DataFrame, boxplot = False) -> None:
    """
    Visualize the data using histograms and box plots.
    
    Args:
        df (pd.DataFrame): The DataFrame to visualize.
    """
    
    # Plot histograms for numerical columns
    df.hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.show()
    
    # Plot box plots for numerical columns
    if boxplot:
        plt.figure(figsize=(15, 6))
        df.select_dtypes(include=[np.number]).boxplot()
        plt.xticks(rotation=45)
        plt.title('Box plots of numerical columns')
        plt.tight_layout()
        plt.show()

def pca_analysis(df: pd.DataFrame, n_variance: float = 0.9) -> None:
    """
    Perform PCA analysis on the DataFrame and visualize the results.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        n_variance (float): The variance threshold to keep (0-1).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if n_variance < 0 or n_variance > 1:
        raise ValueError("n_variance must be between 0 and 1.")
    
    # Standardize the data
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Perform PCA
    pca = PCA(n_components=n_variance)
    pca_result = pca.fit_transform(scaled_data)

    n_components = pca.n_components_
    print(f"Number of components selected: {n_components}")
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Print Loadings with the same format as custom implementation
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numeric_df.columns
    )
    print("\nPCA Loadings:")
    print(loadings.round(3))  # Match rounding from custom implementation

    # Variance calculations to match custom implementation output
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print("\nExplained Variance Ratio:")
    print(np.round(explained_variance, 3))
    print("\nCumulative Variance:")
    print(np.round(cumulative_variance, 3))
    
    # Enhanced Scree Plot to match custom implementation style
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components+1), explained_variance, alpha=0.5, label='Individual')
    plt.plot(range(1, n_components+1), cumulative_variance, 'o--', color='red', label='Cumulative')
    plt.axhline(y=n_variance, color='green', linestyle='--', label=f'Threshold ({n_variance*100}%)')
    plt.axvline(x=n_components, color='purple', linestyle=':', label='Selected Components')
    plt.title('Scree Plot with Cumulative Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, n_components+1))
    plt.legend()
    plt.grid(True)
    plt.show()

    # PCA Scatter plot with same styling as custom implementation
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], alpha=0.7, edgecolors='w')
    plt.title('First Two Principal Components')
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
    plt.grid(True)
    plt.show()

    # PCA Loadings plot with same styling as custom implementation
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Plot variable loadings as points
    plt.scatter(loadings['PC1'], loadings['PC2'], color='red')

    # Annotate each point with the variable name
    for var, (x, y) in loadings[['PC1', 'PC2']].iterrows():
        plt.text(x, y, var, fontsize=10, ha='center', va='center')

    plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
    plt.title('PCA Loadings Plot (Variable Contributions)')
    plt.grid(True)
    plt.axis('equal')  # Makes unit lengths equal on both axes
    plt.show()

    return loadings

def plot_loadings_grid(loadings: pd.DataFrame, explained_variance: list = None, max_pc: int = 4):
    """
    Plots a grid of PCA loading plots for combinations of principal components.

    Parameters:
    - loadings: pd.DataFrame with PCs as columns (e.g., 'PC1', 'PC2', ...) and variables as index
    - explained_variance: list or array of explained variance ratios per PC (optional, for axis labels)
    - max_pc: number of PCs to include in the plot combinations (default is 4)
    """
    import itertools

    pc_names = loadings.columns[:max_pc]
    pc_pairs = list(itertools.combinations(pc_names, 2))

    n_plots = len(pc_pairs)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    axes = axes.flatten()

    for i, (pc_x, pc_y) in enumerate(pc_pairs):
        ax = axes[i]
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

        ax.scatter(loadings[pc_x], loadings[pc_y], color='red')

        for var, (x, y) in loadings[[pc_x, pc_y]].iterrows():
            ax.text(x, y, var, fontsize=9, ha='center', va='center')

        if explained_variance is not None:
            idx_x = int(pc_x[2:]) - 1
            idx_y = int(pc_y[2:]) - 1
            xlabel = f'{pc_x} ({explained_variance[idx_x]*100:.1f}%)'
            ylabel = f'{pc_y} ({explained_variance[idx_y]*100:.1f}%)'
        else:
            xlabel = pc_x
            ylabel = pc_y

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{pc_x} vs {pc_y}')
        ax.grid(True)
        ax.set_aspect('equal')

    # Hide any unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle('Grid of PCA Loadings Plots', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    # Load the data
    file_path = 'HR_data.csv'
    df = load_data(file_path)
    
    # Inspect the data
    inspect_data(df)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Visualize the data
    #visualize_data(df)

    # Perform basic PCA analysis
    loadings = pca_analysis(df, n_variance=0.9)

    # Plot the PCA loadings grid
    plot_loadings_grid(loadings, max_pc=4)

