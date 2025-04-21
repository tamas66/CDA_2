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

    import seaborn as sns
    
    # Set the style of seaborn
    sns.set(style="whitegrid")
    
    # Plot histograms for numerical columns
    df.hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.show()
    
    # Plot box plots for numerical columns
    if boxplot:
        for col in df.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[col])
            plt.title(f'Box plot of {col}')
            plt.show()

def pca_analysis(df: pd.DataFrame, n_variance: float = 0.9) -> None:
    """
    Perform PCA analysis on the DataFrame and visualize the results.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        n_components (int): The number of principal components to keep.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if n_variance < 0 or n_variance > 1:
        raise ValueError("n_variance must be between 0 and 1.")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    # Perform PCA
    pca = PCA(n_components=n_variance)
    pca_result = pca.fit_transform(scaled_data)

    n_components = pca.n_components_
    print(f"Number of components selected: {n_components}")
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Print Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=df.select_dtypes(include=[np.number]).columns)
    print("\nPCA Loadings:")
    print(loadings)

    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    # Scree Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_, marker='o')
    plt.plot(range(1, n_components + 1), cum_explained_variance, marker='o', linestyle='--')
    plt.axhline(y=n_variance, color='r', linestyle='--')
    plt.axvline(x=n_components, color='g', linestyle='--')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.xticks(range(1, n_components + 1))
    plt.grid()
    plt.show()

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1])
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load the data
    file_path = 'HR_data.csv'  # Replace with your actual file path
    df = load_data(file_path)
    
    # Inspect the data
    inspect_data(df)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Visualize the data
    visualize_data(df)

    pca_analysis(df, n_variance=0.9)
