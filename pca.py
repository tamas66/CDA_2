import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_data import load_data, preprocess_data, pca_analysis, scale_data

def validate_dataframe(df):
    """Ensure DataFrame contains only numeric columns and has sufficient features"""
    if not all(df.dtypes.apply(pd.api.types.is_numeric_dtype)):
        raise ValueError("DataFrame contains non-numeric columns")
    if df.shape[1] < 2:
        raise ValueError("PCA requires at least 2 numeric features")

def pca(df, n_components=None, variance_threshold=None, return_model=False, numeric_only=True):
    """
    PCA implementation based on Week 8, Exercise 1
    
    Parameters:
    df (pd.DataFrame): Input data with samples as rows and features as columns
    n_components (int): Number of components to retain (optional)
    variance_threshold (float): Minimum explained variance to keep (0-1) (optional)
    return_model (bool): Return transformation metadata (default: False)
    numeric_only (bool): Auto-filter numeric columns (default: True)
    
    Returns:
    pd.DataFrame: Transformed data with PC columns
    tuple (if return_model): (transformed_df, pca_metadata)
    """
    # Store original metadata
    original_features = df.columns.tolist()
    index = df.index
    
    # Filter numeric columns
    if numeric_only:
        df = df.select_dtypes(include=np.number)
        if df.shape[1] == 0:
            raise ValueError("No numeric columns found in DataFrame")
    
    try:
        validate_dataframe(df)
    except ValueError as e:
        raise ValueError(f"Data validation failed: {str(e)}") from e
    
    X = df.values
    
    X_scaled, mean, _ = scale_data(X)
    
    try:
        cov = np.cov(X_scaled, rowvar=False)
    except:
        raise RuntimeError("Failed to compute covariance matrix - check input data quality")
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Variance calculations
    total_variance = eigenvalues.sum()
    explained_variance = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance)
    
    # Component selection logic
    if variance_threshold is not None:
        if not 0 <= variance_threshold <= 1:
            raise ValueError("Variance threshold must be between 0 and 1")
        
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        if n_components == 0:
            n_components = len(eigenvalues)
            if variance_threshold > cumulative_variance[-1]:
                import warnings
                warnings.warn(f"Variance threshold {variance_threshold} not achievable. Using all components.")
    
    n_components = min(n_components or len(eigenvalues), len(eigenvalues))
    
    # Prepare outputs
    components = eigenvectors[:, :n_components]
    transformed = X_scaled @ components
    
    # Create labeled DataFrame
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    transformed_df = pd.DataFrame(transformed, columns=pc_columns, index=index)
    
    if return_model:
        metadata = {
            'components': pd.DataFrame(components.T, columns=df.columns, index=pc_columns),
            'explained_variance_ratio': explained_variance[:n_components],
            'cumulative_variance': cumulative_variance[n_components-1],
            'mean': pd.Series(mean, index=df.columns),
            'original_features': original_features,
            'eigenvalues': eigenvalues
        }
        return transformed_df, metadata
    
    return transformed_df

def plot_scree_plot(model, variance_threshold=None, plot_variance=True):
    """Plot scree plot with variance threshold indicators (single axis)"""

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract values based on plot type
    values = model['eigenvalues'] / model['eigenvalues'].sum() if plot_variance else model['eigenvalues']
    cumulative = np.cumsum(values)
    n_components = len(values)
    x_range = range(1, n_components + 1)
    
    # Create main plot elements
    if plot_variance:
        # Bar plot for individual variance
        bars = ax.bar(x_range, values, alpha=0.5, label='Individual Variance')
        
        # Line plot for cumulative variance
        line, = ax.plot(x_range, cumulative, 'o--', color='red', label='Cumulative Variance')
        ax.set_ylim(0, 1.1)
        ylabel = 'Explained Variance Ratio'
    else:
        # Simple line plot for eigenvalues
        line, = ax.plot(x_range, values, 'o-', color='blue')
        ylabel = 'Eigenvalues'
    
    # Add threshold indicators if provided
    threshold_lines = []
    if variance_threshold is not None and plot_variance:
        # Horizontal threshold line
        th_line = ax.axhline(variance_threshold, color='green', linestyle='--', 
                            label=f'Threshold ({variance_threshold*100:.1f}%)')
        threshold_lines.append(th_line)
        
        # Vertical component selection line
        selected = np.argmax(cumulative >= variance_threshold) + 1
        if selected == 0:
            selected = n_components
        v_line = ax.axvline(selected, color='purple', linestyle=':', 
                           label=f'Selected (n={selected})')
        threshold_lines.append(v_line)
    
    ax.set_title('Scree Plot: ' + ('Variance Explained' if plot_variance else 'Eigenvalues'))
    ax.set_xlabel('Principal Component')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_range)
    
    # Combine all legend elements
    handles = [bars[0] if plot_variance else line, line] if plot_variance else [line]
    if threshold_lines:
        handles += threshold_lines
    ax.legend(handles=handles, loc='upper left' if plot_variance else 'best')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("HR_data.csv")
    df = preprocess_data(df)
    
    # Perform PCA with variance threshold
    pca_result, pca_model = pca(df, variance_threshold=0.9, return_model=True)
    
    # Plot variance tresholds of analytical model
    plot_scree_plot(pca_model, variance_threshold=0.9, plot_variance=True)

    # Plot variance tresholds of sklearn model
    pca_analysis(df, n_variance=0.9)

    # Plot scree plot without variance threshold
    # Note: This will show the eigenvalues instead of explained variance ratio
    plot_scree_plot(pca_model, variance_threshold=None, plot_variance=False)

    # Display results
    print("Top 5 Transformed Samples:")
    print(pca_result.head())
    print("\nComponent Loadings:")
    print(pca_model['components'].round(3))
    print("\nExplained Variance per Component:")
    print(pd.Series(pca_model['explained_variance_ratio'], 
                    index=pca_model['components'].index).round(3))
    print(f"\nTotal Explained Variance: {pca_model['cumulative_variance']:.1%}")