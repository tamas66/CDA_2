# analysis_pipeline.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from initial_data import load_data, preprocess_data, convert_to_cat

# --------------------------
# Data Loading Module
# --------------------------
def load_and_preprocess(file_path):
    """Load and preprocess data using custom functions"""
    raw_data = load_data(file_path)
    return preprocess_data(raw_data)

# --------------------------
# Modeling Module
# --------------------------
class LogisticRegressionAnalyzer:
    def __init__(self, n_folds=5, random_state=42, scoring="accuracy"):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.n_folds = n_folds
        self.scoring = scoring
        self.results = {}
        self.cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def validate_target(self, target_series):
        """Check if target is suitable for classification"""
        if not np.issubdtype(target_series.dtype, np.number):
            return False, "Non-numeric target"
        if target_series.nunique() < 2:
            return False, "Not enough classes (min 2 required)"
        return True, ""

    def prepare_features(self, df, target):
        """Prepare and scale features matrix"""
        # Convert features only, keep target numeric
        features_df = convert_to_cat(df, target)
        X = features_df.select_dtypes(np.number)
        if X.shape[1] < 3:
            raise ValueError(f"Only {X.shape[1]} predictors available")
        return self.scaler.fit_transform(X)

    def evaluate_target(self, df, target):
        """Run full evaluation pipeline for one target"""
        try:
            # Validate first before processing features
            valid, message = self.validate_target(df[target])
            if not valid:
                return message

            X_scaled = self.prepare_features(df, target)
            scores = cross_val_score(
                self.model, X_scaled, df[target],
                cv=self.cv, scoring=self.scoring
            )
            
            self.results[target] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_classes': df[target].nunique()
            }
            return f"{self.scoring}: {np.mean(scores):.3f} ± {np.std(scores):.3f}"
            
        except Exception as e:
            return f"Failed ({str(e)})"

    def analyze_all_targets(self, df, target_columns):
        """Process multiple targets with progress reporting"""
        for target in target_columns:
            status = self.evaluate_target(df, target)
            print(f"{target:<12s} {status}")

    def plot_results(self):
        """Visualize results with error bars"""
        if not self.results:
            raise ValueError("No results to plot. Run analyze_all_targets first.")
            
        sorted_targets = sorted(self.results.items(), 
                              key=lambda x: x[1]['mean_score'], 
                              reverse=True)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x=[t[0] for t in sorted_targets],
            y=[t[1]['mean_score'] for t in sorted_targets],
            yerr=[t[1]['std_score'] for t in sorted_targets],
            fmt='o',
            capsize=5
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Classification Performance ({self.scoring.capitalize()})")
        plt.xlabel("Target Variable")
        # Write mean values on top of the points
        for i, (target, result) in enumerate(sorted_targets):
            plt.text(i, result['mean_score'] + 0.02, 
                     f"{result['mean_score']:.3f}", 
                     ha='center', va='bottom')
        plt.ylabel(f"{self.scoring.capitalize()} Score")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Configuration
    TARGET_COLUMNS = [
        "Individual", "Puzzler", "Frustrated",
        "upset", "hostile", "alert", "ashamed", "inspired", "nervous",
        "attentive", "afraid", "active", "determined"
    ]
    
    # Run pipeline
    df = load_and_preprocess('HR_data.csv')
    analyzer = LogisticRegressionAnalyzer(n_folds=5, scoring="accuracy")
    analyzer.analyze_all_targets(df, TARGET_COLUMNS)
    analyzer.plot_results()