"""
Visualization script for Synthetic Financial Data Generator.

This script creates visual comparisons between real and synthetic data
to help understand how well the CTGAN model has learned the data distribution.

Generates:
- Distribution overlays (KDE plots) for key features
- Class distribution comparison (bar charts)

All plots are saved to outputs/plots/

Author: ML Engineer
Date: 2026
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def get_project_root():
    """
    Get the project root directory.

    Returns:
        Path: Absolute path to the project root directory
    """
    return Path(__file__).parent.parent


def load_data(project_root):
    """
    Load both real and synthetic datasets.

    Args:
        project_root (Path): Path to project root

    Returns:
        tuple: (real_data, synthetic_data) DataFrames
    """
    print("[INFO] Loading datasets...")

    real_data_path = project_root / "data" / "creditcard.csv"
    synthetic_data_path = project_root / "outputs" / "synthetic_data.csv"

    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    print(f"  - Real data: {real_data.shape}")
    print(f"  - Synthetic data: {synthetic_data.shape}")

    return real_data, synthetic_data


def load_model(project_root):
    """
    Load the trained CTGAN model to extract feature importances.

    Args:
        project_root (Path): Path to project root

    Returns:
        CTGANSynthesizer: Trained model
    """
    print("[INFO] Loading trained model...")

    model_path = project_root / "models" / "ctgan_model.pkl"
    model = joblib.load(model_path)

    print(f"  - Model loaded from: {model_path}")

    return model


def get_important_columns(real_data, target_column="Class", top_n=5):
    """
    Identify the most important columns for visualization.

    For the credit card dataset, we prioritize:
    1. Amount (transaction amount - highly interpretable)
    2. Time (transaction time - important pattern)
    3. Top V features by variance (most informative)

    Args:
        real_data (pd.DataFrame): Real dataset
        target_column (str): Target column name
        top_n (int): Number of top features to select

    Returns:
        list: List of important column names
    """
    print(f"\n[INFO] Selecting top {top_n} columns for visualization...")

    # Get all feature columns (exclude target and non-feature columns)
    feature_cols = [col for col in real_data.columns if col != target_column]

    # For credit card dataset, V1-V28 are PCA-transformed features
    # We'll select based on variance (higher variance = more information)

    # Calculate variance for each feature
    variances = {}
    for col in feature_cols:
        variances[col] = real_data[col].var()

    # Sort by variance (descending)
    sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)

    # Select top features, but prioritize Amount and Time for interpretability
    important_cols = []

    # Always include Amount if it exists (highly interpretable)
    if "Amount" in feature_cols:
        important_cols.append("Amount")

    # Always include Time if it exists (important temporal pattern)
    if "Time" in feature_cols:
        important_cols.append("Time")

    # Add remaining top features by variance
    remaining = top_n - len(important_cols)
    for feat, var in sorted_features:
        if feat not in important_cols and feat.startswith("V"):
            important_cols.append(feat)
            remaining -= 1
            if remaining <= 0:
                break

    # If we still need more columns, add any remaining features
    if len(important_cols) < top_n:
        for feat, var in sorted_features:
            if feat not in important_cols:
                important_cols.append(feat)
                if len(important_cols) >= top_n:
                    break

    print(f"  - Selected columns: {important_cols}")

    return important_cols


def plot_distribution_overlay(real_data, synthetic_data, columns, output_dir):
    """
    Create KDE distribution overlays for specified columns.

    For each column, creates a plot with:
    - Blue KDE curve for real data
    - Orange KDE curve for synthetic data

    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        columns (list): Columns to plot
        output_dir (Path): Directory to save plots
    """
    print("\n[INFO] Creating distribution overlay plots...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Create a figure for each column
    for col in columns:
        print(f"  - Plotting: {col}")

        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get data for this column
            real_values = real_data[col].dropna()
            synthetic_values = synthetic_data[col].dropna()

            # Check if column is binary/categorical (like V features often are)
            is_binary = (len(real_values.unique()) <= 10 and
                        len(synthetic_values.unique()) <= 10)

            if is_binary and col.startswith("V"):
                # For V features (PCA components), use histogram instead of KDE
                # Real data histogram
                ax.hist(real_values, bins=50, alpha=0.5,
                       label='Real', color='steelblue', density=True)

                # Synthetic data histogram
                ax.hist(synthetic_values, bins=50, alpha=0.5,
                       label='Synthetic', color='darkorange', density=True)

                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution Comparison: {col}\n(Real vs Synthetic)')

            else:
                # For continuous features like Amount and Time, use KDE
                # Real data KDE (blue)
                sns.kdeplot(
                    real_values,
                    ax=ax,
                    color='steelblue',
                    label='Real',
                    linewidth=2,
                    fill=True,
                    alpha=0.3
                )

                # Synthetic data KDE (orange)
                sns.kdeplot(
                    synthetic_values,
                    ax=ax,
                    color='darkorange',
                    label='Synthetic',
                    linewidth=2,
                    fill=True,
                    alpha=0.3
                )

                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution Comparison: {col}\n(Real vs Synthetic)')

            # Add legend
            ax.legend(loc='best', framealpha=0.9)

            # Adjust layout
            plt.tight_layout()

            # Save the plot
            output_path = output_dir / f"{col.lower()}_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"    Saved: {output_path.name}")

        except Exception as e:
            print(f"    [WARNING] Could not plot {col}: {e}")


def plot_class_distribution_comparison(real_data, synthetic_data, output_dir, target_column="Class"):
    """
    Create a bar chart comparing class distributions.

    Shows side-by-side comparison of:
    - Real data class counts
    - Synthetic data class counts

    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        output_dir (Path): Directory to save plot
        target_column (str): Target column name
    """
    print("\n[INFO] Creating class distribution comparison plot...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Create figure with two subplots: counts and percentages
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate class counts
    real_counts = real_data[target_column].value_counts().sort_index()
    synthetic_counts = synthetic_data[target_column].value_counts().sort_index()

    # Calculate percentages
    real_pcts = (real_counts / len(real_data)) * 100
    synthetic_pcts = (synthetic_counts / len(synthetic_data)) * 100

    # Define class labels
    class_labels = {0: 'Legitimate', 1: 'Fraud'}
    x_positions = [0, 1]
    x_labels = [class_labels[i] for i in sorted(real_counts.index)]

    # Plot 1: Absolute counts
    ax1 = axes[0]
    bar_width = 0.35
    x = np.arange(len(real_counts))

    # Real data bars (blue)
    bars1 = ax1.bar(
        x - bar_width/2,
        [real_counts[i] for i in sorted(real_counts.index)],
        bar_width,
        label='Real',
        color='steelblue',
        edgecolor='black',
        linewidth=1
    )

    # Synthetic data bars (orange)
    bars2 = ax1.bar(
        x + bar_width/2,
        [synthetic_counts[i] for i in sorted(synthetic_counts.index)],
        bar_width,
        label='Synthetic',
        color='darkorange',
        edgecolor='black',
        linewidth=1
    )

    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution: Absolute Counts\n(Real vs Synthetic)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend()

    # Add count labels on bars
    for bar, count in zip(bars1, [real_counts[i] for i in sorted(real_counts.index)]):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(real_counts.values())*0.01,
            f'{count:,}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    for bar, count in zip(bars2, [synthetic_counts[i] for i in sorted(synthetic_counts.index)]):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(synthetic_counts.values())*0.01,
            f'{count:,}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Plot 2: Percentages
    ax2 = axes[1]

    # Real data bars (blue)
    bars3 = ax2.bar(
        x - bar_width/2,
        [real_pcts[i] for i in sorted(real_pcts.index)],
        bar_width,
        label='Real',
        color='steelblue',
        edgecolor='black',
        linewidth=1
    )

    # Synthetic data bars (orange)
    bars4 = ax2.bar(
        x + bar_width/2,
        [synthetic_pcts[i] for i in sorted(synthetic_pcts.index)],
        bar_width,
        label='Synthetic',
        color='darkorange',
        edgecolor='black',
        linewidth=1
    )

    ax2.set_xlabel('Class')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Class Distribution: Percentage\n(Real vs Synthetic)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend()

    # Add percentage labels on bars
    for bar, pct in zip(bars3, [real_pcts[i] for i in sorted(real_pcts.index)]):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{pct:.3f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=45
        )

    for bar, pct in zip(bars4, [synthetic_pcts[i] for i in sorted(synthetic_pcts.index)]):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{pct:.3f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=45
        )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_path = output_dir / "class_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  - Saved: {output_path.name}")

    # Print comparison statistics
    print("\n  Class Distribution Comparison:")
    print("  " + "-"*50)
    real_fraud_pct = (real_counts[1] / len(real_data)) * 100 if 1 in real_counts else 0
    synthetic_fraud_pct = (synthetic_counts[1] / len(synthetic_data)) * 100 if 1 in synthetic_counts else 0
    print(f"  Real fraud rate:      {real_fraud_pct:.4f}%")
    print(f"  Synthetic fraud rate: {synthetic_fraud_pct:.4f}%")
    print(f"  Difference:           {abs(synthetic_fraud_pct - real_fraud_pct):.4f}%")
    print("  " + "-"*50)


def create_summary_table(real_data, synthetic_data, output_dir):
    """
    Create a summary statistics table comparing real and synthetic data.

    Args:
        real_data (pd.DataFrame): Real dataset
        synthetic_data (pd.DataFrame): Synthetic dataset
        output_dir (Path): Directory to save table
    """
    print("\n[INFO] Creating summary statistics table...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate summary statistics for key columns
    key_columns = ['Amount', 'Time', 'Class']

    summary_data = []

    for col in key_columns:
        if col in real_data.columns and col in synthetic_data.columns:
            row = {
                'Column': col,
                'Real Mean': f"{real_data[col].mean():.4f}",
                'Real Std': f"{real_data[col].std():.4f}",
                'Real Min': f"{real_data[col].min():.4f}",
                'Real Max': f"{real_data[col].max():.4f}",
                'Synth Mean': f"{synthetic_data[col].mean():.4f}",
                'Synth Std': f"{synthetic_data[col].std():.4f}",
                'Synth Min': f"{synthetic_data[col].min():.4f}",
                'Synth Max': f"{synthetic_data[col].max():.4f}",
            }
            summary_data.append(row)

    # Create DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    output_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"  - Saved: {output_path.name}")
    print("\n  Summary Statistics:")
    print(summary_df.to_string(index=False))


def main():
    """
    Main function to orchestrate the visualization pipeline.

    This function:
    1. Loads real and synthetic data
    2. Identifies important columns
    3. Creates distribution overlay plots
    4. Creates class distribution comparison
    5. Creates summary statistics table
    """
    print("\n" + "="*60)
    print("SYNTHETIC DATA VISUALIZATION PIPELINE")
    print("="*60)

    # Get the project root directory
    project_root = get_project_root()

    # Define output directory for plots
    output_dir = project_root / "outputs" / "plots"

    try:
        # Step 1: Load datasets
        print("\n[STEP 1] Loading datasets...")
        real_data, synthetic_data = load_data(project_root)

        # Step 2: Load model (optional, for feature importance)
        print("[STEP 2] Loading model...")
        model = load_model(project_root)

        # Step 3: Identify important columns for visualization
        print("[STEP 3] Identifying important columns...")
        important_cols = get_important_columns(real_data, top_n=5)

        # Step 4: Create distribution overlay plots
        print("[STEP 4] Creating distribution overlays...")
        plot_distribution_overlay(real_data, synthetic_data, important_cols, output_dir)

        # Step 5: Create class distribution comparison
        print("[STEP 5] Creating class distribution comparison...")
        plot_class_distribution_comparison(real_data, synthetic_data, output_dir)

        # Step 6: Create summary statistics table
        print("[STEP 6] Creating summary statistics...")
        create_summary_table(real_data, synthetic_data, output_dir)

        # Final summary
        print("\n" + "="*60)
        print("VISUALIZATION PIPELINE COMPLETED")
        print("="*60)
        print(f"\nPlots saved to: {output_dir}")
        print("\nGenerated files:")

        # List all generated files
        for file_path in sorted(output_dir.glob("*")):
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"  - {file_path.name} ({file_size:.1f} KB)")

        print("\n[INFO] Review plots in the outputs/plots/ directory\n")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please ensure you have run 'python src/train.py' first.")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Execute the main function when script is run directly
    main()
