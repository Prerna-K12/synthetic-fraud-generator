"""
Visualization script for Synthetic Financial Data Generator.
Optimized for Google Colab.

Author: ML Engineer
Date: 2026
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def load_data():
    """Load both real and synthetic datasets."""
    print("[INFO] Loading datasets...")

    real_data = pd.read_csv("/content/creditcard.csv")
    synthetic_data = pd.read_csv("/content/outputs/synthetic_data.csv")

    print(f"  - Real data: {real_data.shape}")
    print(f"  - Synthetic data: {synthetic_data.shape}")

    return real_data, synthetic_data


def get_important_columns(real_data, target_column="Class", top_n=5):
    """Identify the most important columns for visualization."""
    print(f"\n[INFO] Selecting top {top_n} columns for visualization...")

    feature_cols = [col for col in real_data.columns if col != target_column]

    variances = {col: real_data[col].var() for col in feature_cols}
    sorted_features = sorted(variances.items(), key=lambda x: x[1], reverse=True)

    important_cols = []

    if "Amount" in feature_cols:
        important_cols.append("Amount")
    if "Time" in feature_cols:
        important_cols.append("Time")

    remaining = top_n - len(important_cols)
    for feat, var in sorted_features:
        if feat not in important_cols and feat.startswith("V"):
            important_cols.append(feat)
            remaining -= 1
            if remaining <= 0:
                break

    if len(important_cols) < top_n:
        for feat, var in sorted_features:
            if feat not in important_cols:
                important_cols.append(feat)
                if len(important_cols) >= top_n:
                    break

    print(f"  - Selected columns: {important_cols}")

    return important_cols


def plot_distribution_overlay(real_data, synthetic_data, columns, output_dir):
    """Create KDE distribution overlays for specified columns."""
    print("\n[INFO] Creating distribution overlay plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    for col in columns:
        print(f"  - Plotting: {col}")

        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            real_values = real_data[col].dropna()
            synthetic_values = synthetic_data[col].dropna()

            is_binary = (len(real_values.unique()) <= 10 and len(synthetic_values.unique()) <= 10)

            if is_binary and col.startswith("V"):
                ax.hist(real_values, bins=50, alpha=0.5, label='Real', color='steelblue', density=True)
                ax.hist(synthetic_values, bins=50, alpha=0.5, label='Synthetic', color='darkorange', density=True)
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution Comparison: {col}\n(Real vs Synthetic)')
            else:
                sns.kdeplot(real_values, ax=ax, color='steelblue', label='Real', linewidth=2, fill=True, alpha=0.3)
                sns.kdeplot(synthetic_values, ax=ax, color='darkorange', label='Synthetic', linewidth=2, fill=True, alpha=0.3)
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution Comparison: {col}\n(Real vs Synthetic)')

            ax.legend(loc='best', framealpha=0.9)
            plt.tight_layout()

            output_path = output_dir / f"{col.lower()}_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"    Saved: {output_path.name}")

        except Exception as e:
            print(f"    [WARNING] Could not plot {col}: {e}")


def plot_class_distribution_comparison(real_data, synthetic_data, output_dir, target_column="Class"):
    """Create a bar chart comparing class distributions."""
    print("\n[INFO] Creating class distribution comparison plot...")

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    real_counts = real_data[target_column].value_counts().sort_index()
    synthetic_counts = synthetic_data[target_column].value_counts().sort_index()

    real_pcts = (real_counts / len(real_data)) * 100
    synthetic_pcts = (synthetic_counts / len(synthetic_data)) * 100

    class_labels = {0: 'Legitimate', 1: 'Fraud'}
    x_positions = [0, 1]
    x_labels = [class_labels[i] for i in sorted(real_counts.index)]

    ax1 = axes[0]
    bar_width = 0.35
    x = np.arange(len(real_counts))

    bars1 = ax1.bar(x - bar_width/2, [real_counts[i] for i in sorted(real_counts.index)],
                    bar_width, label='Real', color='steelblue', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + bar_width/2, [synthetic_counts[i] for i in sorted(synthetic_counts.index)],
                    bar_width, label='Synthetic', color='darkorange', edgecolor='black', linewidth=1)

    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution: Absolute Counts\n(Real vs Synthetic)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend()

    for bar, count in zip(bars1, [real_counts[i] for i in sorted(real_counts.index)]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(real_counts.values())*0.01,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

    for bar, count in zip(bars2, [synthetic_counts[i] for i in sorted(synthetic_counts.index)]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(synthetic_counts.values())*0.01,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[1]

    bars3 = ax2.bar(x - bar_width/2, [real_pcts[i] for i in sorted(real_pcts.index)],
                    bar_width, label='Real', color='steelblue', edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + bar_width/2, [synthetic_pcts[i] for i in sorted(synthetic_pcts.index)],
                    bar_width, label='Synthetic', color='darkorange', edgecolor='black', linewidth=1)

    ax2.set_xlabel('Class')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Class Distribution: Percentage\n(Real vs Synthetic)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend()

    for bar, pct in zip(bars3, [real_pcts[i] for i in sorted(real_pcts.index)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{pct:.3f}%', ha='center', va='bottom', fontsize=9, rotation=45)

    for bar, pct in zip(bars4, [synthetic_pcts[i] for i in sorted(synthetic_pcts.index)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{pct:.3f}%', ha='center', va='bottom', fontsize=9, rotation=45)

    plt.tight_layout()

    output_path = output_dir / "class_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  - Saved: {output_path.name}")

    print("\n  Class Distribution Comparison:")
    print("  " + "-"*50)
    real_fraud_pct = (real_counts[1] / len(real_data)) * 100 if 1 in real_counts else 0
    synthetic_fraud_pct = (synthetic_counts[1] / len(synthetic_data)) * 100 if 1 in synthetic_counts else 0
    print(f"  Real fraud rate:      {real_fraud_pct:.4f}%")
    print(f"  Synthetic fraud rate: {synthetic_fraud_pct:.4f}%")
    print(f"  Difference:           {abs(synthetic_fraud_pct - real_fraud_pct):.4f}%")
    print("  " + "-"*50)


def create_summary_table(real_data, synthetic_data, output_dir):
    """Create a summary statistics table comparing real and synthetic data."""
    print("\n[INFO] Creating summary statistics table...")

    output_dir.mkdir(parents=True, exist_ok=True)

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

    summary_df = pd.DataFrame(summary_data)
    output_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"  - Saved: {output_path.name}")
    print("\n  Summary Statistics:")
    print(summary_df.to_string(index=False))


def main():
    """Main function to orchestrate the visualization pipeline."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA VISUALIZATION PIPELINE")
    print("="*60)

    output_dir = Path("/content/outputs/plots")

    try:
        print("\n[STEP 1] Loading datasets...")
        real_data, synthetic_data = load_data()

        print("[STEP 2] Identifying important columns...")
        important_cols = get_important_columns(real_data, top_n=5)

        print("[STEP 3] Creating distribution overlays...")
        plot_distribution_overlay(real_data, synthetic_data, important_cols, output_dir)

        print("[STEP 4] Creating class distribution comparison...")
        plot_class_distribution_comparison(real_data, synthetic_data, output_dir)

        print("[STEP 5] Creating summary statistics...")
        create_summary_table(real_data, synthetic_data, output_dir)

        print("\n" + "="*60)
        print("VISUALIZATION PIPELINE COMPLETED")
        print("="*60)
        print(f"\nPlots saved to: {output_dir}")
        print("\nGenerated files:")

        for file_path in sorted(output_dir.glob("*")):
            file_size = file_path.stat().st_size / 1024
            print(f"  - {file_path.name} ({file_size:.1f} KB)")

        print("\n[INFO] To download plots in Colab, run:")
        print("  from google.colab import files")
        print("  import zipfile")
        print("  zipfile.ZipFile('plots.zip', 'w').writeall('/content/outputs/plots')")
        print("  files.download('plots.zip')")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please ensure you have run the training script first.")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
