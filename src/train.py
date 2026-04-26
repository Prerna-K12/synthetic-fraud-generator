"""
Training script for CTGAN-based Synthetic Financial Data Generator.
Optimized for Google Colab with T4 GPU.

Author: ML Engineer
Date: 2026
"""

import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import joblib
import torch


def check_gpu():
    """Check if GPU is available and print info."""
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)

    if torch.cuda.is_available():
        print(f"[OK] CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA version: {torch.version.cuda}")
        print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("[WARNING] CUDA not available - will use CPU")
    print("="*60 + "\n")


def load_data(data_path):
    """Load the credit card fraud detection dataset."""
    print(f"[INFO] Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Please upload creditcard.csv to /content/"
        )

    df = pd.read_csv(data_path)
    print(f"[INFO] Successfully loaded {len(df):,} rows and {len(df.columns)} columns")

    return df


def print_class_distribution(df, target_column="Class"):
    """Print the class distribution to show data imbalance."""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)

    class_counts = df[target_column].value_counts()
    total = len(df)
    class_percentages = (class_counts / total) * 100

    print(f"\nTotal samples: {total:,}")
    print(f"\nClass breakdown:")
    print("-"*40)

    for class_label in sorted(class_counts.index):
        count = class_counts[class_label]
        percentage = class_percentages[class_label]
        label_name = "Fraud" if class_label == 1 else "Legitimate"
        print(f"  Class {class_label} ({label_name:12s}): {count:6,} ({percentage:6.4f}%)")

    if len(class_counts) == 2:
        imbalance_ratio = class_counts[0] / class_counts[1]
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.1f}:1 (Legitimate:Fraud)")
        print(f"  Fraud rate: {class_percentages[1]:.4f}%")

    print("="*60 + "\n")


def create_metadata(df, target_column="Class"):
    """Create SDV metadata for the dataset."""
    print("[INFO] Creating metadata for CTGAN...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column(target_column, sdtype="categorical")

    print(f"[INFO] Marked '{target_column}' as categorical")
    print(f"[INFO] Metadata created with {len(metadata.columns)} columns")

    return metadata


def initialize_ctgan(metadata):
    """Initialize the CTGAN synthesizer with optimal parameters."""
    print("[INFO] Initializing CTGAN synthesizer...")

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=100,
        verbose=True,
        batch_size=500,
        cuda=True,  # Explicitly enable CUDA
    )

    print("[INFO] CTGAN synthesizer initialized with:")
    print("  - epochs: 100")
    print("  - batch_size: 500")
    print("  - verbose: True")
    print("  - cuda: True")

    return synthesizer


def train_model(synthesizer, real_data):
    """Train the CTGAN model on real data."""
    print("\n" + "="*60)
    print("TRAINING CTGAN MODEL")
    print("="*60)
    print("[INFO] Starting model training...")
    print(f"[INFO] Training on {len(real_data):,} samples")
    print("[INFO] Watch for progress updates below...\n")

    # Fit the CTGAN model
    synthesizer.fit(real_data)

    print("\n[INFO] Training completed successfully!")
    print("="*60 + "\n")

    return synthesizer


def save_model(synthesizer, model_path):
    """Save the trained model to disk."""
    print(f"[INFO] Saving model to: {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(synthesizer, model_path)

    print(f"[INFO] Model saved successfully ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")


def generate_synthetic_data(synthesizer, num_rows=5000):
    """Generate synthetic data using the trained CTGAN model."""
    print(f"\n[INFO] Generating {num_rows:,} synthetic rows...")

    synthetic_data = synthesizer.sample(num_rows=num_rows)

    print(f"[INFO] Generated {len(synthetic_data):,} synthetic rows")

    return synthetic_data


def save_synthetic_data(synthetic_data, output_path):
    """Save synthetic data to CSV."""
    print(f"[INFO] Saving synthetic data to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)

    print(f"[INFO] Synthetic data saved ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def print_synthetic_summary(synthetic_data, target_column="Class"):
    """Print summary statistics of the synthetic data."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA SUMMARY")
    print("="*60)

    print("\nFirst 5 rows of synthetic data:")
    print("-"*60)
    print(synthetic_data.head())

    print(f"\nSynthetic data class distribution:")
    print("-"*40)

    class_counts = synthetic_data[target_column].value_counts()
    total = len(synthetic_data)
    class_percentages = (class_counts / total) * 100

    for class_label in sorted(class_counts.index):
        count = class_counts[class_label]
        percentage = class_percentages[class_label]
        label_name = "Fraud" if class_label == 1 else "Legitimate"
        print(f"  Class {class_label} ({label_name:12s}): {count:6,} ({percentage:6.4f}%)")

    if len(class_counts) == 2:
        imbalance_ratio = class_counts[0] / class_counts[1]
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.1f}:1 (Legitimate:Fraud)")

    print("="*60 + "\n")


def main():
    """Main function to orchestrate the training pipeline."""
    print("\n" + "="*60)
    print("CTGAN SYNTHETIC DATA GENERATOR - TRAINING PIPELINE")
    print("="*60)

    # Check GPU availability first
    check_gpu()

    # Define paths for Google Colab
    data_path = Path("/content/creditcard.csv")
    model_path = Path("/content/models/ctgan_model.pkl")
    output_path = Path("/content/outputs/synthetic_data.csv")

    # Create directories
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load the dataset
        print("\n[STEP 1] Loading dataset...")
        real_data = load_data(data_path)

        # Step 2: Analyze class distribution
        print("[STEP 2] Analyzing class distribution...")
        print_class_distribution(real_data)

        # Step 3: Create metadata
        print("[STEP 3] Creating metadata...")
        metadata = create_metadata(real_data)

        # Step 4: Initialize CTGAN
        print("[STEP 4] Initializing CTGAN...")
        synthesizer = initialize_ctgan(metadata)

        # Step 5: Train the model
        print("[STEP 5] Training model...")
        trained_synthesizer = train_model(synthesizer, real_data)

        # Step 6: Save the trained model
        print("[STEP 6] Saving model...")
        save_model(trained_synthesizer, model_path)

        # Step 7: Generate synthetic data
        print("[STEP 7] Generating synthetic data...")
        synthetic_data = generate_synthetic_data(trained_synthesizer, num_rows=5000)

        # Step 8: Save synthetic data
        print("[STEP 8] Saving synthetic data...")
        save_synthetic_data(synthetic_data, output_path)

        # Step 9: Print summary
        print("[STEP 9] Printing synthetic data summary...")
        print_synthetic_summary(synthetic_data)

        # Final summary
        print("="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nOutputs:")
        print(f"  - Trained model: {model_path}")
        print(f"  - Synthetic data: {output_path}")
        print(f"  - Synthetic rows: {len(synthetic_data):,}")

        # Download links for Colab
        print("\n[INFO] To download files in Colab, run:")
        print(f"  from google.colab import files")
        print(f"  files.download('{model_path}')")
        print(f"  files.download('{output_path}')")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nIn Colab, upload the file by running:")
        print("  from google.colab import files")
        print("  uploaded = files.upload()")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
