"""
Training script for CTGAN-based Synthetic Financial Data Generator.

This script loads the Credit Card Fraud Detection dataset, trains a CTGAN model,
and generates synthetic data that preserves the statistical properties of the
original data while protecting sensitive information.

Author: ML Engineer
Date: 2026
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import joblib


def get_project_root():
    """
    Get the project root directory.

    Returns:
        Path: Absolute path to the project root directory
    """
    return Path(__file__).parent.parent


def load_data(data_path):
    """
    Load the credit card fraud detection dataset.

    Args:
        data_path (Path): Path to the creditcard.csv file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    print(f"[INFO] Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Please download the Credit Card Fraud Detection dataset from Kaggle\n"
            "and place it at ./data/creditcard.csv"
        )

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(data_path)

    print(f"[INFO] Successfully loaded {len(df):,} rows and {len(df.columns)} columns")

    return df


def print_class_distribution(df, target_column="Class"):
    """
    Print the class distribution to show data imbalance.

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of the target column
    """
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)

    # Count occurrences of each class
    class_counts = df[target_column].value_counts()

    # Calculate percentages
    total = len(df)
    class_percentages = (class_counts / total) * 100

    print(f"\nTotal samples: {total:,}")
    print(f"\nClass breakdown:")
    print("-"*40)

    # Display counts and percentages for each class
    for class_label in sorted(class_counts.index):
        count = class_counts[class_label]
        percentage = class_percentages[class_label]
        label_name = "Fraud" if class_label == 1 else "Legitimate"
        print(f"  Class {class_label} ({label_name:12s}): {count:6,} ({percentage:6.4f}%)")

    # Calculate and display imbalance ratio
    if len(class_counts) == 2:
        imbalance_ratio = class_counts[0] / class_counts[1]
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.1f}:1 (Legitimate:Fraud)")
        print(f"  Fraud rate: {(class_percentages[1]):.4f}%")

    print("="*60 + "\n")


def create_metadata(df, target_column="Class"):
    """
    Create SDV metadata for the dataset.

    Args:
        df (pd.DataFrame): Input dataset
        target_column (str): Name of the target column

    Returns:
        SingleTableMetadata: Metadata object for SDV
    """
    print("[INFO] Creating metadata for CTGAN...")

    # Initialize a new SingleTableMetadata object
    metadata = SingleTableMetadata()

    # Detect column types and relationships automatically from the data
    metadata.detect_from_dataframe(df)

    # Mark the target column as categorical (important for CTGAN)
    # This tells CTGAN to treat Class as a discrete category, not continuous
    metadata.update_column(target_column, sdtype="categorical")

    print(f"[INFO] Marked '{target_column}' as categorical")
    print(f"[INFO] Metadata created with {len(metadata.columns)} columns")

    return metadata


def initialize_ctgan(metadata):
    """
    Initialize the CTGAN synthesizer with optimal parameters.

    Args:
        metadata (SingleTableMetadata): Metadata for the dataset

    Returns:
        CTGANSynthesizer: Configured CTGAN instance
    """
    print("[INFO] Initializing CTGAN synthesizer...")

    # Create CTGAN synthesizer with tuned hyperparameters
    # - epochs=100: Number of training iterations (more = better quality, slower)
    # - verbose=True: Show training progress
    # - batch_size=500: Samples per gradient update (balance memory/speed)
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=100,        # Training iterations
        verbose=True,      # Show progress bar
        batch_size=500,    # Mini-batch size for training
    )

    print("[INFO] CTGAN synthesizer initialized with:")
    print("  - epochs: 100")
    print("  - batch_size: 500")
    print("  - verbose: True")

    return synthesizer


def train_model(synthesizer, real_data):
    """
    Train the CTGAN model on real data.

    Args:
        synthesizer (CTGANSynthesizer): CTGAN instance
        real_data (pd.DataFrame): Training data

    Returns:
        CTGANSynthesizer: Trained synthesizer
    """
    print("\n" + "="*60)
    print("TRAINING CTGAN MODEL")
    print("="*60)
    print("[INFO] Starting model training...")
    print("[INFO] This may take 10-30 minutes depending on your hardware\n")

    # Fit the CTGAN model on the real data
    # CTGAN learns the underlying distribution of the data
    synthesizer.fit(real_data)

    print("\n[INFO] Training completed successfully!")
    print("="*60 + "\n")

    return synthesizer


def save_model(synthesizer, model_path):
    """
    Save the trained model to disk.

    Args:
        synthesizer (CTGANSynthesizer): Trained model
        model_path (Path): Path to save the model
    """
    print(f"[INFO] Saving model to: {model_path}")

    # Ensure the models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained synthesizer using joblib for efficient serialization
    joblib.dump(synthesizer, model_path)

    print(f"[INFO] Model saved successfully ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")


def generate_synthetic_data(synthesizer, num_rows=5000):
    """
    Generate synthetic data using the trained CTGAN model.

    Args:
        synthesizer (CTGANSynthesizer): Trained model
        num_rows (int): Number of rows to generate

    Returns:
        pd.DataFrame: Generated synthetic data
    """
    print(f"\n[INFO] Generating {num_rows:,} synthetic rows...")

    # Generate synthetic data
    # CTGAN creates new rows that follow the learned distribution
    synthetic_data = synthesizer.sample(num_rows=num_rows)

    print(f"[INFO] Generated {len(synthetic_data):,} synthetic rows")

    return synthetic_data


def save_synthetic_data(synthetic_data, output_path):
    """
    Save synthetic data to CSV.

    Args:
        synthetic_data (pd.DataFrame): Generated data
        output_path (Path): Path to save the data
    """
    print(f"[INFO] Saving synthetic data to: {output_path}")

    # Ensure the outputs directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV without index column
    synthetic_data.to_csv(output_path, index=False)

    print(f"[INFO] Synthetic data saved ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def print_synthetic_summary(synthetic_data, target_column="Class"):
    """
    Print summary statistics of the synthetic data.

    Args:
        synthetic_data (pd.DataFrame): Generated data
        target_column (str): Target column name
    """
    print("\n" + "="*60)
    print("SYNTHETIC DATA SUMMARY")
    print("="*60)

    # Display first few rows
    print("\nFirst 5 rows of synthetic data:")
    print("-"*60)
    print(synthetic_data.head())

    # Display class distribution
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
    """
    Main function to orchestrate the training pipeline.

    This function:
    1. Loads the real data
    2. Analyzes class distribution
    3. Creates metadata
    4. Initializes and trains CTGAN
    5. Generates synthetic data
    6. Saves model and data
    """
    print("\n" + "="*60)
    print("CTGAN SYNTHETIC DATA GENERATOR - TRAINING PIPELINE")
    print("="*60)

    # Get the project root directory for relative path calculations
    project_root = get_project_root()

    # Define all file paths using pathlib for cross-platform compatibility
    data_path = project_root / "data" / "creditcard.csv"
    model_path = project_root / "models" / "ctgan_model.pkl"
    output_path = project_root / "outputs" / "synthetic_data.csv"

    try:
        # Step 1: Load the dataset
        print("\n[STEP 1] Loading dataset...")
        real_data = load_data(data_path)

        # Step 2: Analyze and display class distribution
        print("[STEP 2] Analyzing class distribution...")
        print_class_distribution(real_data)

        # Step 3: Create metadata for CTGAN
        print("[STEP 3] Creating metadata...")
        metadata = create_metadata(real_data)

        # Step 4: Initialize CTGAN synthesizer
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

        # Step 9: Print summary of generated data
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
        print("\nNext step: Run 'python src/evaluate.py' to evaluate data quality\n")

    except FileNotFoundError as e:
        # Handle missing dataset file
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    except Exception as e:
        # Handle any other errors
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please check the error message above and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Execute the main function when script is run directly
    main()
