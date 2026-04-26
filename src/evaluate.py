"""
Evaluation script for Synthetic Financial Data Generator.

This script evaluates the quality of synthetic data using two approaches:
1. SDV Quality Report - Statistical comparison of real vs synthetic data
2. TSTR (Train on Synthetic, Test on Real) - ML utility evaluation

The goal is to verify that synthetic data preserves statistical properties
and can be used to train effective machine learning models.

Author: ML Engineer
Date: 2026
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import xgboost as xgb
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


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

    # Define file paths
    real_data_path = project_root / "data" / "creditcard.csv"
    synthetic_data_path = project_root / "outputs" / "synthetic_data.csv"

    # Load real data
    print(f"  - Loading real data from: {real_data_path}")
    real_data = pd.read_csv(real_data_path)
    print(f"    Real data shape: {real_data.shape}")

    # Load synthetic data
    print(f"  - Loading synthetic data from: {synthetic_data_path}")
    synthetic_data = pd.read_csv(synthetic_data_path)
    print(f"    Synthetic data shape: {synthetic_data.shape}")

    return real_data, synthetic_data


def load_metadata(project_root):
    """
    Load or recreate metadata for the dataset.

    Args:
        project_root (Path): Path to project root

    Returns:
        SingleTableMetadata: Metadata object
    """
    print("[INFO] Loading metadata...")

    # Load real data to recreate metadata
    real_data_path = project_root / "data" / "creditcard.csv"
    real_data = pd.read_csv(real_data_path)

    # Create metadata from dataframe
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    metadata.update_column("Class", sdtype="categorical")

    print(f"  - Metadata created with {len(metadata.columns)} columns")

    return metadata


def run_sdv_quality_report(real_data, synthetic_data, metadata):
    """
    Run SDV's built-in quality evaluation.

    This evaluates how well the synthetic data matches the real data
    in terms of column shapes and data trends.

    Args:
        real_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Generated dataset
        metadata (SingleTableMetadata): Dataset metadata

    Returns:
        dict: Quality scores
    """
    print("\n" + "="*60)
    print("PART A: SDV QUALITY REPORT")
    print("="*60)

    print("\n[INFO] Running SDV quality evaluation...")
    print("[INFO] This compares statistical properties of real vs synthetic data\n")

    # Run the SDV quality evaluation
    # This computes shape scores (column distributions) and trend_scores (relationships)
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=False
    )

    # Get the overall quality score (0-1, higher is better)
    overall_score = quality_report.get_score()

    print("-"*60)
    print("OVERALL QUALITY SCORE")
    print("-"*60)
    print(f"Score: {overall_score:.4f} / 1.0000")

    # Interpret the score
    if overall_score >= 0.9:
        interpretation = "EXCELLENT - Synthetic data closely matches real data"
    elif overall_score >= 0.8:
        interpretation = "GOOD - Synthetic data is suitable for most purposes"
    elif overall_score >= 0.7:
        interpretation = "ACCEPTABLE - Some differences exist but data is usable"
    else:
        interpretation = "NEEDS IMPROVEMENT - Consider retraining with more epochs"

    print(f"Interpretation: {interpretation}")
    print("-"*60)

    # Get detailed column-wise scores
    print("\nCOLUMN-WISE SCORES")
    print("-"*60)

    # Extract property details from the report
    try:
        # Get the column pair details for trend scores
        column_details = quality_report.get_details('Column Pair Trends')

        # Calculate average shape and trend scores
        if 'Score' in column_details.columns:
            avg_trend_score = column_details['Score'].mean()
            print(f"Average Trend Score: {avg_trend_score:.4f}")
            print("  (Measures how well column relationships are preserved)")

        # Get column shape scores
        shape_details = quality_report.get_details('Column Shapes')
        if 'Score' in shape_details.columns:
            avg_shape_score = shape_details['Score'].mean()
            print(f"Average Shape Score: {avg_shape_score:.4f}")
            print("  (Measures how well individual column distributions match)")

    except Exception as e:
        print(f"Note: Could not extract detailed scores: {e}")
        print("Overall score is still valid.")

    print("="*60)

    return {
        "overall_score": overall_score,
        "quality_report": quality_report
    }


def prepare_data_for_tstr(real_data, synthetic_data, target_column="Class"):
    """
    Prepare data for TSTR evaluation.

    Splits real data into train/test sets and uses synthetic data as
    an alternative training set.

    Args:
        real_data (pd.DataFrame): Original dataset
        synthetic_data (pd.DataFrame): Generated dataset
        target_column (str): Target column name

    Returns:
        tuple: (X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test)
    """
    print("\n[INFO] Preparing data for TSTR evaluation...")

    # Separate features and target variable for synthetic data
    # Features: all columns except the target 'Class'
    X_synthetic = synthetic_data.drop(columns=[target_column])
    y_synthetic = synthetic_data[target_column]

    # Split real data into training and testing sets
    # We use 80% for training, 20% for testing (held-out evaluation)
    X_real = real_data.drop(columns=[target_column])
    y_real = real_data[target_column]

    # Split with stratification to maintain class distribution
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real,
        y_real,
        test_size=0.2,           # 20% for testing
        random_state=42,         # For reproducibility
        stratify=y_real,         # Maintain class distribution in splits
    )

    print(f"  - Synthetic training samples: {len(X_synthetic):,}")
    print(f"  - Real training samples: {len(X_real_train):,}")
    print(f"  - Real test samples: {len(X_real_test):,}")

    # Print class distribution in test set
    test_fraud_rate = y_real_test.sum() / len(y_real_test) * 100
    print(f"  - Test set fraud rate: {test_fraud_rate:.4f}%")

    return X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test


def train_xgboost(X_train, y_train, target_column="Class"):
    """
    Train an XGBoost classifier optimized for imbalanced data.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        target_column (str): Target column name

    Returns:
        xgb.XGBClassifier: Trained model
    """
    # Calculate scale_pos_weight for handling class imbalance
    # This tells XGBoost how much more important positive class is
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # Create XGBoost classifier with tuned parameters
    model = xgb.XGBClassifier(
        n_estimators=100,           # Number of trees
        max_depth=6,                # Maximum tree depth
        learning_rate=0.1,          # Step size shrinkage
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,            # For reproducibility
        eval_metric='logloss',      # Evaluation metric
        use_label_encoder=False,    # Suppress warning
    )

    # Train the model
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and print metrics.

    Args:
        model: Trained classifier
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_name (str): Name for display

    Returns:
        dict: Dictionary of metrics
    """
    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate key metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n{model_name} - Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Print confusion matrix
    print(f"{model_name} - Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                    Predicted")
    print(f"                    Legit     Fraud")
    print(f"  Actual  Legit     {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"          Fraud     {cm[1][0]:5d}    {cm[1][1]:5d}")
    print("-"*60)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "y_pred": y_pred,
        "confusion_matrix": cm
    }


def run_tstr_evaluation(X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test):
    """
    Run TSTR (Train on Synthetic, Test on Real) evaluation.

    This evaluates whether synthetic data can be used to train models
    that perform well on real data.

    Args:
        X_synthetic, y_synthetic: Synthetic training data
        X_real_train, y_real_train: Real training data
        X_real_test, y_real_test: Real test data (held-out)

    Returns:
        dict: Comparison metrics
    """
    print("\n" + "="*60)
    print("PART B: TSTR EVALUATION (Train on Synthetic, Test on Real)")
    print("="*60)

    print("\n[INFO] TSTR evaluates if synthetic data can train useful models")
    print("[INFO] We train two models and compare their performance on real test data\n")

    # Model 1: Train on SYNTHETIC data, test on REAL data
    print("-"*60)
    print("MODEL 1: Training on SYNTHETIC data...")
    print("-"*60)
    synthetic_model = train_xgboost(X_synthetic, y_synthetic)
    synthetic_metrics = evaluate_model(synthetic_model, X_real_test, y_real_test, "Synthetic-Trained Model")

    # Model 2: Train on REAL data, test on REAL data (baseline)
    print("\n" + "-"*60)
    print("MODEL 2: Training on REAL data (baseline)...")
    print("-"*60)
    real_model = train_xgboost(X_real_train, y_real_train)
    real_metrics = evaluate_model(real_model, X_real_test, y_real_test, "Real-Trained Model")

    # Compare the two models
    print("\n" + "="*60)
    print("TSTR COMPARISON RESULTS")
    print("="*60)

    # Calculate performance retention
    f1_retention = (synthetic_metrics["f1"] / real_metrics["f1"]) * 100
    precision_retention = (synthetic_metrics["precision"] / real_metrics["precision"]) * 100
    recall_retention = (synthetic_metrics["recall"] / real_metrics["recall"]) * 100

    print(f"\nMetric Comparison:")
    print("-"*60)
    print(f"{'Metric':<15} {'Synthetic':<12} {'Real':<12} {'Retention':<12}")
    print("-"*60)
    print(f"{'F1 Score':<15} {synthetic_metrics['f1']:<12.4f} {real_metrics['f1']:<12.4f} {f1_retention:<12.1f}%")
    print(f"{'Precision':<15} {synthetic_metrics['precision']:<12.4f} {real_metrics['precision']:<12.4f} {precision_retention:<12.1f}%")
    print(f"{'Recall':<15} {synthetic_metrics['recall']:<12.4f} {real_metrics['recall']:<12.4f} {recall_retention:<12.1f}%")
    print("-"*60)

    # Performance gap calculation
    performance_gap = real_metrics["f1"] - synthetic_metrics["f1"]
    performance_gap_pct = (performance_gap / real_metrics["f1"]) * 100

    print(f"\nKey Findings:")
    print(f"  - F1 Score Gap: {performance_gap:.4f} ({performance_gap_pct:.1f}%)")
    print(f"  - Synthetic model achieves {f1_retention:.1f}% of real model's F1 score")

    # Interpretation
    print("\nInterpretation:")
    if f1_retention >= 90:
        print("  EXCELLENT: Synthetic data produces nearly equivalent models to real data")
    elif f1_retention >= 80:
        print("  GOOD: Synthetic data produces models with minor performance loss")
    elif f1_retention >= 70:
        print("  ACCEPTABLE: Synthetic data is usable with some performance trade-off")
    else:
        print("  NEEDS IMPROVEMENT: Consider more training epochs or synthetic samples")

    print(f"\n  >>> Synthetic model is within {performance_gap_pct:.1f}% of real model <<<")

    print("="*60)

    return {
        "synthetic_metrics": synthetic_metrics,
        "real_metrics": real_metrics,
        "f1_retention": f1_retention,
        "performance_gap": performance_gap,
        "performance_gap_pct": performance_gap_pct
    }


def print_final_summary(sdv_results, tstr_results):
    """
    Print a final summary of all evaluation results.

    Args:
        sdv_results (dict): SDV quality report results
        tstr_results (dict): TSTR evaluation results
    """
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)

    print("\nSDV Quality Metrics:")
    print(f"  - Overall Quality Score: {sdv_results['overall_score']:.4f}")

    print("\nTSTR Metrics:")
    print(f"  - Synthetic-Trained F1: {tstr_results['synthetic_metrics']['f1']:.4f}")
    print(f"  - Real-Trained F1: {tstr_results['real_metrics']['f1']:.4f}")
    print(f"  - Performance Retention: {tstr_results['f1_retention']:.1f}%")

    # Overall assessment
    print("\nOverall Assessment:")
    print("-"*60)

    quality = sdv_results['overall_score']
    retention = tstr_results['f1_retention']

    if quality >= 0.85 and retention >= 85:
        print("  STATUS: EXCELLENT")
        print("  Synthetic data is production-ready for most use cases")
    elif quality >= 0.75 and retention >= 75:
        print("  STATUS: GOOD")
        print("  Synthetic data is suitable for development and testing")
    elif quality >= 0.65 and retention >= 65:
        print("  STATUS: ACCEPTABLE")
        print("  Synthetic data can be used with noted limitations")
    else:
        print("  STATUS: NEEDS IMPROVEMENT")
        print("  Consider retraining with more epochs or hyperparameter tuning")

    print("="*60 + "\n")


def main():
    """
    Main function to orchestrate the evaluation pipeline.

    This function:
    1. Loads real and synthetic data
    2. Runs SDV quality evaluation
    3. Runs TSTR evaluation
    4. Prints final summary
    """
    print("\n" + "="*60)
    print("SYNTHETIC DATA EVALUATION PIPELINE")
    print("="*60)

    # Get the project root directory
    project_root = get_project_root()

    try:
        # Step 1: Load datasets
        print("\n[STEP 1] Loading datasets...")
        real_data, synthetic_data = load_data(project_root)

        # Step 2: Load/recreate metadata
        print("[STEP 2] Loading metadata...")
        metadata = load_metadata(project_root)

        # Step 3: Run SDV quality evaluation
        print("[STEP 3] Running SDV quality evaluation...")
        sdv_results = run_sdv_quality_report(real_data, synthetic_data, metadata)

        # Step 4: Prepare data for TSTR
        print("[STEP 4] Preparing data for TSTR...")
        X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test = \
            prepare_data_for_tstr(real_data, synthetic_data)

        # Step 5: Run TSTR evaluation
        print("[STEP 5] Running TSTR evaluation...")
        tstr_results = run_tstr_evaluation(
            X_synthetic, y_synthetic,
            X_real_train, X_real_test,
            y_real_train, y_real_test
        )

        # Step 6: Print final summary
        print("[STEP 6] Generating final summary...")
        print_final_summary(sdv_results, tstr_results)

        print("\n[INFO] Evaluation completed successfully!")
        print("[INFO] Next step: Run 'python src/visualize.py' for visual analysis\n")

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
