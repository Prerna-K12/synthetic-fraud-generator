"""
Evaluation script for Synthetic Financial Data Generator.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import xgboost as xgb
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import joblib


def load_data():
    """Load both real and synthetic datasets."""
    print("[INFO] Loading datasets...")

    real_data_path = Path("/content/creditcard.csv")
    synthetic_data_path = Path("/content/outputs/synthetic_data.csv")

    print(f"  - Loading real data from: {real_data_path}")
    real_data = pd.read_csv(real_data_path)
    print(f"    Real data shape: {real_data.shape}")

    print(f"  - Loading synthetic data from: {synthetic_data_path}")
    synthetic_data = pd.read_csv(synthetic_data_path)
    print(f"    Synthetic data shape: {synthetic_data.shape}")

    return real_data, synthetic_data


def load_metadata(real_data):
    """Load or recreate metadata for the dataset."""
    print("[INFO] Loading metadata...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    metadata.update_column("Class", sdtype="categorical")

    print(f"  - Metadata created with {len(metadata.columns)} columns")

    return metadata


def run_sdv_quality_report(real_data, synthetic_data, metadata):
    """Run SDV's built-in quality evaluation."""
    print("\n" + "="*60)
    print("PART A: SDV QUALITY REPORT")
    print("="*60)

    print("\n[INFO] Running SDV quality evaluation...")
    print("[INFO] This compares statistical properties of real vs synthetic data\n")

    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=False
    )

    overall_score = quality_report.get_score()

    print("-"*60)
    print("OVERALL QUALITY SCORE")
    print("-"*60)
    print(f"Score: {overall_score:.4f} / 1.0000")

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

    print("\nCOLUMN-WISE SCORES")
    print("-"*60)

    try:
        column_details = quality_report.get_details('Column Pair Trends')
        if 'Score' in column_details.columns:
            avg_trend_score = column_details['Score'].mean()
            print(f"Average Trend Score: {avg_trend_score:.4f}")
            print("  (Measures how well column relationships are preserved)")

        shape_details = quality_report.get_details('Column Shapes')
        if 'Score' in shape_details.columns:
            avg_shape_score = shape_details['Score'].mean()
            print(f"Average Shape Score: {avg_shape_score:.4f}")
            print("  (Measures how well individual column distributions match)")

    except Exception as e:
        print(f"Note: Could not extract detailed scores: {e}")

    print("="*60)

    return {"overall_score": overall_score, "quality_report": quality_report}


def prepare_data_for_tstr(real_data, synthetic_data, target_column="Class"):
    """Prepare data for TSTR evaluation."""
    print("\n[INFO] Preparing data for TSTR evaluation...")

    X_synthetic = synthetic_data.drop(columns=[target_column])
    y_synthetic = synthetic_data[target_column]

    X_real = real_data.drop(columns=[target_column])
    y_real = real_data[target_column]

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.25, random_state=42, stratify=y_real
    )

    print(f"  - Synthetic training samples: {len(X_synthetic):,}")
    print(f"  - Real training samples: {len(X_real_train):,}")
    print(f"  - Real test samples: {len(X_real_test):,}")

    test_fraud_rate = y_real_test.sum() / len(y_real_test) * 100
    print(f"  - Test set fraud rate: {test_fraud_rate:.4f}%")

    return X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier optimized for imbalanced data."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a trained model and print metrics."""
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n{model_name} - Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    print(f"{model_name} - Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                    Predicted")
    print(f"                    Legit     Fraud")
    print(f"  Actual  Legit     {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"          Fraud     {cm[1][0]:5d}    {cm[1][1]:5d}")
    print("-"*60)

    return {"f1": f1, "precision": precision, "recall": recall, "y_pred": y_pred, "confusion_matrix": cm}


def run_tstr_evaluation(X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test):
    """Run TSTR (Train on Synthetic, Test on Real) evaluation."""
    print("\n" + "="*60)
    print("PART B: TSTR EVALUATION (Train on Synthetic, Test on Real)")
    print("="*60)

    print("\n[INFO] TSTR evaluates if synthetic data can train useful models")
    print("[INFO] We train two models and compare their performance on real test data\n")

    # Model 1: Train on SYNTHETIC data
    print("-"*60)
    print("MODEL 1: Training on SYNTHETIC data...")
    print("-"*60)
    synthetic_model = train_xgboost(X_synthetic, y_synthetic)
    synthetic_metrics = evaluate_model(synthetic_model, X_real_test, y_real_test, "Synthetic-Trained Model")

    # Model 2: Train on REAL data (baseline)
    print("\n" + "-"*60)
    print("MODEL 2: Training on REAL data (baseline)...")
    print("-"*60)
    real_model = train_xgboost(X_real_train, y_real_train)
    real_metrics = evaluate_model(real_model, X_real_test, y_real_test, "Real-Trained Model")

    # Compare
    print("\n" + "="*60)
    print("TSTR COMPARISON RESULTS")
    print("="*60)

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

    performance_gap = real_metrics["f1"] - synthetic_metrics["f1"]
    performance_gap_pct = (performance_gap / real_metrics["f1"]) * 100

    print(f"\nKey Findings:")
    print(f"  - F1 Score Gap: {performance_gap:.4f} ({performance_gap_pct:.1f}%)")
    print(f"  - Synthetic model achieves {f1_retention:.1f}% of real model's F1 score")

    print("\nInterpretation:")
    if f1_retention >= 90:
        print("  EXCELLENT: Synthetic data produces nearly equivalent models to real data")
    elif f1_retention >= 80:
        print("  GOOD: Synthetic data produces models with minor performance loss")
    elif f1_retention >= 70:
        print("  ACCEPTABLE: Synthetic data is usable with some performance trade-off")
    else:
        print("  NEEDS IMPROVEMENT: Consider more training epochs or synthetic samples")

    print(f"\n>>> Synthetic model is within {performance_gap_pct:.1f}% of real model <<<")
    print("="*60)

    return {
        "synthetic_metrics": synthetic_metrics,
        "real_metrics": real_metrics,
        "f1_retention": f1_retention,
        "performance_gap": performance_gap,
        "performance_gap_pct": performance_gap_pct
    }


def print_final_summary(sdv_results, tstr_results):
    """Print a final summary of all evaluation results."""
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)

    print("\nSDV Quality Metrics:")
    print(f"  - Overall Quality Score: {sdv_results['overall_score']:.4f}")

    print("\nTSTR Metrics:")
    print(f"  - Synthetic-Trained F1: {tstr_results['synthetic_metrics']['f1']:.4f}")
    print(f"  - Real-Trained F1: {tstr_results['real_metrics']['f1']:.4f}")
    print(f"  - Performance Retention: {tstr_results['f1_retention']:.1f}%")

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
    """Main function to orchestrate the evaluation pipeline."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA EVALUATION PIPELINE")
    print("="*60)

    try:
        print("\n[STEP 1] Loading datasets...")
        real_data, synthetic_data = load_data()

        print("[STEP 2] Loading metadata...")
        metadata = load_metadata(real_data)

        print("[STEP 3] Running SDV quality evaluation...")
        sdv_results = run_sdv_quality_report(real_data, synthetic_data, metadata)

        print("[STEP 4] Preparing data for TSTR...")
        X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test = \
            prepare_data_for_tstr(real_data, synthetic_data)

        print("[STEP 5] Running TSTR evaluation...")
        tstr_results = run_tstr_evaluation(
            X_synthetic, y_synthetic, X_real_train, X_real_test, y_real_train, y_real_test
        )

        print("[STEP 6] Generating final summary...")
        print_final_summary(sdv_results, tstr_results)

        print("\n[INFO] Evaluation completed successfully!")

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
