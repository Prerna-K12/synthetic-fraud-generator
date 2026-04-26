# Privacy-Preserving Synthetic Financial Data Generator using CTGAN

## Project Description

This project implements a production-ready pipeline for generating synthetic credit card transaction data that preserves the statistical properties of real financial data while protecting sensitive customer information. Using CTGAN (Conditional Tabular Generative Adversarial Network), we can create realistic fraud and legitimate transaction patterns without exposing actual customer data.

The system addresses the critical challenge of severe class imbalance in fraud detection datasets (~0.17% fraud rate) by learning the underlying data distribution and generating synthetic samples that maintain the complex relationships between transaction features. This enables data scientists and ML engineers to safely share, augment, or prototype with realistic financial data without privacy concerns.

## Reference

This implementation is based on the CTGAN paper:

**Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). "Modeling Tabular Data using Conditional GAN". NeurIPS 2019.**

The paper introduced CTGAN as a solution for modeling tabular data with mixed data types and imbalanced categories, which are common challenges in real-world datasets.

## Why CTGAN?

CTGAN outperforms traditional methods for generating synthetic tabular data, especially for imbalanced datasets like fraud detection. Here's why:

### Mode-Specific Normalization

Real-world financial data often has multiple "modes" or patterns. For example, legitimate transactions cluster around typical purchase amounts, while fraud transactions follow different patterns (unusual amounts, odd timestamps, etc.). Traditional methods like simple random sampling or basic statistical models treat all data uniformly, losing these distinct patterns.

**Mode-Specific Normalization** automatically identifies these different clusters in your data and normalizes each mode separately. This means:
- Legitimate transactions are modeled with their own statistical pattern
- Fraud transactions are modeled with their distinct pattern
- The generator learns both patterns independently, preventing the majority class from overwhelming the minority class

In plain English: Instead of forcing all transactions into one "average" pattern, CTGAN recognizes that fraud and legitimate transactions are fundamentally different and models each type appropriately.

### Conditional Generator

The **Conditional Generator** allows CTGAN to explicitly control the class balance during generation. When you specify "generate 500 rows with Class=1 (fraud)", the generator produces realistic fraud patterns on demand.

This solves two critical problems:
1. **Class Imbalance**: You can generate more synthetic fraud samples than exist in the original data, creating a balanced dataset for training ML models
2. **Targeted Generation**: You can generate specific scenarios (e.g., "high-amount fraud transactions") for stress-testing fraud detection systems

In plain English: Think of it like a "dial" that lets you say "make more fraud samples" or "make more legitimate samples" while ensuring each sample still looks realistic.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Navigate to the project directory:
   ```bash
   cd synthetic-fraud-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset:
   - Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
   - Place the file at `./data/creditcard.csv`

## How to Run

Execute the scripts in the following order:

### Step 1: Train the CTGAN Model and Generate Synthetic Data

```bash
python src/train.py
```

This script will:
- Load the credit card fraud dataset
- Display the class distribution (showing the imbalance)
- Train the CTGAN model for 100 epochs
- Save the trained model to `models/ctgan_model.pkl`
- Generate 5,000 synthetic rows
- Save synthetic data to `outputs/synthetic_data.csv`

**Expected runtime:** 10-30 minutes (depending on hardware)

### Step 2: Evaluate Synthetic Data Quality

```bash
python src/evaluate.py
```

This script will:
- Run SDV's quality evaluation report
- Perform TSTR (Train on Synthetic, Test on Real) evaluation
- Compare synthetic-trained vs real-trained model performance
- Display classification metrics and confusion matrices

**Expected runtime:** 5-15 minutes

### Step 3: Visualize Results

```bash
python src/visualize.py
```

This script will:
- Generate distribution comparison plots for key features
- Create class distribution bar charts
- Save all visualizations to `outputs/plots/`

**Expected runtime:** 1-3 minutes

## Project Structure

```
synthetic-fraud-generator/
├── data/
│   └── creditcard.csv          # Input: Original dataset (user-provided)
├── outputs/
│   ├── synthetic_data.csv      # Output: Generated synthetic data
│   └── plots/                  # Output: Visualization files
│       ├── amount_distribution.png
│       ├── time_distribution.png
│       ├── v1_distribution.png
│       ├── v2_distribution.png
│       ├── v3_distribution.png
│       └── class_comparison.png
├── models/
│   └── ctgan_model.pkl         # Output: Trained CTGAN model
├── src/
│   ├── train.py                # Training and generation script
│   ├── evaluate.py             # Quality evaluation script
│   └── visualize.py            # Visualization script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Example Output Metrics

After running all scripts, you should see metrics similar to:

| Metric | Value | Description |
|--------|-------|-------------|
| Overall Quality Score | 0.85-0.95 | SDV quality report (1.0 = perfect) |
| Shape Score | 0.90-0.98 | How well column shapes match |
| Trend Score | 0.80-0.95 | How well relationships match |
| TSTR F1 (Synthetic-trained) | 0.70-0.85 | Model trained on synthetic data |
| TSTR F1 (Real-trained) | 0.85-0.95 | Model trained on real data (baseline) |
| Performance Retention | 80-90% | Synthetic model as % of real model |

**Note:** Actual values depend on training epochs, random seeds, and dataset characteristics.

## Troubleshooting

### CUDA/GPU Support
If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch for faster training:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
If you encounter memory errors during training, reduce the batch size in `src/train.py`:
```python
CTGANSynthesizer(metadata, epochs=100, verbose=True, batch_size=250)
```

### Low Quality Scores
If quality scores are below 0.7:
1. Increase training epochs to 200-500
2. Ensure the dataset has sufficient samples (>10,000 rows)
3. Check that the "Class" column is properly marked as categorical

## License

This project is provided for educational and research purposes. The Credit Card Fraud Detection dataset is subject to Kaggle's terms of use.
