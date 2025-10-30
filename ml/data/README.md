# ML Data Directory

This directory contains training datasets for the Invoice Processing workflow ML models.

## Expected Datasets

### invoice_risk_training_dataset.csv
Training dataset for invoice risk assessment prediction.

Required columns:
- amount
- vendor_id
- payment_terms
- frequency
- risk_level (target variable: Low, Medium, High)

### invoice_payment_prediction_dataset.csv
Training dataset for payment prediction model.

Required columns:
- amount
- discount_offered
- payment_method
- days_due
- payment_status (target variable: OnTime, Late, Disputed)

## Usage

Place your training datasets here for model training. The system can be enhanced
with ML prediction nodes by adding training scripts to ml/training/.

Current implementation uses primarily rule-based processing for optimal performance
in invoice handling.
