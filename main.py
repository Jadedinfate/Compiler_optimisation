#!/usr/bin/env python3
"""
main.py
End-to-End Machine Learning Pipeline for Compiler Vulnerability Prediction.

Workflow:
1. Data Ingestion: Ingests either full unified dataset (all splits combined, 21,686 samples)
   or single split file via DataLoader.
2. Preprocessing & Validation: Validates features, performs stratified train/val/test split,
   and normalizes input metrics using StandardScaler.
3. Model Training: Trains CompilerVulnerabilityClassifier (Random Forest / Gradient Boosting)
   with class balancing and optimized hyperparameters on compiler IR features.
4. Comprehensive Evaluation: Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC,
   and confusion matrices on Validation and Test sets.
5. Feature Importance Analysis: Ranks importance of compiler IR metrics.
6. Artifact Serialization: Persists trained model and fitted scaler to models/ directory.
7. End-to-End Inference Verification: Reloads artifacts and performs sample inference.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from src.exception.exception import exception_handling
from src.logging.logging import logging
from src.config.config import DataConfig, TrainingConfig
from src.data.data_loader import DataLoader
from src.data.preprocess import DataPreprocessor
from src.models.tabular_model import CompilerVulnerabilityClassifier


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title.center(63)} ")
    print("=" * 65)


def print_metrics_table(split_name: str, metrics: dict):
    print(f"\n--- {split_name.upper()} SET EVALUATION ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix [TN, FP / FN, TP]:")
    cm = metrics["confusion_matrix"]
    print(f"    [[{cm[0][0]:>5}, {cm[0][1]:>5}],")
    print(f"     [{cm[1][0]:>5}, {cm[1][1]:>5}]]")


def run_pipeline(all_splits: bool = True, model_type: str = "random_forest", n_estimators: int = None, max_depth: int = None):
    try:
        logging.info("Starting End-to-End ML Pipeline.")
        print_header("COMPILER VULNERABILITY ML PIPELINE")

        # ----------------------------------------------------------------------
        # 1. Data Ingestion
        # ----------------------------------------------------------------------
        data_loader = DataLoader(str(DataConfig.DATA_PATH))
        if all_splits:
            print("[*] Stage 1: Ingesting full unified dataset across all splits...")
            df = data_loader.load_all_splits(str(DataConfig.CLEAN_SPLITS_DIR))
            print(f"[+] Loaded combined dataset: {df.shape[0]} total rows, {df.shape[1]} columns")
        else:
            print(f"[*] Stage 1: Ingesting dataset from {DataConfig.DATA_PATH}...")
            df = data_loader.load_data()
            print(f"[+] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # ----------------------------------------------------------------------
        # 2. Preprocessing, Splitting & Scaling
        # ----------------------------------------------------------------------
        print("\n[*] Stage 2: Preprocessing and feature engineering...")
        preprocessor = DataPreprocessor(df)
        X, y = preprocessor.split_features_target()

        print(f"    Features : {list(X.columns)}")
        print(f"    Target Distribution:")
        val_counts = y.value_counts().to_dict()
        print(f"      Benign (0)     : {val_counts.get(0, 0)} ({val_counts.get(0, 0)/len(y)*100:.1f}%)")
        print(f"      Vulnerable (1) : {val_counts.get(1, 0)} ({val_counts.get(1, 0)/len(y)*100:.1f}%)")

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = preprocessor.split_data(X, y)

        (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            scaler
        ) = preprocessor.scale_features(
            X_train,
            X_val,
            X_test
        )

        print(f"[+] Partitioned splits:")
        print(f"    Train Set: {X_train_scaled.shape[0]} samples")
        print(f"    Val Set  : {X_val_scaled.shape[0]} samples")
        print(f"    Test Set : {X_test_scaled.shape[0]} samples")

        # ----------------------------------------------------------------------
        # 3. Model Training
        # ----------------------------------------------------------------------
        n_est = n_estimators or TrainingConfig.N_ESTIMATORS
        m_depth = max_depth or TrainingConfig.MAX_DEPTH
        print(f"\n[*] Stage 3: Training CompilerVulnerabilityClassifier ({model_type}, n_estimators={n_est}, max_depth={m_depth})...")
        classifier = CompilerVulnerabilityClassifier(
            model_type=model_type,
            n_estimators=n_est,
            max_depth=m_depth,
            random_state=TrainingConfig.RANDOM_STATE
        )
        classifier.fit(X_train_scaled, y_train)
        print(f"[+] Model training completed successfully.")

        # ----------------------------------------------------------------------
        # 4. Evaluation
        # ----------------------------------------------------------------------
        print("\n[*] Stage 4: Evaluating model performance...")
        val_metrics = classifier.evaluate(X_val_scaled, y_val)
        test_metrics = classifier.evaluate(X_test_scaled, y_test)

        print_metrics_table("Validation", val_metrics)
        print_metrics_table("Test", test_metrics)

        # ----------------------------------------------------------------------
        # 5. Feature Importance Analysis
        # ----------------------------------------------------------------------
        print("\n[*] Stage 5: Analyzing Compiler IR Feature Importances...")
        feature_names = DataConfig.FEATURE_COLUMNS
        importances = classifier.get_feature_importances(feature_names)
        print("  Top Compiler IR Risk Indicators:")
        for rank, (feat, imp) in enumerate(list(importances.items())[:8], 1):
            print(f"    [{rank}] {feat:<16} : {imp * 100:>5.2f}% weight")

        # ----------------------------------------------------------------------
        # 6. Artifact Serialization
        # ----------------------------------------------------------------------
        print("\n[*] Stage 6: Saving model and scaler artifacts...")
        os.makedirs(DataConfig.MODELS_DIR, exist_ok=True)
        classifier.save(DataConfig.TABULAR_MODEL_PATH)
        joblib.dump(scaler, DataConfig.SCALER_PATH)
        print(f"[+] Model saved to  : {DataConfig.TABULAR_MODEL_PATH}")
        print(f"[+] Scaler saved to : {DataConfig.SCALER_PATH}")

        # ----------------------------------------------------------------------
        # 7. End-to-End Inference Verification
        # ----------------------------------------------------------------------
        print("\n[*] Stage 7: Testing inference with saved artifacts...")
        loaded_model = CompilerVulnerabilityClassifier.load(DataConfig.TABULAR_MODEL_PATH)
        loaded_scaler = joblib.load(DataConfig.SCALER_PATH)

        sample_subset = X_test.iloc[:5]
        sample_scaled = loaded_scaler.transform(sample_subset)
        sample_preds = loaded_model.predict(sample_scaled)
        sample_probas = loaded_model.predict_proba(sample_scaled)

        print("  Sample Test Predictions:")
        for i in range(len(sample_subset)):
            true_label = "Vulnerable (1)" if y_test.iloc[i] == 1 else "Benign (0)"
            pred_label = "VULNERABLE (1)" if sample_preds[i] == 1 else "SAFE (0)"
            conf = sample_probas[i][sample_preds[i]] * 100.0
            print(f"    Sample {i+1}: True=[{true_label}] | Predicted=[{pred_label}] with {conf:.2f}% confidence")

        print_header("PIPELINE EXECUTION COMPLETE")
        logging.info("End-to-End ML Pipeline completed successfully.")

    except Exception as e:
        logging.error("ML pipeline execution failed.")
        raise exception_handling(e, sys)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Compiler Optimization ML Pipeline.")
    parser.add_argument("--all_splits", action="store_true", default=True, help="Combine all splits into big dataset (21,686 samples)")
    parser.add_argument("--single_split", action="store_false", dest="all_splits", help="Train only on single train.csv file")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "gradient_boosting"], help="Model type")
    parser.add_argument("--n_estimators", type=int, default=150, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=16, help="Maximum tree depth")
    args = parser.parse_args()

    run_pipeline(
        all_splits=args.all_splits,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )


if __name__ == "__main__":
    main()