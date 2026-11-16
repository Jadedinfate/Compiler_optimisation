#!/usr/bin/env python3
"""
tabular_model.py
Modular Tabular Machine Learning Model for Compiler Vulnerability Prediction.
Supports ensemble learning (Random Forest & Gradient Boosting), comprehensive evaluation,
feature importance analysis, model serialization, and inference.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.logging.logging import logging
from src.exception.exception import exception_handling
from src.config.config import TrainingConfig


class CompilerVulnerabilityClassifier:
    """
    Supervised Machine Learning Classifier for predicting code vulnerability
    based on static compiler intermediate representation (IR) metrics.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = TrainingConfig.N_ESTIMATORS,
        max_depth: Optional[int] = TrainingConfig.MAX_DEPTH,
        random_state: int = TrainingConfig.RANDOM_STATE,
        **kwargs
    ):
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
                **kwargs
            )
        elif self.model_type in ["gradient_boosting", "gbm"]:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth else 5,
                learning_rate=getattr(TrainingConfig, "LEARNING_RATE", 0.1),
                random_state=self.random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'random_forest' or 'gradient_boosting'.")

        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "CompilerVulnerabilityClassifier":
        """Fits the ML model to training data."""
        try:
            logging.info(f"Training {self.model_type} model with {X.shape[0]} samples and {X.shape[1]} features.")
            self.model.fit(X, y)
            self.is_fitted = True
            logging.info("Model training completed successfully.")
            return self
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise exception_handling(e, sys)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generates binary predictions (0: Benign, 1: Vulnerable)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generates prediction probabilities for [Benign, Vulnerable]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() before predict_proba().")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics on given dataset.
        Returns accuracy, precision, recall, f1, roc_auc, confusion_matrix, and classification report.
        """
        try:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)[:, 1]

            acc = float(accuracy_score(y, y_pred))
            prec = float(precision_score(y, y_pred, zero_division=0))
            rec = float(recall_score(y, y_pred, zero_division=0))
            f1 = float(f1_score(y, y_pred, zero_division=0))

            try:
                auc = float(roc_auc_score(y, y_proba))
            except Exception:
                auc = 0.5

            cm = confusion_matrix(y, y_pred).tolist()
            report = classification_report(y, y_pred, output_dict=True, zero_division=0)

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
                "confusion_matrix": cm,
                "classification_report": report
            }

            logging.info(f"Evaluation metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")
            return metrics

        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise exception_handling(e, sys)

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Returns sorted feature importances for compiler metrics."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            sorted_idx = np.argsort(importances)[::-1]
            return {feature_names[i]: float(importances[i]) for i in sorted_idx}
        return {}

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        """Serializes and saves model to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            payload = {
                "model_type": self.model_type,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "model": self.model,
                "is_fitted": self.is_fitted
            }
            joblib.dump(payload, filepath)
            logging.info(f"Saved model artifact to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model artifact to {filepath}: {e}")
            raise exception_handling(e, sys)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike]) -> "CompilerVulnerabilityClassifier":
        """Loads serialized model from disk."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model artifact not found at {filepath}")
            payload = joblib.load(filepath)
            instance = cls(
                model_type=payload.get("model_type", "random_forest"),
                n_estimators=payload.get("n_estimators", 100),
                max_depth=payload.get("max_depth", None),
                random_state=payload.get("random_state", 42)
            )
            instance.model = payload["model"]
            instance.is_fitted = payload.get("is_fitted", True)
            logging.info(f"Loaded model artifact from {filepath}")
            return instance
        except Exception as e:
            logging.error(f"Error loading model artifact from {filepath}: {e}")
            raise exception_handling(e, sys)
