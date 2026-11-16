#!/usr/bin/env python3
"""
test_pipeline.py
Automated Unit and Integration Tests for Compiler Optimization ML Pipeline.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from src.config.config import DataConfig, TrainingConfig
from src.data.data_loader import DataLoader
from src.data.preprocess import DataPreprocessor
from src.models.tabular_model import CompilerVulnerabilityClassifier
from src.models.gnn_model import CPGGGNNModel, get_model
from src.data.cpg_dataset import build_cpg_from_code


class TestDataLoaderAndPreprocessor(unittest.TestCase):
    """Tests data ingestion and preprocessing modules."""

    def setUp(self):
        # Create a small synthetic dataframe matching DataConfig.FEATURE_COLUMNS
        data = {col: np.random.randint(0, 50, size=50) for col in DataConfig.FEATURE_COLUMNS}
        data[DataConfig.TARGET_COLUMN] = np.random.choice([0, 1], size=50)
        self.df = pd.DataFrame(data)

    def test_preprocessor_split_features_target(self):
        preprocessor = DataPreprocessor(self.df)
        X, y = preprocessor.split_features_target()
        self.assertEqual(X.shape, (50, len(DataConfig.FEATURE_COLUMNS)))
        self.assertEqual(len(y), 50)

    def test_preprocessor_split_and_scale(self):
        preprocessor = DataPreprocessor(self.df)
        X, y = preprocessor.split_features_target()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 50)

        X_train_s, X_val_s, X_test_s, scaler = preprocessor.scale_features(X_train, X_val, X_test)
        self.assertEqual(X_train_s.shape, X_train.shape)
        self.assertEqual(X_val_s.shape, X_val.shape)
        self.assertIsNotNone(scaler)


class TestTabularModel(unittest.TestCase):
    """Tests CompilerVulnerabilityClassifier training, evaluation, saving, and inference."""

    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 12)
        self.y_train = np.random.choice([0, 1], size=100)
        self.X_test = np.random.randn(20, 12)
        self.y_test = np.random.choice([0, 1], size=20)

    def test_train_and_evaluate(self):
        clf = CompilerVulnerabilityClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(self.X_train, self.y_train)
        self.assertTrue(clf.is_fitted)

        preds = clf.predict(self.X_test)
        self.assertEqual(len(preds), 20)
        self.assertTrue(set(preds).issubset({0, 1}))

        probas = clf.predict_proba(self.X_test)
        self.assertEqual(probas.shape, (20, 2))

        metrics = clf.evaluate(self.X_test, self.y_test)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)

    def test_save_and_load(self):
        clf = CompilerVulnerabilityClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(self.X_train, self.y_train)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            clf.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            loaded = CompilerVulnerabilityClassifier.load(tmp_path)
            self.assertTrue(loaded.is_fitted)

            orig_preds = clf.predict(self.X_test)
            loaded_preds = loaded.predict(self.X_test)
            np.testing.assert_array_equal(orig_preds, loaded_preds)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_feature_importances(self):
        clf = CompilerVulnerabilityClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(self.X_train, self.y_train)
        feature_names = [f"f_{i}" for i in range(12)]
        importances = clf.get_feature_importances(feature_names)
        self.assertEqual(len(importances), 12)
        self.assertAlmostEqual(sum(importances.values()), 1.0, places=2)


class TestGNNModelAndCPG(unittest.TestCase):
    """Tests CPG extraction and PyG GGNN model architecture."""

    def test_cpg_extraction(self):
        sample_code = """
        int vulnerable(char *src) {
            char buffer[32];
            strcpy(buffer, src);
            return 0;
        }
        """
        nodes, edges, edge_types = build_cpg_from_code(sample_code)
        self.assertGreater(len(nodes), 0)
        self.assertGreater(len(edges), 0)
        self.assertEqual(len(edges), len(edge_types))

    def test_gnn_model_forward(self):
        model = get_model(in_channels=64, hidden_dim=32, num_layers=2)
        num_nodes = 10
        num_edges = 15

        x = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 3, (num_edges,))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        logits, attn = model(x, edge_index, edge_type=edge_type, batch=batch, return_attention=True)
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(attn.shape, (num_nodes, 1))


if __name__ == "__main__":
    unittest.main()
