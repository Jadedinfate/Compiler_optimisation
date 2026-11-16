import os
import sys
from typing import Optional, Tuple
import pandas as pd

from src.exception.exception import exception_handling
from src.logging.logging import logging


class DataLoader:

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from self.data_path."""
        try:
            if not self.data_path:
                raise ValueError("data_path must be specified to call load_data().")
            logging.info(f"Attempting to load dataset from: {self.data_path}")
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    f"Dataset file not found: {self.data_path}"
                )
            df = pd.read_csv(self.data_path)

            logging.info(
                f"Dataset loaded successfully. Shape: {df.shape}"
            )
            return df

        except Exception as e:
            logging.error(f"Failed to load dataset from {self.data_path}")
            raise exception_handling(e, sys)

    def load_all_splits(self, splits_dir: str = "data/clean_splits") -> pd.DataFrame:
        """Loads and combines all split files (train, val, test) into a single unified large dataset."""
        try:
            logging.info(f"Loading and combining all splits from: {splits_dir}")
            split_files = [
                os.path.join(splits_dir, f)
                for f in ["train.csv", "val.csv", "test.csv"]
                if os.path.exists(os.path.join(splits_dir, f))
            ]

            if not split_files:
                raise FileNotFoundError(f"No split CSV files found in {splits_dir}")

            dfs = []
            for f in split_files:
                sub_df = pd.read_csv(f)
                dfs.append(sub_df)
                logging.info(f"Loaded {f}: {sub_df.shape}")

            combined_df = pd.concat(dfs, ignore_index=True)
            # Deduplicate by sample_id if present
            if "sample_id" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

            logging.info(f"Combined dataset total shape: {combined_df.shape}")
            return combined_df

        except Exception as e:
            logging.error(f"Failed to load all splits from {splits_dir}")
            raise exception_handling(e, sys)

    def load_predefined_splits(self, splits_dir: str = "data/clean_splits") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train.csv, val.csv, and test.csv as separate DataFrames."""
        try:
            train_path = os.path.join(splits_dir, "train.csv")
            val_path = os.path.join(splits_dir, "val.csv")
            test_path = os.path.join(splits_dir, "test.csv")

            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Loaded train ({train_df.shape}), val ({val_df.shape}), test ({test_df.shape})")
            return train_df, val_df, test_df

        except Exception as e:
            logging.error(f"Failed to load predefined splits from {splits_dir}")
            raise exception_handling(e, sys)