import os
import sys
import pandas as pd

from src.exception.exception import exception_handling
from src.logging.logging import logging


class DataLoader:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        try:
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