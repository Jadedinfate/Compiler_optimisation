import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config.config import DataConfig, TrainingConfig
from src.exception.exception import exception_handling
from src.logging.logging import logging


class DataPreprocessor:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_data(self):
        """
        Validate that the dataset contains all required
        feature and target columns and has no missing values.
        """

        try:
            logging.info("Starting dataset validation.")

            required_columns = (DataConfig.FEATURE_COLUMNS+ [DataConfig.TARGET_COLUMN])

            missing_columns = [
                column
                for column in required_columns
                if column not in self.df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            ml_columns = (DataConfig.FEATURE_COLUMNS+ [DataConfig.TARGET_COLUMN])

            ml_data = self.df[ml_columns]

            if ml_data.isnull().any().any():

                missing_values = ml_data.isnull().sum()

                missing_values = missing_values[
                    missing_values > 0
                ]

                raise ValueError(
                    f"Missing values found:\n{missing_values}"
                )

            logging.info("Dataset validation completed successfully.")

        except Exception as e:

            logging.error("Dataset validation failed.")
            raise exception_handling(e, sys)

    def split_features_target(self):
        """
        Separate the dataset into input features (X)
        and target variable (y).
        """

        try:
            logging.info("Starting feature-target separation.")

            self.validate_data()

            X = self.df[DataConfig.FEATURE_COLUMNS].copy()

            y = self.df[DataConfig.TARGET_COLUMN].copy()

            logging.info(f"Features shape: {X.shape}")
            logging.info(f"Target shape: {y.shape}")
            return X, y

        except Exception as e:

            logging.error("Failed to separate features and target.")
            raise exception_handling(e, sys)

    def split_data(self, X, y):
        """
        Split the dataset into:

        80% training
        10% validation
        10% testing
        """

        try:
            logging.info(
                "Starting train-validation-test split."
            )
            X_train, X_temp, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=(TrainingConfig.VALIDATION_SIZE+ TrainingConfig.TEST_SIZE),
                random_state=TrainingConfig.RANDOM_STATE,
                stratify=y
            )

            test_ratio = (
                TrainingConfig.TEST_SIZE/ (TrainingConfig.VALIDATION_SIZE+ TrainingConfig.TEST_SIZE)
            )

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=test_ratio,
                random_state=TrainingConfig.RANDOM_STATE,
                stratify=y_temp
            )

            logging.info(f"Training set shape: {X_train.shape}")

            logging.info(f"Validation set shape: {X_val.shape}")

            logging.info(f"Test set shape: {X_test.shape}")

            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            )

        except Exception as e:

            logging.error("Failed during dataset splitting.")
            raise exception_handling(e, sys)

    def scale_features(self, X_train, X_val, X_test):
        try:
            logging.info("Starting feature scaling.")

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_val_scaled = scaler.transform(
                X_val
            )

            X_test_scaled = scaler.transform(
                X_test
            )

            logging.info(
                "Feature scaling completed successfully."
            )

            return (
                X_train_scaled,
                X_val_scaled,
                X_test_scaled,
                scaler
            )

        except Exception as e:

            logging.error(
                "Feature scaling failed."
            )

            raise exception_handling(e, sys)