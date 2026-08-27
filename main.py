import sys
from src.exception.exception import exception_handling
from src.logging.logging import logging

from src.config.config import DataConfig
from src.data.data_loader import DataLoader
from src.data.preprocess import DataPreprocessor


if __name__ == "__main__":
    try:
        logging.info("ML pipeline started.")

        data_loader = DataLoader(DataConfig.DATA_PATH)
        df = data_loader.load_data()
        logging.info(f"Dataset shape: {df.shape}")
        print("\nDataset shape:")
        print(df.shape)

        preprocessor = DataPreprocessor(df)

        X, y = preprocessor.split_features_target()

        logging.info(
            f"Features shape: {X.shape}"
        )

        logging.info(
            f"Target shape: {y.shape}"
        )

        print("\nFeatures shape:")
        print(X.shape)

        print("\nTarget shape:")
        print(y.shape)

        print("\nClass distribution:")
        print(y.value_counts())

        logging.info(
            f"Class distribution:\n{y.value_counts()}"
        )

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

        print("\nFinal dataset shapes:")

        print("X_train:", X_train_scaled.shape)
        print("X_val:", X_val_scaled.shape)
        print("X_test:", X_test_scaled.shape)

        print("y_train:", y_train.shape)
        print("y_val:", y_val.shape)
        print("y_test:", y_test.shape)

        logging.info("ML data pipeline completed successfully.")

    except Exception as e:
        logging.error("ML pipeline failed.")
        raise exception_handling(e, sys)