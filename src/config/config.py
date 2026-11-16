from pathlib import Path

##for dataset
class DataConfig:

    DATA_DIR = Path("data")
    CLEAN_SPLITS_DIR = Path("data/clean_splits")
    DATA_PATH = Path("data/clean_splits/train.csv")
    MODELS_DIR = Path("models")
    TABULAR_MODEL_PATH = Path("models/tabular_model.joblib")
    SCALER_PATH = Path("models/scaler.joblib")

    TARGET_COLUMN = "target"

    FEATURE_COLUMNS = [
        "ast_nodes",
        "cfg_nodes",
        "ast_depth",
        "branches",
        "loops",
        "assignments",
        "returns",
        "function_calls",
        "operators",
        "pointer_ops",
        "array_accesses",
        "field_accesses",
    ]

##for training and splitting
class TrainingConfig:
    TEST_SIZE = 0.10
    VALIDATION_SIZE = 0.10
    RANDOM_STATE = 42
    N_ESTIMATORS = 150
    MAX_DEPTH = 16
    LEARNING_RATE = 0.08
    USE_ALL_SPLITS = True


class ProjectConfig:
    PROJECT_NAME = "Compiler_Optimisation"