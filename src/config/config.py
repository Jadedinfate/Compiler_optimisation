from pathlib import Path

##for dataset
class DataConfig:

    DATA_PATH = Path("dataset/train_features_final.csv")

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


class ProjectConfig:
    PROJECT_NAME = "Compiler_Optimisation"