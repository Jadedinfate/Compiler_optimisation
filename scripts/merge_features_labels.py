import pandas as pd

FEATURES = "data/devign/cpg_test/features.csv"
METADATA = "data/devign/train_metadata.csv"
OUTPUT = "data/devign/cpg_test/final_features.csv"

features = pd.read_csv(FEATURES)
metadata = pd.read_csv(METADATA)

# Extract sample_id from sample_00000.c
features["sample_id"] = (
    features["filename"]
    .str.extract(r"sample_(\d+)\.c")[0]
)

features["sample_id"] = features["sample_id"].astype(int)

metadata["sample_id"] = metadata["sample_id"].astype(int)

# Add labels
result = features.merge(
    metadata[["sample_id", "target"]],
    on="sample_id",
    how="left"
)

# Put sample_id and target first
cols = ["sample_id", "target"] + [
    c for c in result.columns
    if c not in ["sample_id", "target"]
]

result = result[cols]

result.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Rows:", len(result))
print("Missing labels:", result["target"].isna().sum())
