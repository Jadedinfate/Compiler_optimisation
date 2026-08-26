import os
import shutil
import subprocess
import pandas as pd

BASE = "data/devign"
SOURCE = f"{BASE}/source/train"
FAILED_FILE = "/tmp/still_failed_ids.txt"

WORK = f"{BASE}/failed_retry"
CLEAN = f"{WORK}/clean"
CPG = f"{WORK}/retry.cpg.bin"

os.makedirs(CLEAN, exist_ok=True)

# Read failed IDs
with open(FAILED_FILE) as f:
    ids = [int(x.strip()) for x in f if x.strip()]

print("Failed samples:", len(ids))

# Clean failed files
for sample_id in ids:
    filename = f"sample_{sample_id:05d}.c"

    src = os.path.join(SOURCE, filename)
    dst = os.path.join(CLEAN, filename)

    subprocess.run([
        "python",
        "scripts/clean_source.py",
        src,
        dst
    ], check=True)

# Parse cleaned files
print("\nRunning Joern...")

result = subprocess.run([
    "joern-parse",
    CLEAN,
    "--output",
    CPG
])

if result.returncode != 0:
    print("Joern parsing failed.")
    raise SystemExit(1)

# Extract features
print("\nExtracting features...")

result = subprocess.run([
    "joern",
    CPG,
    "--script",
    "scripts/extract_features.sc"
])

if result.returncode != 0:
    print("Feature extraction failed.")
    raise SystemExit(1)

FEATURE_FILE = f"{BASE}/cpg_test/features.csv"

if not os.path.exists(FEATURE_FILE):
    print("Feature file not found.")
    raise SystemExit(1)

features = pd.read_csv(FEATURE_FILE)

features["sample_id"] = (
    features["filename"]
    .str.extract(r"sample_(\d+)\.c")[0]
)

features = features.dropna(subset=["sample_id"])
features["sample_id"] = features["sample_id"].astype(int)

# Only retain our failed samples
features = features[
    features["sample_id"].isin(ids)
]

# Remove duplicate samples
features = features.drop_duplicates(
    subset=["sample_id"],
    keep="first"
)

print("\n===================================")
print("Retry complete")
print("===================================")

print("Input failures:", len(ids))
print("Recovered:", len(features))

recovered = set(features["sample_id"])
still_failed = sorted(set(ids) - recovered)

print("Still failed:", len(still_failed))

if still_failed:
    print("\nStill failing:")
    print(still_failed)

# Save recovered features separately
output = f"{BASE}/features/recovered_train_features.csv"

features.to_csv(output, index=False)

print("\nSaved:", output)

# Cleanup
shutil.rmtree(WORK)

