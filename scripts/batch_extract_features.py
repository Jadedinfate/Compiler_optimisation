import os
import sys
import shutil
import subprocess
import pandas as pd

BASE = "data/devign"
BATCH_SIZE = 500
MAX_BATCHES = None  # TEST ONLY: change to None later

if len(sys.argv) != 2 or sys.argv[1] not in ["train", "val", "test"]:
    print("Usage: python scripts/batch_extract_features.py train|val|test")
    sys.exit(1)

split = sys.argv[1]

source_dir = f"{BASE}/source/{split}"
metadata_file = f"{BASE}/{split}_metadata.csv"
work_dir = f"{BASE}/batch_work"
output_dir = f"{BASE}/features"

os.makedirs(output_dir, exist_ok=True)

output_file = f"{output_dir}/{split}_features.csv"
metadata = pd.read_csv(metadata_file)
metadata["sample_id"] = metadata["sample_id"].astype(int)

files = sorted(f for f in os.listdir(source_dir) if f.endswith(".c"))

print(f"Split: {split}")
print(f"Functions: {len(files)}")
print(f"Batch size: {BATCH_SIZE}")

if os.path.exists(output_file):
    os.remove(output_file)

processed = 0
failed = []

max_files = len(files) if MAX_BATCHES is None else min(
    len(files), BATCH_SIZE * MAX_BATCHES
)

for start in range(0, max_files, BATCH_SIZE):

    batch_files = files[start:start + BATCH_SIZE]
    batch_num = start // BATCH_SIZE + 1

    print(f"\n========== Batch {batch_num} ==========")
    print(f"Input: {len(batch_files)}")

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    os.makedirs(work_dir)

    for filename in batch_files:
        shutil.copy(
            os.path.join(source_dir, filename),
            os.path.join(work_dir, filename)
        )

    cpg_file = os.path.join(work_dir, "batch.cpg.bin")

    # First Joern attempt
    result = subprocess.run([
        "joern-parse",
        work_dir,
        "--output",
        cpg_file
    ])

    if result.returncode != 0:
        print(f"Joern failed on batch {batch_num}")
        sys.exit(1)

    # Extract features from the batch
    result = subprocess.run([
        "joern",
        cpg_file,
        "--script",
        "scripts/extract_features.sc"
    ])

    if result.returncode != 0:
        print(f"Feature extraction failed on batch {batch_num}")
        sys.exit(1)

    batch_features = f"{BASE}/cpg_test/features.csv"

    if not os.path.exists(batch_features):
        print("Feature CSV was not created.")
        sys.exit(1)

    df = pd.read_csv(batch_features)

    # Map filename -> sample ID
    df["sample_id"] = (
        df["filename"]
        .str.extract(r"sample_(\d+)\.c")[0]
    )

    df = df.dropna(subset=["sample_id"])
    df["sample_id"] = df["sample_id"].astype(int)

    found = set(df["sample_id"])

    expected = set(
        int(f.split("_")[1].split(".")[0])
        for f in batch_files
    )

    missing = sorted(expected - found)

    # Retry missing files after cleaning known attributes
    if missing:

        print(f"Missing after first parse: {len(missing)}")
        print("Retrying with source cleaning...")

        retry_dir = os.path.join(work_dir, "retry")
        os.makedirs(retry_dir)

        for sample_id in missing:

            filename = f"sample_{sample_id:05d}.c"
            src = os.path.join(source_dir, filename)
            dst = os.path.join(retry_dir, filename)

            with open(src, "r") as f:
                code = f.read()

            result = subprocess.run([
               sys.executable,
               "scripts/clean_source.py",
               src,
               dst
            ])

            if result.returncode != 0:
              print(f"Cleaning failed for {filename}")
              continue

        retry_cpg = os.path.join(work_dir, "retry.cpg.bin")

        result = subprocess.run([
            "joern-parse",
            retry_dir,
            "--output",
            retry_cpg
        ])

        if result.returncode == 0:

            result = subprocess.run([
                "joern",
                retry_cpg,
                "--script",
                "scripts/extract_features.sc"
            ])

            if result.returncode == 0 and os.path.exists(batch_features):

                retry_df = pd.read_csv(batch_features)

                retry_df["sample_id"] = (
                    retry_df["filename"]
                    .str.extract(r"sample_(\d+)\.c")[0]
                )

                retry_df = retry_df.dropna(subset=["sample_id"])
                retry_df["sample_id"] = retry_df["sample_id"].astype(int)

                # Add only newly recovered samples
                recovered = retry_df[
                    retry_df["sample_id"].isin(missing)
                ]

                if len(recovered):
                    df = pd.concat(
                        [df, recovered],
                        ignore_index=True
                    )

                # Keep exactly one row per input sample
                df = df.drop_duplicates(subset=["sample_id"], keep="first")

                found = set(df["sample_id"])
                still_missing = sorted(expected - found)

                if still_missing:
                    print(f"Still failed: {len(still_missing)}")
                    print(still_missing)
                    failed.extend(still_missing)
    # Safety check: never allow more rows than input files
    df = df[df["sample_id"].isin(expected)]
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    # Add labels
    df = df.merge(
        metadata[["sample_id", "target"]],
        on="sample_id",
        how="left"
    )

    if df["target"].isna().any():
        print("ERROR: Missing labels.")
        sys.exit(1)

    columns = [
        "sample_id",
        "target"
    ] + [
        c for c in df.columns
        if c not in ["sample_id", "target"]
    ]

    df = df[columns]

    df.to_csv(
        output_file,
        mode="a",
        header=not os.path.exists(output_file),
        index=False
    )

    processed += len(df)

    print(f"Final extracted from batch: {len(df)}")

    shutil.rmtree(work_dir)

print("\n===================================")
print("Feature extraction complete")
print("===================================")
print(f"Output: {output_file}")
print(f"Rows: {processed}")
print(f"Failed: {len(failed)}")

if failed:
    print("Failed sample IDs:")
    print(failed)

final = pd.read_csv(output_file)

print("\nFinal rows:", len(final))
print("Missing labels:", final["target"].isna().sum())
print("\nLabels:")
print(final["target"].value_counts())
