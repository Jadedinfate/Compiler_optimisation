#!/usr/bin/env python3
"""
fix_data.py
Fixes the accidental double-split by reading all existing split datasets,
merging and deduplicating them into a single unified dataset, and cleanly
re-partitioning them into a single 80% Train / 10% Val / 10% Test split.
Saves the resulting clean partitions into both CSV and JSON formats.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset_from_csv(files):
    """Loads and concatenates multiple CSV split files into a single DataFrame."""
    dfs = []
    for f in files:
        if os.path.exists(f):
            print(f"[*] Reading CSV split: {f}")
            df = pd.read_csv(f)
            dfs.append(df)
        else:
            print(f"[!] Warning: File not found: {f}")
    if not dfs:
        raise FileNotFoundError(f"No valid CSV files found among: {files}")
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_dataset_from_json(files):
    """Loads and concatenates multiple JSON split files into a single list/DataFrame."""
    all_records = []
    for f in files:
        if os.path.exists(f):
            print(f"[*] Reading JSON split: {f}")
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    all_records.extend(data)
                elif isinstance(data, dict):
                    all_records.append(data)
        else:
            print(f"[!] Warning: File not found: {f}")
    if not all_records:
        raise FileNotFoundError(f"No valid JSON files found among: {files}")
    return pd.DataFrame(all_records)


def deduplicate_dataset(df, target_col="target"):
    """
    Deduplicates records that may have been duplicated during previous splits.
    Uses sample_id or filename if available, otherwise deduplicates on entire row.
    """
    initial_len = len(df)
    if "sample_id" in df.columns:
        df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    elif "filename" in df.columns:
        df = df.drop_duplicates(subset=["filename"]).reset_index(drop=True)
    elif "func" in df.columns:
        df = df.drop_duplicates(subset=["func"]).reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)
    
    dropped = initial_len - len(df)
    print(f"[*] Deduplication complete: Removed {dropped} duplicates. Remaining total: {len(df)}")
    return df


def split_data(df, target_col="target", train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42):
    """
    Performs clean stratified 80/10/10 re-partitioning.
    Ensures class distribution is preserved across all three splits.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    if target_col not in df.columns:
        # Fallback to 'label' or 'y' if target is not named 'target'
        for candidate in ["label", "y", "vuln", "target"]:
            if candidate in df.columns:
                target_col = candidate
                break
    
    print(f"[*] Using target column: '{target_col}' for stratified splitting.")
    print(f"[*] Overall class distribution:\n{df[target_col].value_counts(normalize=True).to_dict()}")
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df[target_col]
    )
    
    # Second split: Val vs Test (50/50 split of the 20% temp pool)
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df[target_col]
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df, target_col


def save_splits(train_df, val_df, test_df, output_dir="data/clean_splits"):
    """Saves the splits in both CSV and JSON formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    print("\n" + "=" * 50)
    print("           DATASET RE-SPLIT SUMMARY")
    print("=" * 50)
    
    for split_name, split_df in splits.items():
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        json_path = os.path.join(output_dir, f"{split_name}.json")
        
        split_df.to_csv(csv_path, index=False)
        split_df.to_json(json_path, orient="records", indent=2)
        
        target_counts = split_df["target"].value_counts().to_dict() if "target" in split_df else {}
        print(f"[*] Split: {split_name.upper():<5} | Samples: {len(split_df):<6} | "
              f"Path: {csv_path} | Targets: {target_counts}")
    
    print("=" * 50)
    print(f"[+] All clean splits successfully saved to: {output_dir}/\n")


def main():
    parser = argparse.ArgumentParser(description="Clean and re-split dataset into 80/10/10 Train/Val/Test.")
    parser.add_argument("--input_dir", type=str, default="data",
                        help="Directory containing the split CSVs or JSONs")
    parser.add_argument("--output_dir", type=str, default="data/clean_splits",
                        help="Destination directory for clean splits")
    parser.add_argument("--target_col", type=str, default="target",
                        help="Label column name (e.g. target, label)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Discover candidate files
    csv_candidates = [
        os.path.join(args.input_dir, "train_features_final.csv"),
        os.path.join(args.input_dir, "val_features.csv"),
        os.path.join(args.input_dir, "test_features.csv"),
        os.path.join(args.input_dir, "train.csv"),
        os.path.join(args.input_dir, "val.csv"),
        os.path.join(args.input_dir, "test.csv")
    ]
    existing_csvs = [f for f in csv_candidates if os.path.exists(f)]

    json_candidates = [
        os.path.join(args.input_dir, "train.json"),
        os.path.join(args.input_dir, "val.json"),
        os.path.join(args.input_dir, "test.json"),
        "data/devign/train.json",
        "data/devign/val.json",
        "data/devign/test.json"
    ]
    existing_jsons = [f for f in json_candidates if os.path.exists(f)]

    if existing_csvs:
        print(f"[*] Found {len(existing_csvs)} CSV files. Merging...")
        df = load_dataset_from_csv(existing_csvs)
    elif existing_jsons:
        print(f"[*] Found {len(existing_jsons)} JSON files. Merging...")
        df = load_dataset_from_json(existing_jsons)
    else:
        print(f"[!] Error: No split files found in {args.input_dir}.")
        sys.exit(1)

    # Deduplicate
    df = deduplicate_dataset(df, target_col=args.target_col)

    # Re-split
    train_df, val_df, test_df, target_col = split_data(
        df,
        target_col=args.target_col,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=args.seed
    )

    # Save to disk
    save_splits(train_df, val_df, test_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
