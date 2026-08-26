import json
import csv
import os

BASE = "data/devign"

splits = {
    "train": "train.json",
    "val": "val.json",
    "test": "test.json"
}

for split, filename in splits.items():

    input_file = os.path.join(BASE, filename)
    output_dir = os.path.join(BASE, "source", split)
    metadata_file = os.path.join(BASE, f"{split}_metadata.csv")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as f:
        data = json.load(f)

    with open(metadata_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "project", "commit_id", "target", "file"])

        for i, item in enumerate(data):

            sample_id = f"{i:05d}"
            source_file = f"sample_{sample_id}.c"
            source_path = os.path.join(output_dir, source_file)

            with open(source_path, "w") as source:
                source.write(item["func"])

            writer.writerow([
                sample_id,
                item["project"],
                item["commit_id"],
                item["target"],
                source_file
            ])

    print(f"{split}: {len(data)} functions extracted")

print("Extraction complete.")
