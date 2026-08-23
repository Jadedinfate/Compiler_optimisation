import json
from collections import defaultdict

INPUT = "data/devign/dataset.json"
OUTPUT = "data/devign/dataset_clean.json"

# Load dataset
with open(INPUT, "r") as f:
    data = json.load(f)

print("Original samples:", len(data))

# Find labels associated with each function
labels = defaultdict(set)

for item in data:
    labels[item["func"]].add(item["target"])

# Functions having conflicting labels
conflicting = {
    func for func, targets in labels.items()
    if len(targets) > 1
}

print("Conflicting functions:", len(conflicting))

# Remove ALL occurrences of conflicting functions
clean_data = [
    item for item in data
    if item["func"] not in conflicting
]

print("Clean samples:", len(clean_data))

# Save cleaned dataset
with open(OUTPUT, "w") as f:
    json.dump(clean_data, f, indent=2)

print("Saved to:", OUTPUT)