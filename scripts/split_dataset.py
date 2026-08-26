import json
from sklearn.model_selection import train_test_split

INPUT = "data/devign/dataset_clean.json"

TRAIN = "data/devign/train.json"
VAL = "data/devign/val.json"
TEST = "data/devign/test.json"

# Load cleaned dataset
with open(INPUT, "r") as f:
    data = json.load(f)

print("Total samples:", len(data))

# First split: 80% train, 20% temporary
train, temp = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=[x["target"] for x in data]
)

# Second split: 10% validation, 10% test
val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=[x["target"] for x in temp]
)

# Save
for path, dataset in [
    (TRAIN, train),
    (VAL, val),
    (TEST, test)
]:
    with open(path, "w") as f:
        json.dump(dataset, f)

    print(f"{path}: {len(dataset)} samples")

print("Split complete.")
