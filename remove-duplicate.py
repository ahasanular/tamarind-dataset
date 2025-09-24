import pandas as pd

# Load your splits
train_df = pd.read_csv("./output/split_train.csv")
val_df   = pd.read_csv("./output/split_val_clean.csv")
test_df  = pd.read_csv("./output/split_test.csv")  # if you have one
meta_df  = pd.read_csv("./output/tamarind_metadata.csv")    # optional full metadata

# Find duplicate phash across splits
train_hashes = set(train_df["phash"])
val_hashes   = set(val_df["phash"])
test_hashes  = set(test_df["phash"]) if "split_test.csv" in locals() else set()

dup_train_val  = train_hashes & val_hashes
dup_train_test = train_hashes & test_hashes
dup_val_test   = val_hashes & test_hashes

print("Train–Val duplicates:", len(dup_train_val))
print("Train–Test duplicates:", len(dup_train_test))
print("Val–Test duplicates:", len(dup_val_test))

dup_train_val = False

if dup_train_val:
    # dup_rows = train_df[train_df["phash"].isin(dup_train_val)]
    # print("Train–Val duplicate rows:")
    # print(dup_rows.head())


    train_df = pd.read_csv("./output/split_train.csv")
    val_df   = pd.read_csv("./output/split_val.csv")

    # Get duplicate phash overlap
    dup_train_val = set(train_df["phash"]) & set(val_df["phash"])
    print(f"Found {len(dup_train_val)} duplicate phash in Train–Val")

    # Remove those from val
    val_clean = val_df[~val_df["phash"].isin(dup_train_val)]
    print(f"Val size before: {len(val_df)}, after cleaning: {len(val_clean)}")

    # Save cleaned val split
    val_clean.to_csv("./output/split_val_clean.csv", index=False)
    print("Saved ./output/split_val_clean.csv")