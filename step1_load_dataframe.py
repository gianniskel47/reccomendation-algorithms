import numpy as np
import pandas as pd
import os


DATAFILE = "Dataset.npy"
OUT_DIR = "datafiles"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load the .npy dataset (array of strings like: "ur...,tt...,rating,date")
dataset = np.load(DATAFILE, allow_pickle=True)

print("Dataset loaded successfully!")
print("Total rows:", len(dataset))

# 2) Split each row by comma into 4 fields
spliter = lambda s: s.split(",")
dataset = np.array([spliter(x) for x in dataset])

# 3) Create DataFrame
df = pd.DataFrame(dataset, columns=["user", "item", "rating", "date"])

# 4) Clean columns
df["user"] = df["user"].str.replace("ur", "", regex=False).astype("int64")
df["item"] = df["item"].str.replace("tt", "", regex=False).astype("int64")
df["rating"] = df["rating"].astype("int64")
df["date"] = pd.to_datetime(df["date"])

# 5) Quick preview
print("\nPreview (raw cleaned):")
print(df.head())

print("\nInfo:")
print(df.info())

# 6) Save for next steps (so we don’t reload/parse every time)
pkl_path = os.path.join(OUT_DIR, "dataframe_full.pkl")
df.to_pickle(pkl_path)
print(f"\n Saved cleaned full DataFrame to: {pkl_path}")
