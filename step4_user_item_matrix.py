import pandas as pd
import os

# ----------------------------
# STEP 4: User–Item Matrix P(u)
# ----------------------------

DATA_DIR = "datafiles"

# Load reduced dataset from Step 2
df = pd.read_pickle(
    os.path.join(DATA_DIR, "dataframe_reduced_R100_300.pkl")
)

print("Loaded reduced dataset:")
print("Rows:", len(df))
print("Unique users:", df["user"].nunique())
print("Unique items:", df["item"].nunique())

# ----------------------------
# User–Item Matrix
# ----------------------------
user_item_matrix = df.pivot_table(
    index="user",
    columns="item",
    values="rating",
    fill_value=0
)

print("\nUser–Item Matrix shape:", user_item_matrix.shape)

print("\nMatrix preview (first 5 users, first 5 items):")
print(user_item_matrix.iloc[:5, :5])

# Save matrix
out_path = os.path.join(DATA_DIR, "user_item_matrix.csv")
user_item_matrix.to_csv(out_path)

print(f"\n User–Item Matrix saved as: {out_path}")
