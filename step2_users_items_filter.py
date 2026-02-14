import pandas as pd
import os

# ----------------------------
# STEP 2 (FULL): Load cleaned full dataset from Step 1
# ----------------------------
DATA_DIR = "datafiles"
df = pd.read_pickle(os.path.join(DATA_DIR, "dataframe_full.pkl"))

print("Loaded full dataset!")
print("Total rows:", len(df))

# ----------------------------
# 1. Unique users and items
# ----------------------------
users = df["user"].unique()
items = df["item"].unique()

print("Total unique users:", len(users))
print("Total unique items:", len(items))
print("Total ratings:", len(df))

# ----------------------------
# 2. Ratings per user
# ----------------------------
ratings_per_user = df.groupby("user")["rating"].count()

print("\nRatings per user stats:")
print("Min:", int(ratings_per_user.min()))
print("Median:", int(ratings_per_user.median()))
print("Max:", int(ratings_per_user.max()))

print("\nRatings per user (first 10):")
print(ratings_per_user.sort_values(ascending=False).head(10))

# ----------------------------
# 3. Filtering U′ (Rmin, Rmax)
# ----------------------------
r_min = 100
r_max = 300

filtered_users = ratings_per_user[
    (ratings_per_user >= r_min) &
    (ratings_per_user <= r_max)
]

print(f"\nUsers kept after filtering (Rmin={r_min}, Rmax={r_max}):", len(filtered_users))

# Final reduced dataset
final_df = df[df["user"].isin(filtered_users.index)].copy()

print("Final reduced dataset rows:", len(final_df))
print("Final reduced unique users:", final_df["user"].nunique())
print("Final reduced unique items:", final_df["item"].nunique())

# ----------------------------
# 4. Save reduced dataframe for next steps
# ----------------------------
out_path = os.path.join(DATA_DIR, f"dataframe_reduced_R{r_min}_{r_max}.pkl")
final_df.to_pickle(out_path)
print(f"\n Saved reduced dataframe to: {out_path}")
