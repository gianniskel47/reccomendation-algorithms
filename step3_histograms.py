import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ----------------------------
# STEP 3: Histograms (FULL) for reduced dataset
# ----------------------------

DATA_DIR = "datafiles"
df = pd.read_pickle(os.path.join(DATA_DIR, "dataframe_reduced_R100_300.pkl"))

print("Loaded reduced dataset:", df.shape)
print("Unique users:", df["user"].nunique())
print("Unique items:", df["item"].nunique())

# Create figures folder
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------
# Histogram 1: Ratings per user
# ----------------------------
ratings_per_user = df.groupby("user")["rating"].count()

plt.figure()
n, bins, patches = plt.hist(ratings_per_user, bins="auto", alpha=0.75)
plt.title("Number of Ratings per User")
plt.xlabel("Ratings count")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)

# nice y-limit (like professor code)
maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

out1 = os.path.join(FIGURES_DIR, "ratings_per_user.png")
plt.savefig(out1, dpi=120, bbox_inches="tight")
plt.show()
print("Saved:", out1)

# ----------------------------
# Histogram 2: Time span per user
# ----------------------------
span_per_user = df.groupby("user")["date"].apply(lambda x: (x.max() - x.min()).days)

plt.figure()
n, bins, patches = plt.hist(span_per_user, bins="auto", alpha=0.75)
plt.title("Time Span of Ratings per User (days)")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)

maxfreq = n.max()
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

out2 = os.path.join(FIGURES_DIR, "ratings_span_per_user.png")
plt.savefig(out2, dpi=120, bbox_inches="tight")
plt.show()
print("Saved:", out2)

# Print stats (useful for report)
print("\nRatings per user stats:")
print(ratings_per_user.describe())

print("\nTime span (days) stats:")
print(span_per_user.describe())
