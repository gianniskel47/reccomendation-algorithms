import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_distances

# Load matrix
matrix = pd.read_csv("datafile/user_item_matrix.csv", index_col=0)

print("Matrix shape:", matrix.shape)

# Compute cosine distance matrix
D = cosine_distances(matrix)

print("Distance matrix computed!")

# Choose K
K = 5

# Run KMedoids clustering
kmedoids = KMedoids(n_clusters=K, metric="precomputed", random_state=42)
labels = kmedoids.fit_predict(D)

print("\nKMedoids clustering completed!")

# Count users per cluster
counts = pd.Series(labels).value_counts().sort_index()

print("\nUsers per cluster:")
print(counts)
