import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

# Load user-item matrix
matrix = pd.read_csv("datafiles/user_item_matrix.csv", index_col=0)
matrix = matrix.astype(np.float32)

print("Matrix shape:", matrix.shape)

# Compute cosine distance matrix (NxN)
D = cosine_distances(matrix)
print("Distance matrix computed:", D.shape)

# Choose number of clusters
K = 5

# Agglomerative clustering on precomputed distances
model = AgglomerativeClustering(
    n_clusters=K,
    metric="precomputed",
    linkage="average"
)

labels = model.fit_predict(D)

print("\nClustering completed!")
print("Cluster labels (first 20 users):")
print(labels[:20])

counts = pd.Series(labels).value_counts().sort_index()
print("\nUsers per cluster:")
print(counts)

# Save results for report / next steps
out = pd.DataFrame({"user": matrix.index.astype(int), "cluster": labels})
out.to_csv(f"datafiles/clusters_cosine_K{K}.csv", index=False)
print(f"\n Saved: datafiles/clusters_cosine_K{K}.csv")
