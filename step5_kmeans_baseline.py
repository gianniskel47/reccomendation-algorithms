import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Load user-item matrix
matrix = pd.read_csv("datafiles/user_item_matrix.csv", index_col=0)
matrix = matrix.astype(np.float32)

print("Matrix shape:", matrix.shape)

# Choose number of clusters
K = 5

# Run MiniBatchKMeans (faster for large sparse-ish data)
kmeans = MiniBatchKMeans(
    n_clusters=K,
    random_state=42,
    batch_size=256,
    n_init=10
)

labels = kmeans.fit_predict(matrix)

print("\nClustering completed!")
print("Cluster labels (first 20 users):")
print(labels[:20])

# Count users per cluster
counts = pd.Series(labels).value_counts().sort_index()

print("\nUsers per cluster:")
print(counts)

# Save results
out = pd.DataFrame({"user": matrix.index.astype(int), "cluster": labels})
out.to_csv(f"datafiles/clusters_kmeans_K{K}.csv", index=False)
print(f"\n Saved: datafiles/clusters_kmeans_K{K}.csv")
