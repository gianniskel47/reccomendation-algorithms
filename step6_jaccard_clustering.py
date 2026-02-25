import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
from sklearn_extra.cluster import KMedoids  

print("Start Jaccard calculation & Clustering")

# 1. Load user-item matrix
DATA_DIR = "datafiles"
matrix_path = os.path.join(DATA_DIR, "user_item_matrix.csv")

print("Loading matrix data...")
matrix = pd.read_csv(matrix_path, index_col=0)
print(f"Matrix dimensions: {matrix.shape}")

# 2. Convert to SPARSE binary matrix
print("\nConverting to SPARSE binary matrix...")
binary_matrix_sparse = sp.csr_matrix((matrix > 0).astype(int))

# 3. Jaccard calculation using Linear Algebra (Fast method)
print("Calculating Jaccard square distance matrix...")

# Intersection = Matrix multiplication with its transpose
intersection = binary_matrix_sparse.dot(binary_matrix_sparse.T).toarray()

# Sum of movies watched by each user
row_sums = binary_matrix_sparse.sum(axis=1).A.flatten()
 
# Union = |A| + |B| - |A ∩ B|
union = row_sums[:, None] + row_sums[None, :] - intersection

# Jaccard Distance = 1 - (Intersection / Union)
D_jaccard = 1.0 - (intersection / np.maximum(union, 1))

# Distance of a user from themselves must be strictly 0
np.fill_diagonal(D_jaccard, 0)

# Save for the Neural Network (Question 2b)
dist_path = os.path.join(DATA_DIR, "distance_matrix_jaccard.npy")
np.save(dist_path, D_jaccard)
print(f"The Jaccard square matrix was successfully saved to: {dist_path}")

# 4. Clustering
L = 5 

# K-Medoids handles sparse outliers much better
model = KMedoids(n_clusters=L, metric="precomputed", random_state=42)
labels = model.fit_predict(D_jaccard)

# 5. Display Results
counts = pd.Series(labels).value_counts().sort_index()
print("\n RESULTS: Users per Cluster (Jaccard)")
print(counts)

# 6. Save the Clusters
out = pd.DataFrame({"user": matrix.index.astype(int), "cluster": labels})
clusters_path = os.path.join(DATA_DIR, f"clusters_jaccard_L{L}.csv")
out.to_csv(clusters_path, index=False)
print(f"\nSuccess! The clusters were saved to: {clusters_path}")