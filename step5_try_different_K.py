import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

# Load matrix
matrix = pd.read_csv("datafiles/user_item_matrix.csv", index_col=0)

# Compute cosine distance matrix
D = cosine_distances(matrix)

# Try different K values
for K in [3, 5, 8, 10]:
    print("\n========================")
    print("Clustering with K =", K)

    model = AgglomerativeClustering(
        n_clusters=K,
        metric="precomputed",
        linkage="average"
    )

    labels = model.fit_predict(D)

    counts = pd.Series(labels).value_counts().sort_index()
    print("Users per cluster:")
    print(counts)
