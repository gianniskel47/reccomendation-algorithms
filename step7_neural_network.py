import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

print("START: Neural Networks & Recommendation Generation")

# 1. Load Data
DATA_DIR = "datafiles"
matrix = pd.read_csv(os.path.join(DATA_DIR, "user_item_matrix.csv"), index_col=0)
clusters = pd.read_csv(os.path.join(DATA_DIR, "clusters_jaccard_L5.csv"))
distances = np.load(os.path.join(DATA_DIR, "distance_matrix_jaccard.npy"))

# Convert to NumPy arrays for maximum speed
matrix_np = matrix.to_numpy()
users_list = matrix.index.tolist()
user_to_idx = {user: i for i, user in enumerate(users_list)}

K_NEIGHBORS = 30

results = [] # Here we store the MAE for the final table

# Iterate over each of the 5 clusters
for cluster_id in range(5):
    # Get the users of this cluster
    cluster_users = clusters[clusters['cluster'] == cluster_id]['user'].tolist()
    
    # Split into Train 80% and Test 20%
    train_users, test_users = train_test_split(cluster_users, test_size=0.2, random_state=42)
    
    # Function to create the Features & Labels Matrix
    def build_dataset(user_subset):
        X, y = [], []
        for u in user_subset:
            u_idx = user_to_idx[u]
            
            # Find the other users in the same cluster
            cluster_indices = [user_to_idx[cu] for cu in cluster_users if cu != u]
            
            # Get the Jaccard distances of user u from them
            u_distances = distances[u_idx, cluster_indices]
            
            # Find the K Nearest Neighbors
            nearest_relative = np.argsort(u_distances)[:K_NEIGHBORS]
            nearest_global = [cluster_indices[i] for i in nearest_relative]
            
            # Find which movies user u has rated
            rated_items_indices = np.where(matrix_np[u_idx] > 0)[0]
            
            for item_idx in rated_items_indices:
                target = matrix_np[u_idx, item_idx]
                
                # Features: The ratings of the K neighbors 
                features = [matrix_np[neighbor_idx, item_idx] for neighbor_idx in nearest_global]
                
                X.append(features)
                y.append(target)
                
        return np.array(X), np.array(y)

    print(f"\nProcessing Cluster {cluster_id} (Users: {len(cluster_users)}) ...")
    
    # Create training and testing data
    X_train, y_train = build_dataset(train_users)
    X_test, y_test = build_dataset(test_users)
    
    # The Final Neural Network (Protected against Overfitting)
    nn = MLPRegressor(
        hidden_layer_sizes=(64, 32),    # Large enough to learn complex patterns
        activation='relu',              
        solver='adam',                  
        alpha=0.05,                     # Regularization: Penalty to prevent memorization
        early_stopping=True,            # Stops automatically if it detects overfitting!
        validation_fraction=0.1,        
        max_iter=500,                   
        random_state=42
    )
    
    nn.fit(X_train, y_train)
    
    # Predict Ratings
    train_preds = nn.predict(X_train)
    test_preds = nn.predict(X_test)
    
    # Calculate MAE
    mae_train = mean_absolute_error(y_train, train_preds)
    mae_test = mean_absolute_error(y_test, test_preds)
    
    print(f"-> Completed! Train MAE: {mae_train:.4f} | Test MAE: {mae_test:.4f}")
    
    # Save results
    results.append({
        "Cluster": cluster_id,
        "Train Users": len(train_users),
        "Test Users": len(test_users),
        "Train MAE": round(mae_train, 4),
        "Test MAE": round(mae_test, 4)
    })

#  FINAL RESULTS TABLE 
print("\n" + "="*50)
print(" RESULTS TABLE")
print("="*50)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save the final table
results_df.to_csv(os.path.join(DATA_DIR, "final_mae_results.csv"), index=False)