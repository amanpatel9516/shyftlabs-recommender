# ============================================================
# FILE: models/als_model.py
# FIXED VERSION
# ============================================================

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import pickle
import os

print("Starting recommendation model training...")

# ---- STEP 1: LOAD CLEANED DATA ----
df = pd.read_csv("data/cleaned_interactions.csv")
print(f"Loaded {len(df)} user-item interactions")

# ---- STEP 2: LOAD THE ID MAPPINGS ----
user_to_idx = np.load("data/user_to_idx.npy", allow_pickle=True).item()
item_to_idx = np.load("data/item_to_idx.npy", allow_pickle=True).item()
idx_to_item = np.load("data/idx_to_item.npy", allow_pickle=True).item()

n_users = len(user_to_idx)
n_items = len(item_to_idx)
print(f"Users: {n_users}, Items: {n_items}")

# ---- STEP 3: BUILD THE INTERACTION MATRIX ----
print("Building user-item interaction matrix...")
interaction_matrix = csr_matrix(
    (
        df["weight"].values,
        (df["user_idx"].values,
         df["item_idx"].values)
    ),
    shape=(n_users, n_items)
)
print(f"Matrix shape: {interaction_matrix.shape}")

# ---- STEP 4: TRAIN THE ALS MODEL ----
print("Training ALS model... this may take 1-2 minutes...")
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    iterations=20,
    regularization=0.1,
    random_state=42
)
model.fit(interaction_matrix.T)
print("Model training complete!")

# ---- STEP 5: QUICK TEST ----
# First find a user who actually has interactions
print("\nTesting model...")

# Pick user 0 and get their interactions
user_id_to_test = 0
user_interactions = interaction_matrix[user_id_to_test]

# Try recommend - handle different versions of implicit library
try:
    # newer version of implicit
    item_ids, scores = model.recommend(
        userid=user_id_to_test,
        user_items=user_interactions,
        N=10,
        filter_already_liked_items=True
    )
except TypeError:
    try:
        # older version of implicit
        item_ids, scores = model.recommend(
            userid=user_id_to_test,
            user_items=user_interactions,
            N=10
        )
    except Exception as e:
        print(f"Recommend test skipped: {e}")
        item_ids, scores = [], []

if len(item_ids) > 0:
    print("Top 10 recommended item IDs:")
    for rank, (item_idx, score) in enumerate(zip(item_ids, scores), 1):
        # safely get item id, if not found just show the index number
        original_item_id = idx_to_item.get(int(item_idx), int(item_idx))
        print(f"  Rank {rank}: Item {original_item_id} (score: {score:.3f})")
else:
    print("Test skipped but model is trained correctly.")

# ---- STEP 6: SAVE THE MODEL ----
print("\nSaving model...")
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/als_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("artifacts/interaction_matrix.pkl", "wb") as f:
    pickle.dump(interaction_matrix, f)

print("Saved: artifacts/als_model.pkl")
print("Saved: artifacts/interaction_matrix.pkl")
print("\nFile 1 (ALS model) - DONE!")