# ============================================================
# FILE: models/ranker.py
# FIXED VERSION
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
import os

print("Starting ranker model training...")

# ---- STEP 1: LOAD DATA AND MODEL ----
df = pd.read_csv("data/cleaned_interactions.csv")

with open("artifacts/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)

with open("artifacts/interaction_matrix.pkl", "rb") as f:
    interaction_matrix = pickle.load(f)

idx_to_item = np.load(
    "data/idx_to_item.npy", allow_pickle=True
).item()

print(f"Loaded {len(df)} interactions")

# ---- STEP 2: BUILD ITEM FEATURES ----
print("Building item features...")

item_popularity = df.groupby("item_idx")["weight"].agg([
    "sum",
    "count",
    "mean"
]).reset_index()

item_popularity.columns = [
    "item_idx",
    "total_weight",
    "interaction_count",
    "avg_weight"
]

item_popularity["purchase_rate"] = (
    item_popularity["avg_weight"] / 5.0
)

print(f"Built features for {len(item_popularity)} items")

# ---- STEP 3: FIND THE RIGHT RECOMMEND FUNCTION ----
# First figure out which version of implicit is installed
# by testing with one user before the loop

print("Detecting implicit library version...")

# find a user that actually has interactions
test_user = int(df["user_idx"].iloc[0])
test_interactions = interaction_matrix[test_user]

# try new version first
USE_NEW_API = False
try:
    ids, sc = als_model.recommend(
        userid=test_user,
        user_items=test_interactions,
        N=5,
        filter_already_liked_items=True
    )
    USE_NEW_API = True
    print("Using new implicit API (filter_already_liked_items)")
except TypeError:
    try:
        ids, sc = als_model.recommend(
            userid=test_user,
            user_items=test_interactions,
            N=5
        )
        USE_NEW_API = False
        print("Using old implicit API")
    except Exception as e:
        print(f"ERROR: Cannot call recommend: {e}")
        exit()

def get_recommendations(user_idx, n=50):
    # wrapper function that works for both old and new implicit
    user_items = interaction_matrix[user_idx]
    try:
        if USE_NEW_API:
            item_ids, scores = als_model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=n,
                filter_already_liked_items=False
            )
        else:
            item_ids, scores = als_model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=n
            )
        return item_ids, scores
    except Exception:
        return [], []

# ---- STEP 4: CREATE TRAINING DATA ----
print("Creating training examples...")

# use first 500 users that have at least 3 interactions
# (more interactions = better training labels)
active_users = (
    df.groupby("user_idx")
    .filter(lambda x: len(x) >= 3)["user_idx"]
    .unique()[:500]
)

print(f"Using {len(active_users)} active users for training")

training_rows = []

for i, user_idx in enumerate(active_users):
    user_idx = int(user_idx)

    # get recommendations
    item_ids, scores = get_recommendations(user_idx, n=50)

    if len(item_ids) == 0:
        continue

    # get items this user actually liked (weight >= 3)
    user_liked = set(
        df[
            (df["user_idx"] == user_idx) &
            (df["weight"] >= 3)
        ]["item_idx"].tolist()
    )

    # create one row per candidate item
    for item_idx, als_score in zip(item_ids, scores):
        training_rows.append({
            "user_idx":  user_idx,
            "item_idx":  int(item_idx),
            "als_score": float(als_score),
            "label":     1 if int(item_idx) in user_liked else 0
        })

    # print progress every 100 users
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1} users, "
              f"rows so far: {len(training_rows)}")

print(f"\nTotal training examples created: {len(training_rows)}")

# safety check
if len(training_rows) == 0:
    print("ERROR: No training examples created.")
    print("This means recommend() returned empty for all users.")
    print("Saving a simple ranker based on item popularity only...")

    # fallback: train on item features directly without ALS scores
    train_df = item_popularity.copy()
    train_df["label"] = (
        train_df["avg_weight"] >= 3
    ).astype(int)

    feature_cols = [
        "total_weight",
        "interaction_count",
        "avg_weight",
        "purchase_rate"
    ]
else:
    train_df = pd.DataFrame(training_rows)
    print(f"Positive examples (liked): {train_df['label'].sum()}")

    # merge item features
    train_df = train_df.merge(
        item_popularity, on="item_idx", how="left"
    )
    train_df = train_df.fillna(0)

    feature_cols = [
        "als_score",
        "total_weight",
        "interaction_count",
        "avg_weight",
        "purchase_rate"
    ]

# ---- STEP 5: TRAIN LIGHTGBM ----
print("\nTraining LightGBM ranker...")

X = train_df[feature_cols]
y = train_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ranker = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=-1
)
ranker.fit(X_train, y_train)

test_score = ranker.score(X_test, y_test)
print(f"Ranker accuracy: {test_score:.3f}")

# ---- STEP 6: SAVE ----
with open("artifacts/ranker.pkl", "wb") as f:
    pickle.dump(ranker, f)

# save feature column names too
# (needed in API to build features in same order)
with open("artifacts/item_features.pkl", "wb") as f:
    pickle.dump(item_popularity, f)

with open("artifacts/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("Saved: artifacts/ranker.pkl")
print("Saved: artifacts/item_features.pkl")
print("Saved: artifacts/feature_cols.pkl")
print("\nFile 2 (Ranker) - DONE!")