# ============================================================
# FILE: api/main.py
# FIXED VERSION - no class imports needed
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="ShyftLabs Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading all models...")

with open("artifacts/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)

with open("artifacts/interaction_matrix.pkl", "rb") as f:
    interaction_matrix = pickle.load(f)

with open("artifacts/ranker.pkl", "rb") as f:
    ranker = pickle.load(f)

with open("artifacts/item_features.pkl", "rb") as f:
    item_features = pickle.load(f)

with open("artifacts/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

with open("artifacts/bandit.pkl", "rb") as f:
    bandit = pickle.load(f)

idx_to_item = np.load(
    "data/idx_to_item.npy", allow_pickle=True
).item()


# load product catalog with names and prices
import os
if os.path.exists("data/product_catalog.csv"):
    product_catalog = pd.read_csv("data/product_catalog.csv")
    # make a dictionary: item_id -> product info for fast lookup
    product_dict = product_catalog.set_index("itemid").to_dict("index")
    print("Product catalog loaded!")
else:
    product_dict = {}
    print("No product catalog found - will show item IDs only")
print("All models loaded! Server is ready.")


# ---- BANDIT HELPER FUNCTIONS ----
# Instead of a class, we use simple functions
# that work directly with the bandit dictionary

def bandit_select_ad():
    alpha  = bandit["alpha"]
    beta   = bandit["beta"]
    samples = np.random.beta(alpha, beta)
    return int(np.argmax(samples))

def bandit_update(ad_idx, clicked):
    if clicked:
        bandit["alpha"][ad_idx] += 1
    else:
        bandit["beta"][ad_idx]  += 1

def bandit_get_ctrs():
    return bandit["alpha"] / (bandit["alpha"] + bandit["beta"])


# ---- RECOMMEND HELPER ----
def get_recommendations_for_user(user_idx, n=50):
    user_items = interaction_matrix[user_idx]
    try:
        item_ids, scores = als_model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=n,
            filter_already_liked_items=True
        )
    except TypeError:
        item_ids, scores = als_model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=n
        )
    return item_ids, scores


# ============================================================
# HOME PAGE
# http://localhost:8000
# ============================================================

@app.get("/")
def home():
    return {
        "project":     "ShyftLabs Recommender System",
        "description": "AI-powered recommendation and AdTech system",
        "endpoints": {
            "recommendations": "/recommend/{user_id}",
            "ad_serving":      "/ad-serve",
            "metrics":         "/metrics"
        },
        "built_with": [
            "ALS Matrix Factorization",
            "LightGBM Ranker",
            "Thompson Sampling Bandit"
        ]
    }


# ============================================================
# LINK 1: GET /recommend/{user_id}
# http://localhost:8000/recommend/0
# ============================================================

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    try:
        item_ids, als_scores = get_recommendations_for_user(
            user_id, n=50
        )

        if len(item_ids) == 0:
            return {
                "user_id": user_id,
                "recommendations": []
            }

        candidates_df = pd.DataFrame({
            "item_idx":  [int(i) for i in item_ids],
            "als_score": [float(s) for s in als_scores]
        })

        candidates_df = candidates_df.merge(
            item_features, on="item_idx", how="left"
        ).fillna(0)

        for col in feature_cols:
            if col not in candidates_df.columns:
                candidates_df[col] = 0

        X           = candidates_df[feature_cols]
        rank_scores = ranker.predict_proba(X)[:, 1]

        candidates_df["rank_score"] = rank_scores
        candidates_df = candidates_df.sort_values(
            "rank_score", ascending=False
        ).head(n)

        recommendations = []
        for _, row in candidates_df.iterrows():
            real_item_id = idx_to_item.get(
                int(row["item_idx"]), int(row["item_idx"])
            )
            real_item_id = int(real_item_id)

            # get product name and details if available
            if real_item_id in product_dict:
                prod = product_dict[real_item_id]
                recommendations.append({
                    "item_id":         real_item_id,
                    "name":            prod["name"],
                    "category":        prod["category"],
                    "price":           f"Rs. {prod['price']}",
                    "rating":          prod["rating"],
                    "relevance_score": round(float(row["rank_score"]), 4),
                    "popularity":      int(row.get("interaction_count", 0))
                })
            else:
                recommendations.append({
                    "item_id":         real_item_id,
                    "name":            f"Product {real_item_id}",
                    "category":        "General",
                    "price":           "N/A",
                    "rating":          0.0,
                    "relevance_score": round(float(row["rank_score"]), 4),
                    "popularity":      int(row.get("interaction_count", 0))
                })
        return {
            "user_id":         int(user_id),
            "total_results":   int(len(recommendations)),
            "recommendations": recommendations
        }

    except Exception as e:
        return {"error": str(e), "user_id": user_id}


# ============================================================
# LINK 2: GET /ad-serve
# http://localhost:8000/ad-serve
# ============================================================

@app.get("/ad-serve")
def serve_ad(user_id: int = 0):
    ad_names = [
        "10% off on Electronics",
        "Buy 2 Get 1 Free",
        "Flash Sale — 50% off",
        "Free Shipping Today",
        "New Arrivals — Shop Now"
    ]

    # ---- DYNAMIC CTR BASED ON USER ----
    # In real life CTR depends on the user's interests
    # We simulate this by varying CTR based on user_id
    # Different users respond differently to different ads
    # This makes the system truly dynamic

    # base CTRs for each ad
    base_ctrs = [0.12, 0.15, 0.28, 0.18, 0.10]

    # add user-based variation
    # so different users have different ad preferences
    np.random.seed(user_id % 100)   # seed based on user
    variation = np.random.uniform(-0.05, 0.05, size=5)
    user_ctrs = np.clip(
        np.array(base_ctrs) + variation, 0.05, 0.45
    )

    # pick ad using bandit
    chosen_ad = bandit_select_ad()

    # simulate click based on this user's CTR for this ad
    clicked = bool(np.random.rand() < user_ctrs[chosen_ad])

    # update bandit
    bandit_update(chosen_ad, clicked)

    ctr_estimates = bandit_get_ctrs()

    return {
        "chosen_ad_id":   chosen_ad,
        "chosen_ad_name": ad_names[chosen_ad],
        "user_clicked":   clicked,
        "user_id":        user_id,
        "all_ad_ctrs": [
            {
                "ad_id":         i,
                "ad_name":       ad_names[i],
                "estimated_ctr": round(float(ctr), 4)
            }
            for i, ctr in enumerate(ctr_estimates)
        ]
    }


@app.get("/reset-bandit")
def reset_bandit():
    # reset bandit to starting state
    # so it forgets everything and learns again from zero
    bandit["alpha"] = np.ones(5)
    bandit["beta"]  = np.ones(5)
    bandit["history"] = []
    return {
        "message": "Bandit reset! All ads back to equal CTR. "
                   "Watch it learn again from scratch.",
        "all_ad_ctrs": {
            f"ad_{i}": 0.5
            for i in range(5)
        }
    }


# ============================================================
# LINK 3: GET /metrics
# http://localhost:8000/metrics
# ============================================================

@app.get("/metrics")
def metrics():
    ctr_estimates   = bandit_get_ctrs()
    best_ctr        = float(ctr_estimates.max())
    best_ad         = int(ctr_estimates.argmax())
    random_baseline = 0.12
    lift_pct        = round(
        (best_ctr - random_baseline) / random_baseline * 100, 1
    )

    return {
        "business_metrics": {
            "best_ad_id":              best_ad,
            "best_ad_ctr":             round(best_ctr, 4),
            "random_baseline_ctr":     random_baseline,
            "improvement_over_random": f"{lift_pct}%",
            "message": (
                f"ML ad targeting gives {lift_pct}% better CTR "
                f"than random selection"
            )
        },
        "all_ad_ctrs": {
            f"ad_{i}": round(float(ctr), 4)
            for i, ctr in enumerate(ctr_estimates)
        },
        "model_info": {
            "recommendation_model": "ALS Matrix Factorization",
            "ranking_model":        "LightGBM Classifier",
            "ad_selection_model":   "Thompson Sampling Bandit",
            "total_users":          interaction_matrix.shape[0],
            "total_items":          interaction_matrix.shape[1]
        }
    }
# ============================================================
# LINK 4: GET /user-profile/{user_id}
# http://localhost:8000/user-profile/0
# Shows what kind of user this is based on their history
# ============================================================

@app.get("/user-profile/{user_id}")
def user_profile(user_id: int):
    try:
        # load interaction data to understand this user
        import pandas as pd 
        df = pd.read_csv("data/cleaned_interactions.csv")

        # get this user's interactions
        user_data = df[df["user_idx"] == user_id]

        if len(user_data) == 0:
            return {
                "user_id":    user_id,
                "is_new_user": True,
                "message":    "New user — no history yet",
                "total_interactions": 0
            }

        # basic stats
        total_interactions = len(user_data)
        total_purchases    = len(user_data[user_data["weight"] == 5])
        total_cart         = len(user_data[user_data["weight"] == 3])
        total_views        = len(user_data[user_data["weight"] == 1])

        # get top items this user interacted with
        top_items = (
            user_data.nlargest(5, "weight")["itemid"].tolist()
            if "itemid" in user_data.columns
            else user_data.nlargest(5, "weight")["item_idx"].tolist()
        )

        # figure out user type based on interaction count
        if total_interactions >= 20:
            user_type    = "Power Shopper"
            user_emoji   = "Power buyer — very active"
        elif total_interactions >= 10:
            user_type    = "Regular Shopper"
            user_emoji   = "Shops regularly"
        elif total_interactions >= 5:
            user_type    = "Occasional Shopper"
            user_emoji   = "Shops sometimes"
        else:
            user_type    = "New Shopper"
            user_emoji   = "Just getting started"

        # engagement score 0 to 100
        engagement = min(
            100,
            int((total_purchases * 5 + total_cart * 3 + total_views) / 
                max(total_interactions, 1) * 20)
        )

        return {
            "user_id":            user_id,
            "is_new_user":        False,
            "user_type":          user_type,
            "user_description":   user_emoji,
            "total_interactions": int(total_interactions),
            "total_purchases":    int(total_purchases),
            "total_cart_adds":    int(total_cart),
            "total_views":        int(total_views),
            "engagement_score":   engagement,
            "top_item_ids":       [int(i) for i in top_items]
        }

    except Exception as e:
        return {"error": str(e), "user_id": user_id}


# ============================================================
# LINK 5: GET /similar-items/{item_id}
# http://localhost:8000/similar-items/461686
# Finds products similar to a given product
# ============================================================

@app.get("/similar-items/{item_id}")
def similar_items(item_id: int, n: int = 6):
    try:
        # check if item exists in our mapping
        item_to_idx = np.load(
            "data/item_to_idx.npy", allow_pickle=True
        ).item()

        if item_id not in item_to_idx:
            return {
                "item_id": item_id,
                "message": "Item not found",
                "similar": []
            }

        item_idx = item_to_idx[item_id]

        # use ALS model to find similar items
        # similar_items() finds items with similar "fingerprints"
        similar_ids, scores = als_model.similar_items(
            itemid=item_idx,
            N=n + 1   # +1 because it includes the item itself
        )

        # remove the item itself from results
        results = []
        for sid, score in zip(similar_ids, scores):
            real_id = idx_to_item.get(int(sid), int(sid))
            if int(real_id) != item_id:
                results.append({
                    "item_id":        int(real_id),
                    "similarity":     round(float(score), 4)
                })

        return {
            "item_id":      item_id,
            "similar_items": results[:n]
        }

    except Exception as e:
        return {"error": str(e), "item_id": item_id}


# ============================================================
# LINK 6: GET /popular-items
# http://localhost:8000/popular-items
# Returns most popular items — used for new users (cold start)
# ============================================================

@app.get("/popular-items")
def popular_items(category: str = "all", n: int = 10):
    try:
        catalog = pd.read_csv("data/product_catalog.csv")

        # load interaction counts to find popular items
        df        = pd.read_csv("data/cleaned_interactions.csv")
        item_pop  = df.groupby("itemid")["weight"].sum().reset_index()
        item_pop.columns = ["itemid", "total_weight"]

        # merge with catalog to get names
        merged = catalog.merge(item_pop, on="itemid", how="left")
        merged = merged.fillna(0)

        # filter by category if given
        if category != "all":
            merged = merged[
                merged["category"].str.lower() == category.lower()
            ]

        # sort by popularity
        top = merged.nlargest(n, "total_weight")

        results = []
        for _, row in top.iterrows():
            results.append({
                "item_id":    int(row["itemid"]),
                "name":       row["name"],
                "category":   row["category"],
                "price":      f"Rs. {int(row['price'])}",
                "rating":     float(row["rating"]),
                "popularity": int(row["total_weight"])
            })

        return {
            "category": category,
            "items":    results
        }

    except Exception as e:
        return {"error": str(e)}
@app.get("/search")
def search_products(q: str = "", n: int = 6):
    try:
        catalog = pd.read_csv("data/product_catalog.csv")
        if not q:
            return {"results": []}

        mask    = catalog["name"].str.contains(q, case=False, na=False)
        results = catalog[mask].head(20)

        if len(results) == 0:
            return {"results": []}

        # get similar items for first result
        first   = results.iloc[0]
        item_id = int(first["itemid"])

        item_to_idx = np.load(
            "data/item_to_idx.npy", allow_pickle=True
        ).item()

        output = []
        if item_id in item_to_idx:
            item_idx   = item_to_idx[item_id]
            sim_ids, sim_scores = als_model.similar_items(
                itemid=item_idx, N=n+1
            )
            for sid, score in zip(sim_ids, sim_scores):
                real_id = idx_to_item.get(int(sid), int(sid))
                match   = catalog[catalog["itemid"] == real_id]
                if len(match) > 0:
                    row = match.iloc[0]
                    output.append({
                        "item_id":    int(real_id),
                        "name":       row["name"],
                        "category":   row["category"],
                        "price":      f"Rs. {int(row['price'])}",
                        "rating":     float(row["rating"]),
                        "similarity": round(float(score), 4)
                    })

        if not output:
            for _, row in results.head(n).iterrows():
                output.append({
                    "item_id":  int(row["itemid"]),
                    "name":     row["name"],
                    "category": row["category"],
                    "price":    f"Rs. {int(row['price'])}",
                    "rating":   float(row["rating"])
                })

        return {"results": output[:n]}

    except Exception as e:
        return {"error": str(e), "results": []}
    