# ============================================================
# FILE: data/preprocess.py
# JOB: Read the raw data, clean it, save it ready for ML
# ============================================================

import pandas as pd
import numpy as np
import os

# ---- STEP 1: READ THE RAW FILE ----
# pd.read_csv = open a CSV file and load it into Python
# This file has all user clicks, views, purchases
print("Reading data file...")
df = pd.read_csv("data/events.csv")

# Let's see what the data looks like first
print("Shape of data:", df.shape)       # how many rows and columns
print("Column names:", df.columns.tolist())
print("First 3 rows:")
print(df.head(3))


# ---- STEP 2: UNDERSTAND THE COLUMNS ----
# The events.csv file has these columns:
#   timestamp   - when the event happened (a number like 1433221332117)
#   visitorid   - which user did the action (like user number 12345)
#   event       - what they did: "view", "addtocart", or "transaction"
#   itemid      - which product they interacted with
#   transactionid - only filled if they bought something


# ---- STEP 3: REMOVE BAD ROWS ----
# Drop rows where visitorid or itemid is missing (NaN = empty cell)
print("\nRemoving rows with missing values...")
before = len(df)
df = df.dropna(subset=["visitorid", "itemid"])
after = len(df)
print(f"Removed {before - after} bad rows. Remaining: {after} rows")


# ---- STEP 4: KEEP ONLY USEFUL COLUMNS ----
# We don't need transactionid for our model
df = df[["visitorid", "itemid", "event", "timestamp"]]


# ---- STEP 5: GIVE WEIGHTS TO EACH ACTION ----
# Not all actions are equal:
#   view       = user just looked     → small interest  → weight 1
#   addtocart  = user almost bought   → medium interest → weight 3
#   transaction= user actually bought → strong interest → weight 5
#
# Why? Because if we treat all actions same, model won't learn properly.
# A purchase means much more than just a view.

print("\nAdding weight scores to events...")

# Create a dictionary: event name → weight number
event_weights = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5
}

# Map each event to its weight number
# .map() goes through each row and replaces event name with number
df["weight"] = df["event"].map(event_weights)

# If any event type is unknown, fill with 1 (safe default)
df["weight"] = df["weight"].fillna(1)


# ---- STEP 6: COMBINE DUPLICATE INTERACTIONS ----
# Same user may view same item 5 times
# Instead of 5 rows, we want 1 row with total weight = 5
# groupby = group all rows with same (user, item) together
# sum = add up all their weights

print("Combining multiple interactions from same user-item pair...")
df_grouped = df.groupby(
    ["visitorid", "itemid"],
    as_index=False   # keep visitorid and itemid as columns, not index
)["weight"].sum()

print(f"Unique user-item pairs: {len(df_grouped)}")


# ---- STEP 7: GIVE USERS AND ITEMS SIMPLE ID NUMBERS ----
# Our model needs numbers like 0, 1, 2, 3...
# But visitorid and itemid are big random numbers like 1452384
# So we convert them to simple 0-based numbers

print("\nCreating simple ID mappings...")

# Get list of all unique users and items
unique_users = df_grouped["visitorid"].unique()
unique_items = df_grouped["itemid"].unique()

# Create dictionaries: original_id → simple_id
# Example: {1452384: 0, 9823741: 1, 3847291: 2, ...}
user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

# Reverse dictionaries: simple_id → original_id
# We need this later to show real item IDs in results
idx_to_user = {idx: user for user, idx in user_to_idx.items()}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}

# Add the simple IDs to our dataframe
df_grouped["user_idx"] = df_grouped["visitorid"].map(user_to_idx)
df_grouped["item_idx"] = df_grouped["itemid"].map(item_to_idx)

print(f"Total unique users: {len(unique_users)}")
print(f"Total unique items: {len(unique_items)}")


# ---- STEP 8: SAVE EVERYTHING ----
# Save cleaned data as a CSV file
print("\nSaving cleaned data...")
df_grouped.to_csv("data/cleaned_interactions.csv", index=False)

# Save the ID mapping dictionaries using numpy
# We need these later when running the model
np.save("data/user_to_idx.npy", user_to_idx)
np.save("data/item_to_idx.npy", item_to_idx)
np.save("data/idx_to_user.npy", idx_to_user)
np.save("data/idx_to_item.npy", idx_to_item)

print("Saved: data/cleaned_interactions.csv")
print("Saved: data/user_to_idx.npy")
print("Saved: data/item_to_idx.npy")
print("All done! Data is ready for ML model.")


# ---- STEP 9: SHOW SUMMARY ----
print("\n===== DATA SUMMARY =====")
print(f"Total interactions: {len(df_grouped)}")
print(f"Total users: {len(unique_users)}")
print(f"Total items: {len(unique_items)}")
print("\nEvent breakdown (original data):")
print(df["event"].value_counts())
print("\nSample of cleaned data:")
print(df_grouped.head(5))