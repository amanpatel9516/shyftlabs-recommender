# ============================================================
# FILE: data/product_names.py
# JOB: Assign real-looking product names to all item IDs
# RUN: python data/product_names.py
# ============================================================

import pandas as pd
import numpy as np
import random

print("Creating product names for all items...")

# ---- LOAD THE ITEM IDs ----
df = pd.read_csv("data/cleaned_interactions.csv")
all_item_ids = df["itemid"].unique()
print(f"Total items to name: {len(all_item_ids)}")

# ---- PRODUCT NAME PARTS ----
# We will combine these parts to make realistic product names
# like "Nike Running Shoes Blue Medium" or "Whey Protein Chocolate 1kg"

categories = {

    "Shoes": {
        "brands":  ["Nike", "Adidas", "Puma", "Reebok", "Skechers",
                    "New Balance", "Woodland", "Bata", "Sparx", "Campus"],
        "types":   ["Running Shoes", "Sports Shoes", "Casual Sneakers",
                    "Formal Shoes", "Sandals", "Loafers", "Boots",
                    "Slip-ons", "Training Shoes", "Walking Shoes"],
        "colors":  ["Black", "White", "Blue", "Red", "Grey",
                    "Brown", "Navy", "Green"],
        "sizes":   ["Size 6", "Size 7", "Size 8", "Size 9", "Size 10"]
    },

    "Clothing": {
        "brands":  ["Zara", "H&M", "Levi's", "Allen Solly", "Van Heusen",
                    "Peter England", "Arrow", "Raymond", "Wrangler", "Lee"],
        "types":   ["T-Shirt", "Jeans", "Formal Shirt", "Casual Shirt",
                    "Jacket", "Hoodie", "Track Pants", "Shorts",
                    "Polo T-Shirt", "Sweatshirt"],
        "colors":  ["Black", "White", "Blue", "Red", "Grey",
                    "Navy", "Green", "Yellow", "Pink", "Maroon"],
        "sizes":   ["S", "M", "L", "XL", "XXL"]
    },

    "Electronics": {
        "brands":  ["Samsung", "Apple", "Sony", "Boat", "JBL",
                    "Realme", "MI", "OnePlus", "Noise", "boAt"],
        "types":   ["Wireless Earbuds", "Bluetooth Speaker",
                    "Phone Case", "Screen Protector", "Power Bank",
                    "USB Cable", "Headphones", "Smartwatch",
                    "Laptop Stand", "Webcam"],
        "colors":  ["Black", "White", "Blue", "Red", "Silver"],
        "sizes":   ["Standard", "Pro", "Lite", "Plus", "Max"]
    },

    "Fitness": {
        "brands":  ["MuscleBlaze", "Optimum Nutrition", "GNC",
                    "HealthKart", "Decathlon", "Nivia", "Cosco",
                    "Vector X", "SG", "DSC"],
        "types":   ["Whey Protein", "Yoga Mat", "Dumbbells",
                    "Resistance Bands", "Jump Rope", "Gym Gloves",
                    "Protein Bar", "BCAA Supplement", "Creatine",
                    "Water Bottle"],
        "colors":  ["Chocolate", "Vanilla", "Strawberry",
                    "Black", "Blue", "Green", "Red"],
        "sizes":   ["1kg", "2kg", "500g", "Standard", "Pack of 2"]
    },

    "Books": {
        "brands":  ["Penguin", "Harper Collins", "Arihant",
                    "S.Chand", "Oxford", "Wiley", "O'Reilly",
                    "Pearson", "McGraw Hill", "Disha"],
        "types":   ["Python Programming", "Data Structures",
                    "Machine Learning", "System Design",
                    "DSA for Interviews", "GATE Preparation",
                    "Competitive Programming", "Web Development",
                    "Database Systems", "Operating Systems"],
        "colors":  [""],
        "sizes":   ["Paperback", "Hardcover", "Ebook"]
    },

    "Home": {
        "brands":  ["Prestige", "Philips", "Havells", "Bajaj",
                    "Milton", "Borosil", "Cello", "Asian Paints",
                    "Godrej", "Usha"],
        "types":   ["Water Bottle", "Lunch Box", "LED Bulb",
                    "Extension Board", "Table Fan", "Mixer Grinder",
                    "Pressure Cooker", "Non-stick Pan",
                    "Storage Box", "Wall Clock"],
        "colors":  ["Black", "White", "Blue", "Red", "Silver", "Grey"],
        "sizes":   ["Small", "Medium", "Large", "Standard", "Pack of 3"]
    },

    "Beauty": {
        "brands":  ["Lakme", "Maybelline", "L'Oreal", "Nivea",
                    "Dove", "Himalaya", "Biotique", "Mamaearth",
                    "Plum", "WOW"],
        "types":   ["Face Wash", "Moisturizer", "Sunscreen",
                    "Shampoo", "Conditioner", "Hair Oil",
                    "Body Lotion", "Lip Balm", "Serum", "Toner"],
        "colors":  [""],
        "sizes":   ["50ml", "100ml", "200ml", "Pack of 2", "Standard"]
    },

    "Kitchen": {
        "brands":  ["Prestige", "Hawkins", "Pigeon", "TTK",
                    "Wonderchef", "Borosil", "Tupperware",
                    "Milton", "Cello", "Nayasa"],
        "types":   ["Knife Set", "Cutting Board", "Spatula",
                    "Measuring Cups", "Colander", "Vegetable Peeler",
                    "Can Opener", "Grater", "Mixing Bowl",
                    "Baking Tray"],
        "colors":  ["Black", "Silver", "Red", "Blue", "White"],
        "sizes":   ["Small", "Medium", "Large", "Set of 3", "Standard"]
    }
}

# ---- PRICE RANGES PER CATEGORY ----
price_ranges = {
    "Shoes":       (499,  4999),
    "Clothing":    (299,  2999),
    "Electronics": (199,  9999),
    "Fitness":     (299,  3999),
    "Books":       (99,   899),
    "Home":        (99,   1999),
    "Beauty":      (99,   999),
    "Kitchen":     (99,   1499)
}

# ---- RATING RANGES ----
# higher priced items tend to have better ratings
rating_ranges = {
    "Shoes":       (3.8, 4.9),
    "Clothing":    (3.5, 4.8),
    "Electronics": (3.6, 4.9),
    "Fitness":     (3.7, 4.8),
    "Books":       (4.0, 4.9),
    "Home":        (3.5, 4.7),
    "Beauty":      (3.6, 4.8),
    "Kitchen":     (3.5, 4.7)
}

# ---- ASSIGN A NAME TO EVERY ITEM ID ----
random.seed(42)   # same seed = same names every time you run
np.random.seed(42)

category_names = list(categories.keys())
product_data   = []

for item_id in all_item_ids:
    # pick a random category for this item
    cat_name = random.choice(category_names)
    cat      = categories[cat_name]

    # pick random parts
    brand = random.choice(cat["brands"])
    ptype = random.choice(cat["types"])
    color = random.choice(cat["colors"])
    size  = random.choice(cat["sizes"])

    # build the product name
    if color:
        name = f"{brand} {ptype} - {color} ({size})"
    else:
        name = f"{brand} {ptype} ({size})"

    # assign a price
    low, high = price_ranges[cat_name]
    price = random.randint(low // 10, high // 10) * 10

    # assign a rating
    low_r, high_r = rating_ranges[cat_name]
    rating = round(random.uniform(low_r, high_r), 1)

    product_data.append({
        "itemid":    item_id,
        "name":      name,
        "category":  cat_name,
        "price":     price,
        "rating":    rating
    })

# ---- SAVE TO CSV ----
product_df = pd.DataFrame(product_data)
product_df.to_csv("data/product_catalog.csv", index=False)

print(f"Created names for {len(product_df)} products!")
print("\nSample products:")
print(product_df.head(10).to_string(index=False))
print("\nSaved: data/product_catalog.csv")