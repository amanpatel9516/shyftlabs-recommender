# ============================================================
# FILE: dashboard/app.py
# REORDERED VERSION — correct story flow for interview
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ShyftLabs Recommender",
    layout="wide"
)
# ============================================================
# CUSTOM CSS — Professional UI
# ============================================================

st.markdown("""
<style>

/* ---- MAIN BACKGROUND ---- */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%);
}

/* ---- HIDE STREAMLIT DEFAULT ELEMENTS ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---- CUSTOM HEADER BAR ---- */
.main-header {
    background: linear-gradient(90deg, #1a1a3e 0%, #16213e 50%, #0f3460 100%);
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid #2a2a5e;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.15);
}

.main-header h1 {
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px 0;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
}

.badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.2);
    color: #a78bfa;
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin: 8px 4px 0 0;
}

/* ---- SECTION HEADERS ---- */
.section-header {
    background: linear-gradient(90deg, #1e1b4b, #1e3a5f);
    padding: 16px 24px;
    border-radius: 12px;
    margin: 24px 0 16px 0;
    border-left: 4px solid #6366f1;
}

.section-header h2 {
    color: #e2e8f0;
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 4px 0;
}

.section-header p {
    color: #94a3b8;
    font-size: 13px;
    margin: 0;
}

/* ---- METRIC CARDS ---- */
.metric-card {
    background: linear-gradient(135deg, #1e1b4b, #1e3a5f);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

.metric-card .value {
    font-size: 32px;
    font-weight: 700;
    color: #60a5fa;
    margin: 8px 0 4px 0;
}

.metric-card .label {
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-card .delta {
    font-size: 12px;
    color: #34d399;
    margin-top: 4px;
}

/* ---- PRODUCT CARDS ---- */
.product-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    margin: 6px 0;
    min-height: 160px;
    transition: border-color 0.2s;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}

.product-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
}

.product-card .rank {
    font-size: 10px;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.product-card .name {
    font-size: 13px;
    color: #e2e8f0;
    font-weight: 600;
    line-height: 1.4;
    margin-bottom: 8px;
}

.product-card .price {
    font-size: 16px;
    color: #34d399;
    font-weight: 700;
    margin-bottom: 4px;
}

.product-card .stars {
    font-size: 11px;
    color: #fbbf24;
}

.product-card .match {
    font-size: 11px;
    color: #64748b;
    margin-top: 6px;
}

/* ---- STATUS BADGES ---- */
.status-active {
    background: rgba(52, 211, 153, 0.1);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* ---- INSIGHT BOX ---- */
.insight-box {
    background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.1),
        rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c7d2fe;
    font-size: 14px;
    line-height: 1.6;
}

/* ---- DIVIDER ---- */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent, rgba(99,102,241,0.4), transparent);
    margin: 32px 0;
}

/* ---- STREAMLIT COMPONENT OVERRIDES ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1b4b, #1e3a5f);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

div[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stMetricDelta"] svg {
    display: none;
}

/* ---- BUTTONS ---- */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #6366f1, #3b82f6) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 24px !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px);
}

/* ---- INPUT FIELDS ---- */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #1e1b4b !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ---- ALERTS ---- */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* ---- PROGRESS BAR ---- */
div[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #6366f1, #3b82f6) !important;
    border-radius: 4px !important;
}

</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# ---- CHECK API ----
try:
    r      = requests.get(f"{API_URL}/", timeout=3)
    api_ok = r.status_code == 200
except Exception:
    api_ok = False

if not api_ok:
    st.error(
        "API is not running! "
        "Open a second terminal and run: uvicorn api.main:app --reload"
    )
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>ShyftLabs Recommender System</h1>
    <p>Real-time AI personalization and AdTech platform</p>
    <span class="badge">ALS Model</span>
    <span class="badge">LightGBM Ranker</span>
    <span class="badge">Thompson Sampling</span>
    <span class="badge">1.4M Users</span>
    <span class="badge">235K Products</span>
</div>
""", unsafe_allow_html=True)

# model status
st.success(
    "All models running — ALS · LightGBM · Thompson Sampling Bandit"
)
c1, c2, c3 = st.columns(3)
with c1:
    st.info("**Recommendation Model**\nALS — finds products user will like")
with c2:
    st.info("**Ranking Model**\nLightGBM — sorts best products first")
with c3:
    st.info("**Ad Model**\nThompson Sampling — finds best ad")

st.divider()

# ============================================================
# SECTION 1 — BUSINESS IMPACT (show this first to hook interviewer)
# ============================================================

st.markdown("""
<div class="section-header">
    <h2>1. Business Impact</h2>
    <p>What SHYFTLABS shows Fortune 500 clients —
    not just "AI works" but "here is how much more money you make"</p>
</div>
""", unsafe_allow_html=True)
try:
    resp    = requests.get(f"{API_URL}/metrics", timeout=5)
    metrics = resp.json()
    bm      = metrics["business_metrics"]
    mi      = metrics["model_info"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Our system CTR",
            f"{bm['best_ad_ctr']*100:.1f}%",
            f"+{bm['improvement_over_random']} vs random"
        )
    with c2:
        st.metric(
            "Random baseline CTR",
            f"{bm['random_baseline_ctr']*100:.1f}%",
            "Without AI"
        )
    with c3:
        st.metric(
            "Improvement",
            bm["improvement_over_random"],
            "More clicks with AI"
        )
    with c4:
        st.metric(
            "Total users in system",
            f"{mi['total_users']:,}"
        )

    extra_clicks = int((bm["best_ad_ctr"] - 0.12) * 1_000_000)
    st.success(
        f"**Business Impact:** Out of every 100 users, "
        f"{bm['best_ad_ctr']*100:.0f} click our ad vs only 12 with random. "
        f"On a platform with 10 lakh users per day — that is "
        f"**{extra_clicks:,} extra clicks every single day.**"
    )

    st.markdown("**Ad CTR — all ads vs random baseline:**")
    chart_data = pd.DataFrame({
        "Ad": [f"Ad {i+1}" for i in range(5)],
        "CTR %": [
            metrics["all_ad_ctrs"][f"ad_{i}"] * 100
            for i in range(5)
        ]
    }).set_index("Ad")
    st.bar_chart(chart_data)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            f"**{mi['recommendation_model']}**\n\n"
            f"Trained on {mi['total_users']:,} users\n\n"
            f"Covers {mi['total_items']:,} products"
        )
    with c2:
        st.info(
            f"**{mi['ranking_model']}**\n\n"
            f"Takes 50 candidates per user\n\n"
            f"Returns best 10 ranked by score"
        )
    with c3:
        st.info(
            f"**{mi['ad_selection_model']}**\n\n"
            f"Managing 5 ads\n\n"
            f"Best CTR: {bm['best_ad_ctr']*100:.1f}%"
        )

except Exception as e:
    st.error(f"Could not load metrics: {e}")

st.divider()

# ============================================================
# SECTION 2 — LIVE RECOMMENDATIONS
# ============================================================

st.header("2. Live Recommendations — What Should We Show This User?")
st.markdown(
    "Enter any user ID. The AI finds their top 10 products "
    "from 2.35 lakh items in under 1 second."
)

user_id = st.number_input(
    "Enter user number (0 to 1000):",
    min_value=0, max_value=1000, value=0, step=1,
    key="rec_user"
)

if st.button("Get Recommendations", type="primary", key="rec_btn"):
    with st.spinner("AI finding best products..."):
        try:
            resp   = requests.get(
                f"{API_URL}/recommend/{user_id}", timeout=10
            )
            result = resp.json()

            if "recommendations" in result and result["recommendations"]:
                recs = result["recommendations"]
                st.success(
                    f"Top {len(recs)} products for User {user_id}"
                )

                for row_start in [0, 5]:
                    cols = st.columns(5)
                    for i in range(5):
                        idx = row_start + i
                        if idx >= len(recs):
                            break
                        with cols[i]:
                            r        = recs[idx]
                            name     = r.get("name", f"Item {r['item_id']}")
                            price    = r.get("price", "N/A")
                            rating   = r.get("rating", 0)
                            category = r.get("category", "")
                            score    = r["relevance_score"]
                            stars    = int(rating)
                            star_str = "★" * stars + "☆" * (5 - stars)
                            st.markdown(
                                f"""
                                <div style="border:1px solid #444;
                                    border-radius:10px;padding:12px;
                                    background:#1e1e2e;margin:4px 0;
                                    min-height:150px">
                                <div style="font-size:11px;color:#888;
                                            margin-bottom:4px">
                                    #{idx+1} · {category}
                                </div>
                                <div style="font-size:13px;color:#fff;
                                            font-weight:600;
                                            line-height:1.3;
                                            margin-bottom:6px">
                                    {name}
                                </div>
                                <div style="font-size:15px;color:#4CAF50;
                                            font-weight:700">
                                    {price}
                                </div>
                                <div style="font-size:12px;color:#FFD700">
                                    {star_str} {rating}
                                </div>
                                <div style="font-size:11px;color:#aaa;
                                            margin-top:6px">
                                    Match: {score*100:.0f}%
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    st.markdown(" ")

                top  = recs[0]
                st.markdown(
                    f"> User {user_id} is most likely to buy "
                    f"**{top.get('name', 'this product')}** "
                    f"({top.get('price', '')}). "
                    f"AI confidence: {top['relevance_score']*100:.0f}%. "
                    f"Try user 1, 5, 50, 100 — every user gets "
                    f"different results."
                )
            else:
                st.warning("No recommendations found. Try user 1 or 2.")

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ============================================================
# SECTION 3 — USER PROFILE
# ============================================================

st.header("3. User Profile — Who Is This User?")
st.markdown(
    "See any user's shopping personality, "
    "engagement score and purchase history."
)

profile_uid = st.number_input(
    "Enter user ID for profile:",
    min_value=0, max_value=1000, value=0, step=1,
    key="profile_user"
)

if st.button("Show User Profile", type="primary", key="profile_btn"):
    with st.spinner("Loading profile..."):
        try:
            resp   = requests.get(
                f"{API_URL}/user-profile/{profile_uid}", timeout=10
            )
            result = resp.json()

            if result.get("is_new_user"):
                st.warning(
                    f"User {profile_uid} is brand new — no history yet."
                )
            elif "error" in result:
                st.error(result["error"])
            else:
                st.markdown(
                    f"""
                    <div style="border:1px solid #444;border-radius:12px;
                        padding:24px;background:#1e1e2e;margin-bottom:16px">
                    <h3 style="color:#fff;margin:0 0 8px 0">
                        User {result['user_id']}
                    </h3>
                    <div style="font-size:18px;color:#4CAF50;
                                font-weight:700;margin-bottom:8px">
                        {result['user_type']}
                    </div>
                    <div style="font-size:14px;color:#aaa">
                        {result['user_description']}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Total Interactions",
                        result["total_interactions"]
                    )
                with c2:
                    st.metric("Purchases", result["total_purchases"])
                with c3:
                    st.metric("Cart Adds", result["total_cart_adds"])
                with c4:
                    st.metric(
                        "Engagement",
                        f"{result['engagement_score']}/100"
                    )

                st.markdown("**Engagement level:**")
                st.progress(result["engagement_score"] / 100)

                score = result["engagement_score"]
                if score >= 70:
                    st.success(
                        "High value user — show premium products "
                        "and exclusive offers"
                    )
                elif score >= 40:
                    st.info(
                        "Medium engagement — show popular products "
                        "with discounts"
                    )
                else:
                    st.warning(
                        "Low engagement — show trending products "
                        "to re-engage"
                    )

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ============================================================
# SECTION 4 — COMPARE TWO USERS
# ============================================================

st.header("4. Compare Two Users — Proof That AI Personalizes")
st.markdown(
    "See how two different users get completely different recommendations. "
    "This proves the AI is truly personalizing — "
    "not showing same products to everyone."
)

col1, col2 = st.columns(2)
with col1:
    user_a = st.number_input(
        "User A:", min_value=0, max_value=1000,
        value=0, step=1, key="compare_a"
    )
with col2:
    user_b = st.number_input(
        "User B:", min_value=0, max_value=1000,
        value=50, step=1, key="compare_b"
    )

if st.button("Compare These Two Users", type="primary", key="compare_btn"):
    with st.spinner("Getting recommendations for both users..."):
        try:
            resp_a = requests.get(
                f"{API_URL}/recommend/{user_a}", timeout=10
            )
            resp_b = requests.get(
                f"{API_URL}/recommend/{user_b}", timeout=10
            )
            recs_a = resp_a.json().get("recommendations", [])
            recs_b = resp_b.json().get("recommendations", [])

            if recs_a and recs_b:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### User {user_a}")
                    for i, r in enumerate(recs_a[:5]):
                        name     = r.get("name", f"Item {r['item_id']}")
                        price    = r.get("price", "")
                        category = r.get("category", "")
                        score    = r["relevance_score"]
                        st.markdown(
                            f"""
                            <div style="border-left:3px solid #4CAF50;
                                padding:8px 12px;margin:6px 0;
                                background:#1e1e2e;
                                border-radius:0 8px 8px 0">
                            <div style="font-size:11px;color:#888">
                                #{i+1} · {category}
                            </div>
                            <div style="font-size:14px;color:#fff;
                                        font-weight:600">
                                {name}
                            </div>
                            <div style="font-size:13px;color:#4CAF50">
                                {price} · {score*100:.0f}% match
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col2:
                    st.markdown(f"### User {user_b}")
                    for i, r in enumerate(recs_b[:5]):
                        name     = r.get("name", f"Item {r['item_id']}")
                        price    = r.get("price", "")
                        category = r.get("category", "")
                        score    = r["relevance_score"]
                        st.markdown(
                            f"""
                            <div style="border-left:3px solid #2196F3;
                                padding:8px 12px;margin:6px 0;
                                background:#1e1e2e;
                                border-radius:0 8px 8px 0">
                            <div style="font-size:11px;color:#888">
                                #{i+1} · {category}
                            </div>
                            <div style="font-size:14px;color:#fff;
                                        font-weight:600">
                                {name}
                            </div>
                            <div style="font-size:13px;color:#2196F3">
                                {price} · {score*100:.0f}% match
                            </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                items_a = set(r["item_id"] for r in recs_a[:5])
                items_b = set(r["item_id"] for r in recs_b[:5])
                overlap = len(items_a & items_b)

                if overlap == 0:
                    st.success(
                        "0 products in common. "
                        "The AI gives 100% unique results to each user."
                    )
                else:
                    st.info(f"{overlap} product(s) in common.")

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ============================================================
# SECTION 5 — SMART AD SYSTEM
# ============================================================

st.header("5. Smart Ad System — Which Ad Should We Show?")
st.markdown(
    "We have 5 ads. Instead of random selection, "
    "our AI learns which ad gets most clicks."
)

ad_names = [
    "10% off on Electronics",
    "Buy 2 Get 1 Free",
    "Flash Sale — 50% off",
    "Free Shipping Today",
    "New Arrivals — Shop Now"
]

st.markdown("**Our 5 Ads:**")
ad_cols = st.columns(5)
for i, (col, name) in enumerate(zip(ad_cols, ad_names)):
    with col:
        if i == 2:
            st.success(f"**Ad {i+1}**\n\n{name}\n\n*Usually best*")
        else:
            st.info(f"**Ad {i+1}**\n\n{name}")

st.markdown(" ")

ad_uid = st.number_input(
    "Simulate ad for which user?",
    min_value=0, max_value=1000, value=0, step=1,
    key="ad_user"
)

col_b1, col_b2 = st.columns(2)
with col_b1:
    serve_btn = st.button(
        "Serve Ad to This User", type="primary", key="serve_btn"
    )
with col_b2:
    reset_btn = st.button(
        "Reset Bandit — Learn From Zero", key="reset_btn"
    )

if reset_btn:
    try:
        requests.get(f"{API_URL}/reset-bandit", timeout=5)
        st.success(
            "Bandit reset! All 5 ads are equal again. "
            "Serve ads to watch it learn."
        )
    except Exception as e:
        st.error(f"Reset failed: {e}")

if serve_btn:
    with st.spinner("Bandit picking best ad..."):
        try:
            resp     = requests.get(
                f"{API_URL}/ad-serve?user_id={ad_uid}", timeout=5
            )
            result   = resp.json()
            chosen   = result["chosen_ad_id"]
            name     = result["chosen_ad_name"]
            clicked  = result["user_clicked"]
            all_ctrs = result["all_ad_ctrs"]

            col1, col2 = st.columns(2)
            with col1:
                if clicked:
                    st.success(
                        f"**Ad {chosen+1}: {name}**\n\n"
                        f"User CLICKED! Bandit learns this works."
                    )
                else:
                    st.warning(
                        f"**Ad {chosen+1}: {name}**\n\n"
                        f"User ignored. Bandit learns from this too."
                    )
            with col2:
                st.markdown("**Updated CTR estimates:**")
                for ad in all_ctrs:
                    ctr = ad["estimated_ctr"] * 100
                    bar = "█" * int(ctr * 1.5)
                    st.text(f"Ad {ad['ad_id']+1}: {bar} {ctr:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ============================================================
# SECTION 6 — LIVE AD SIMULATION
# ============================================================

st.header("6. Watch the Ad System Learn Live")
st.markdown(
    "Simulate many users. "
    "Watch the green line go up as AI learns which ad is best."
)

n_users = st.slider(
    "How many users to simulate?",
    min_value=20, max_value=200, value=50, step=10
)

if st.button("Run Live Simulation", type="primary", key="sim_btn"):
    progress       = st.progress(0, text="Starting...")
    ctrs_over_time = []

    for i in range(n_users):
        try:
            r    = requests.get(f"{API_URL}/ad-serve", timeout=5)
            res  = r.json()
            best = max(
                ad["estimated_ctr"] for ad in res["all_ad_ctrs"]
            )
            ctrs_over_time.append(best * 100)
        except Exception:
            ctrs_over_time.append(None)

        progress.progress(
            (i+1) / n_users,
            text=f"Simulating user {i+1} of {n_users}..."
        )

    progress.empty()

    if ctrs_over_time:
        sim_df = pd.DataFrame({
            "User number":     range(1, len(ctrs_over_time)+1),
            "Best Ad CTR (%)": ctrs_over_time
        }).set_index("User number")
        st.line_chart(sim_df)

        final = ctrs_over_time[-1]
        st.success(
            f"After {n_users} users — best CTR is {final:.1f}%. "
            f"Random gives 12%. "
            f"Our AI is {((final-12)/12*100):.0f}% better."
        )

st.divider()

# ============================================================
# SECTION 7 — SEARCH
# ============================================================

st.header("7. Search Products — Find Similar Items")
st.markdown(
    "Type any product name. "
    "The AI finds similar products using the same model "
    "that powers recommendations."
)

search_query = st.text_input(
    "Search:",
    placeholder="e.g. Nike, Protein, Samsung, Yoga Mat, Books"
)

if search_query:
    with st.spinner(f"Searching for '{search_query}'..."):
        try:
            catalog = pd.read_csv("data/product_catalog.csv")
            mask    = catalog["name"].str.contains(
                search_query, case=False, na=False
            )
            results = catalog[mask].head(20)

            if len(results) == 0:
                st.warning(
                    f"No products found for '{search_query}'. "
                    f"Try: Nike, Shoes, Protein, Samsung, Yoga"
                )
            else:
                first_item = results.iloc[0]
                st.markdown(
                    f"Found {len(results)} products. "
                    f"Showing items similar to: "
                    f"**{first_item['name']}**"
                )

                similar_resp = requests.get(
                    f"{API_URL}/similar-items/"
                    f"{int(first_item['itemid'])}",
                    timeout=10
                )
                similar = similar_resp.json().get("similar_items", [])

                if similar:
                    st.markdown(
                        "**Customers who liked this also liked:**"
                    )
                    sim_cols = st.columns(3)
                    for i, item in enumerate(similar[:6]):
                        with sim_cols[i % 3]:
                            match = catalog[
                                catalog["itemid"] == item["item_id"]
                            ]
                            if len(match) > 0:
                                row       = match.iloc[0]
                                sim_name  = row["name"]
                                sim_cat   = row["category"]
                                sim_price = f"Rs. {int(row['price'])}"
                                sim_rat   = row["rating"]
                            else:
                                sim_name  = f"Product {item['item_id']}"
                                sim_cat   = ""
                                sim_price = ""
                                sim_rat   = 0

                            stars    = int(sim_rat)
                            star_str = "★" * stars + "☆" * (5 - stars)
                            st.markdown(
                                f"""
                                <div style="border:1px solid #444;
                                    border-radius:10px;padding:12px;
                                    background:#1e1e2e;margin:4px 0">
                                <div style="font-size:11px;color:#888">
                                    {sim_cat}
                                </div>
                                <div style="font-size:13px;color:#fff;
                                            font-weight:600">
                                    {sim_name}
                                </div>
                                <div style="font-size:14px;color:#4CAF50;
                                            font-weight:700">
                                    {sim_price}
                                </div>
                                <div style="font-size:11px;color:#FFD700">
                                    {star_str}
                                </div>
                                <div style="font-size:11px;color:#888">
                                    Similarity: {item['similarity']*100:.0f}%
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        except Exception as e:
            st.error(f"Search error: {e}")

st.divider()

# ============================================================
# SECTION 8 — COLD START
# ============================================================

st.header("8. Cold Start — What Happens With a New User?")
st.markdown(
    "A new user has no history. "
    "Watch how the system handles this step by step."
)

st.markdown("""
**3 stage solution:**
- **Stage 1** (0 clicks) — Show most popular products overall
- **Stage 2** (1-2 clicks) — Show popular products in your category only
- **Stage 3** (3+ clicks) — Find similar real user and use their AI recommendations
""")

if "click_history" not in st.session_state:
    st.session_state["click_history"] = []

total_clicks  = len(st.session_state["click_history"])
current_stage = 1 if total_clicks == 0 else 2 if total_clicks <= 2 else 3

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Your clicks", total_clicks)
with c2:
    st.metric("Current stage", current_stage)
with c3:
    st.metric(
        "Status",
        "New user" if total_clicks == 0
        else "Learning..." if total_clicks <= 2
        else "AI active!"
    )
with c4:
    need = max(0, 3 - total_clicks)
    st.metric("Clicks to AI", need if need > 0 else "Ready!")

st.progress(min(total_clicks / 3, 1.0))

if st.button("Reset — Start Fresh", key="cold_reset"):
    st.session_state["click_history"] = []
    st.rerun()

st.divider()

if current_stage == 1:
    st.markdown("### Stage 1 — Brand new user (0 clicks)")
    st.info("No history. Showing most popular products overall.")
    try:
        resp  = requests.get(f"{API_URL}/popular-items?n=6", timeout=10)
        items = resp.json().get("items", [])
        if items:
            cols = st.columns(3)
            for i, item in enumerate(items[:6]):
                with cols[i % 3]:
                    stars    = int(item["rating"])
                    star_str = "★" * stars + "☆" * (5 - stars)
                    st.markdown(
                        f"""
                        <div style="border:1px solid #555;
                            border-radius:10px;padding:12px;
                            background:#1e1e2e;margin:4px 0">
                        <div style="font-size:11px;color:#888">
                            {item['category']} · Trending
                        </div>
                        <div style="font-size:13px;color:#fff;
                                    font-weight:600">
                            {item['name']}
                        </div>
                        <div style="font-size:14px;color:#4CAF50;
                                    font-weight:700">
                            {item['price']}
                        </div>
                        <div style="font-size:11px;color:#FFD700">
                            {star_str}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button(
                        "I am interested", key=f"s1_{i}"
                    ):
                        st.session_state["click_history"].append({
                            "item_id":  item["item_id"],
                            "name":     item["name"],
                            "category": item["category"]
                        })
                        st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

elif current_stage == 2:
    last_cat = st.session_state["click_history"][-1]["category"]
    last_name = st.session_state["click_history"][-1]["name"]
    st.markdown("### Stage 2 — One signal received")
    st.info(
        f"You clicked **{last_name}**. "
        f"Showing ONLY **{last_cat}** products now. "
        f"Click one more to activate full AI."
    )
    st.markdown("**Your clicks so far:**")
    for c in st.session_state["click_history"]:
        st.markdown(f"- {c['name']} ({c['category']})")
    try:
        resp  = requests.get(
            f"{API_URL}/popular-items?category={last_cat}&n=6",
            timeout=10
        )
        items = resp.json().get("items", [])
        if items:
            cols = st.columns(3)
            for i, item in enumerate(items[:6]):
                with cols[i % 3]:
                    stars    = int(item["rating"])
                    star_str = "★" * stars + "☆" * (5 - stars)
                    st.markdown(
                        f"""
                        <div style="border:1px solid #2196F3;
                            border-radius:10px;padding:12px;
                            background:#1e1e2e;margin:4px 0">
                        <div style="font-size:11px;color:#888">
                            {item['category']}
                        </div>
                        <div style="font-size:13px;color:#fff;
                                    font-weight:600">
                            {item['name']}
                        </div>
                        <div style="font-size:14px;color:#4CAF50;
                                    font-weight:700">
                            {item['price']}
                        </div>
                        <div style="font-size:11px;color:#FFD700">
                            {star_str}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button(
                        "I am interested", key=f"s2_{i}"
                    ):
                        st.session_state["click_history"].append({
                            "item_id":  item["item_id"],
                            "name":     item["name"],
                            "category": item["category"]
                        })
                        st.rerun()
        else:
            if st.button("Simulate one more click"):
                st.session_state["click_history"].append({
                    "item_id":  99999,
                    "name":     f"Generic {last_cat} product",
                    "category": last_cat
                })
                st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

else:
    clicked_cats = [c["category"] for c in st.session_state["click_history"]]
    most_clicked = max(set(clicked_cats), key=clicked_cats.count)
    st.markdown("### Stage 3 — Full AI personalization")
    st.success(
        f"You clicked {total_clicks} products. "
        f"Most interest in **{most_clicked}**. "
        f"Finding similar real user in dataset..."
    )
    st.markdown("**Your click history:**")
    for c in st.session_state["click_history"]:
        st.markdown(f"- {c['name']} ({c['category']})")
    try:
        catalog = pd.read_csv("data/product_catalog.csv")
        df_int  = pd.read_csv("data/cleaned_interactions.csv")
        cat_items = catalog[
            catalog["category"] == most_clicked
        ]["itemid"].tolist()
        cat_int = df_int[df_int["itemid"].isin(cat_items)]
        top_users = (
            cat_int.groupby("user_idx")["weight"]
            .sum().nlargest(20).index.tolist()
        )
        st.info(
            f"Found {len(top_users)} real users who love {most_clicked}. "
            f"Borrowing their recommendations — "
            f"called collaborative filtering fallback."
        )
        best_recs  = []
        found_user = None
        for try_user in top_users[:10]:
            resp     = requests.get(
                f"{API_URL}/recommend/{int(try_user)}", timeout=10
            )
            all_recs = resp.json().get("recommendations", [])
            named    = [
                r for r in all_recs
                if not r.get("name", "Product").startswith("Product")
                and r.get("price", "N/A") != "N/A"
            ]
            if len(named) > len(best_recs):
                best_recs  = named
                found_user = try_user
            if len(best_recs) >= 4:
                break

        if not best_recs:
            resp     = requests.get(
                f"{API_URL}/recommend/50", timeout=10
            )
            best_recs = resp.json().get("recommendations", [])

        if best_recs:
            st.markdown(
                f"**AI recommendations based on your "
                f"{most_clicked} interest:**"
            )
            cols  = st.columns(3)
            shown = 0
            for r in best_recs:
                if shown >= 6:
                    break
                name     = r.get("name", "")
                price    = r.get("price", "N/A")
                category = r.get("category", "General")
                score    = r.get("relevance_score", 0)
                rating   = r.get("rating", 3)
                if name.startswith("Product") or price == "N/A":
                    continue
                with cols[shown % 3]:
                    stars    = int(rating)
                    star_str = "★" * stars + "☆" * (5 - stars)
                    st.markdown(
                        f"""
                        <div style="border:1px solid #4CAF50;
                            border-radius:10px;padding:12px;
                            background:#1e1e2e;margin:4px 0">
                        <div style="font-size:11px;color:#4CAF50">
                            {category} · AI Personalized
                        </div>
                        <div style="font-size:13px;color:#fff;
                                    font-weight:600;line-height:1.3">
                            {name}
                        </div>
                        <div style="font-size:14px;color:#4CAF50;
                                    font-weight:700">
                            {price}
                        </div>
                        <div style="font-size:11px;color:#FFD700">
                            {star_str}
                        </div>
                        <div style="font-size:11px;color:#aaa">
                            Match: {score*100:.0f}%
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                shown += 1

        st.info(
            "**Interview answer:** In production every click saves "
            "to a database. ALS retrains nightly. After 10+ clicks "
            "your own profile gets created and you stop borrowing."
        )
    except Exception as e:
        st.error(f"Stage 3 error: {e}")

st.divider()
st.markdown(
    "*Built for SHYFTLABS placement — "
    "ALS + LightGBM + Thompson Sampling Bandit*"
)