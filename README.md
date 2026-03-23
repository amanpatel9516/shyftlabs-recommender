# ShyftLabs AI Recommendation and AdTech System

A real-time personalization and ad targeting system built 
for SHYFTLABS placement drive 2027.

## What This Does

- Recommends products to 1.4M users from 235K items
- Smart ad targeting — 133% better CTR than random
- User intent scoring — hot, warm, cold segmentation
- Cold start handling for new users
- Live dashboard with business metrics

## Tech Stack

- ALS Matrix Factorization (implicit library)
- LightGBM Ranker
- Thompson Sampling Bandit
- FastAPI backend
- Streamlit + HTML dashboard

## Dataset

RetailRocket E-commerce Dataset
- 2.7M user interactions
- 1.4M unique users  
- 235K unique products

## How to Run

### Install dependencies
pip install pandas numpy implicit lightgbm fastapi 
uvicorn streamlit scikit-learn requests plotly

### Prepare data
Download RetailRocket dataset from Kaggle
Place events.csv in data/ folder
python data/preprocess.py

### Train models
python models/als_model.py
python models/ranker.py
python models/bandit.py

### Start API (Terminal 1)
uvicorn api.main:app --reload

### Start Dashboard (Terminal 2)
streamlit run dashboard/app.py

### Open in browser
Dashboard: http://localhost:8501
API docs:  http://localhost:8000/docs

## Business Impact

| Metric | Value |
|--------|-------|
| Ad CTR with ML | 28% |
| Random baseline | 12% |
| Improvement | 133% |
| Users trained on | 1.4M |
| Products indexed | 235K |

## Key Design Decisions

**Why ALS over SVD?**
Our data is implicit — users never rated products.
ALS handles implicit feedback. SVD needs explicit ratings.

**Why Thompson Sampling?**
Learns which ad wins while still exploring alternatives.
Achieved 133% CTR improvement over random selection.

**Why LightGBM for ranking?**
ALS gives 50 candidates. LightGBM picks best 10 using
popularity and purchase rate features. Fast and accurate.
