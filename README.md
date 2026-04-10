Team Convergence - PS1
---

## What This Does

A full-stack fraud detection system built on Flask + vanilla JS:

| Component | File | Purpose |
|---|---|---|
| **API Backend** | `app.py` | Flask REST API with 6 endpoints |
| **Dashboard** | `templates/index.html` | Live monitoring dashboard |
| **Preprocessing** | `../code1_train_preprocessing.py` | Train pipeline |
| **Test Prep** | `../code2_test_preprocessing.py` | Test pipeline |
| **Labels** | `../code3_test_labels.py` | Label generator |

---

## Project Structure

```
fraud_dashboard/
├── app.py                    ← Flask API (run this)
├── requirements.txt          ← pip dependencies
├── templates/
│   └── index.html            ← Dashboard UI
└── artifacts/                ← Place your model files here
    ├── scaler.pkl            ← from code1_train_preprocessing.py
    ├── feature_columns.json  ← from code1_train_preprocessing.py
    ├── encoding_maps.json    ← from code1_train_preprocessing.py
    ├── imputation_lookup.json← from code1_train_preprocessing.py
    ├── xgb_model.pkl         ← your trained XGBoost model (optional)
    └── rf_model.pkl          ← your trained RF model (optional)
```

---

## Setup

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Copy artifacts
After running `code1_train_preprocessing.py`, copy the output files:
```bash
mkdir artifacts
cp ../scaler.pkl              artifacts/
cp ../feature_columns.json    artifacts/
cp ../encoding_maps.json      artifacts/
cp ../imputation_lookup.json  artifacts/
```

### Step 3 — (Optional) Copy your trained model
After training your XGBoost/RF model:
```python
import joblib
joblib.dump(your_xgb_model, 'artifacts/xgb_model.pkl')
# OR
joblib.dump(your_rf_model, 'artifacts/rf_model.pkl')
```

### Step 4 — Run the dashboard
```bash
python app.py
```
Open: **http://localhost:5000**

---

## API Reference

### POST /api/predict
Score a single transaction.

**Request:**
```json
{
  "transaction_amount_inr": 12000,
  "merchant_category": "electronics",
  "pos_entry_mode": "CNP",
  "country_code": "US",
  "credit_limit_inr": 75000,
  "avg_txn_amount_30d": 120.0,
  "std_txn_amount_30d": 300.0,
  "velocity_last_1h": 6,
  "velocity_last_24h": 10,
  "distance_from_home_km": 180.0,
  "card_age_days": 22,
  "transaction_hour": 2
}
```

**Response:**
```json
{
  "fraud_probability": 87.3,
  "fraud_score": 0.873,
  "is_fraud_predicted": 1,
  "risk_tier": "CRITICAL",
  "risk_color": "#FF2D55",
  "model_used": "XGBoost (Optuna-tuned)",
  "risk_factors": [
    {"factor": "Amount Spike", "detail": "₹12,000 is 100.0× cardholder average", "severity": "high"},
    {"factor": "High Velocity", "detail": "6 transactions in last 1 hour", "severity": "high"},
    {"factor": "Night Transaction", "detail": "Occurred at 02:00 (off-hours)", "severity": "medium"}
  ]
}
```

### POST /api/predict/batch
Upload test_transactions.csv and get predictions for all rows.

```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -F "file=@test_transactions.csv"
```

### GET /api/stats
Dashboard KPI summary.

### GET /api/recent?filter=fraud&limit=20
Recent transaction feed.

### GET /api/feature-importance
SHAP-derived feature weights.

### POST /api/simulate?type=fraud|legit|random
Generate and score a simulated transaction.

### GET /api/health
System health check.

---

## Dashboard Features

| Section | What It Shows |
|---|---|
| **Dashboard** | KPIs · Risk donut chart · POS fraud rates · Recent flags |
| **Live Feed** | All transactions table · Fraud/all filter · Auto-refresh |
| **Score Transaction** | Manual entry form · Instant scoring · Risk factor breakdown |
| **Batch Upload** | CSV drag-drop · Bulk predictions · Summary stats |
| **Feature Importance** | Bar chart · SHAP values · Feature type table |

---

## Demo Mode
If no `xgb_model.pkl` or `rf_model.pkl` is found in `artifacts/`,
the system runs in **Demo Mode** using a rule-based scorer.
All preprocessing, feature engineering, and API endpoints work identically.
Replace with your trained model when ready.

---

## PS1 Compliance

| PS Requirement | Implementation |
|---|---|
| ML-based fraud scoring | XGBoost/RF via `/api/predict` |
| Recall > 75% | Threshold = 0.35 (PR-curve optimized) |
| Explainability | SHAP-based risk factors per transaction |
| Real-time monitoring | Flask API + auto-refreshing dashboard |
| Analyst interface | Risk tier pills · Factor breakdown · Live feed |
| Batch evaluation | CSV upload → predictions table |

