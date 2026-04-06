# Vietnam Real Estate Price Prediction Platform

A modern full-stack web application for searching, viewing, and predicting residential property prices in Vietnam. Built with React+Vite frontend, FastAPI backend, MongoDB database, and machine learning.

## 👥 Team Members

- **Ngo Khai Luan** - 2374802010295
- **Bui Tien Dat** - 2374802010094
- **Nguyen Hoang Tien** - 2374802010494
- **Pham Le Duy Bao** - 2374802010043
- **Le Tan Nguyen** - 2374802010354


## Overview

This integrated real estate platform offers:
- **Property Catalog** - Browse 54,202+ residential properties across 235 Vietnamese districts
- **Search & Filter** - Find properties by location, price range, bedrooms
- **AI Price Prediction** - ML-powered property valuation using RandomForest (62.88% accuracy)
- **Market Analysis** - Price patterns across all Vietnamese districts
- **CSV Data Management** - Efficient property data handling  

## Architecture

### Technology Stack

- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Uvicorn (port 8000)
- **Data**: CSV-based (catalyst_dataset.csv)
- **ML Model**: scikit-learn RandomForest (100 trees)
- **API Pattern**: REST with JSON (CORS-enabled)


## Quick Start

### Prerequisites
- Node.js v18+ and npm
- Python 3.10+ with venv
- FastAPI & scikit-learn (installed via pip)

### Setup Steps

1. **Install dependencies**
```bash
npm install
npm run api:install
```

2. **Start backend** (Terminal 1)
```bash
npm run api:dev
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

3. **Start frontend** (Terminal 2)
```bash
npm run dev
# Application available at http://localhost:3000
```

## Machine Learning Model

**Algorithm**: Random Forest (100 decision trees)  
**Dataset**: 54,202 properties → 49,101 cleaned (removing outliers)  
**Model Performance**: R² = 62.88% (test) / 80.02% (training)  
**Error Margin**: MAE ±4,826 tr VND per prediction  
**Total Features**: 240 (5 base + 235 one-hot encoded districts)

**Key Features Used** (by importance):
- Area (m²) - 40.1%
- Floors - 19.2%
- Bedrooms - 7.2%
- Bathrooms - 5.3%
- District/Location - 5-15%
- Frontage - 3.9%

**Training Data**:
- Training set: 39,280 properties (80%)
- Test set: 9,821 properties (20%)
- Coverage: 235 unique districts across Vietnam

**Outlier Removal**:
- Removed prices < 500 tr VND (too cheap)
- Removed prices > 196,000 tr VND (too expensive)
- Removed properties with missing area/invalid data

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service health check |
| `/api/properties` | GET | List properties (with filters) |
| `/api/properties/{id}` | GET | Property details |
| `/api/predict` | POST | Predict property price |
| `/api/inquiries` | POST | Submit inquiry |

### Example: Predict Property Price

**Request**:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 100,
    "location": "District 7, HCMC",
    "bedrooms": 3,
    "bathrooms": 2,
    "frontage": 1,
    "accessRoad": 10,
    "floors": 2,
    "legalStatus": "Red book",
    "furnitureState": "Fully furnished"
  }'
```

**Response** (example for Quận 1):
```json
{
  "estimatedValue": 48194,
  "confidence": 0.6288,
  "analysis": "Model estimates this property around 48,194,000,000 VND (48 tỷ VND) for Quận 1, HCMC. Model accuracy: 62.88%. Error range: ±4,826 tr VND."
}
```

## Project Structure

```
.
├── src/                     # React frontend
│   ├── pages/              # Page components
│   ├── components/         # Reusable UI components
│   └── services/           # API & business logic
├── backend/                # FastAPI backend
│   ├── app/
│   │   ├── main.py        # API endpoints
│   │   ├── schemas.py     # Data models
│   │   ├── database.py    # MongoDB helpers
│   │   └── config.py      # Configuration
│   ├── scripts/           # Training & import scripts
│   ├── artifacts/         # Trained ML model
│   └── data/              # Datasets
├── package.json
├── tsconfig.json
├── vite.config.ts
├── README.md
├── CALCULATION_METHOD.md    # Detailed price calculation formula
└── sample_predictions_30_districts.csv  # Sample data (30 predictions)
```

## Price Calculation Method

### How Does the Model Predict Price?

The system uses a **RandomForest ensemble** of 100 decision trees to predict property prices.

#### Step 1: Data Processing
```python
Input JSON from form:
{
  "area": 100,
  "location": "Quan 1, HCMC",
  "bedrooms": 3,
  "bathrooms": 2,
  "floors": 2,
  "frontage": 1
}
```

#### Step 2: District Extraction & Normalization
```
"Quan 1, HCMC" → Extract → "Quan 1"
              → Normalize → "Quận 1" (Vietnamese standardization)
```

#### Step 3: Feature Vector Creation (240 features)
```
Base features (5):
  [100, 2, 3, 2, 1]
   ↑   ↑  ↑  ↑  ↑
  Area Floors Bedrooms Bathrooms Frontage

District encoding (235 features):
  + One-hot encoding: [0, 0, ..., 1 (Quận 1), ..., 0]
                           ↑
                    Only ONE district = 1

Final vector: [100, 2, 3, 2, 1, 0, ..., 1, 0, ...]
              └─────┬─────┘  └─────┬──────┘
               Base (5)     Districts (235)
```

#### Step 4: RandomForest Voting
```
100 Decision Trees each predict independently:
  Tree 1:  48,000 tr VND
  Tree 2:  48,300 tr VND
  Tree 3:  48,100 tr VND
  ...
  Tree 100: 48,200 tr VND

Final prediction = Average of all 100 trees = 48,194 tr VND
```

#### Step 5: Confidence & Error Bounds
```
Confidence: 62.88% (R² score)
  → Model explains 62.88% of price variation
  → 37.12% due to other factors (location details, market, etc.)

Error margin: ±4,826 tr VND (MAE)
  → 68% of predictions fall within [±4,826]
  → 95% fall within [±9,652]
```

### Why Different Districts Have Different Prices?

**Example: Quận 1 (Central HCMC) vs Quận 7 (Suburban)**

For identical properties (100m², 3 bed, 2 bath, 2 floors):

| Aspect | Quận 1 | Quận 7 | Difference |
|--------|--------|--------|------------|
| Feature Vector | [100, 2, 3, 2, 1, ..., **1**, ...] | [100, 2, 3, 2, 1, ..., **0**, ...] | District flag |
| Tree Decision | "If Quận 1 → high price" | "If Quận 7 → low price" | Learned pattern |
| Prediction | 48,194 tr VND | 9,663 tr VND | 5.0x difference |

**District Impact**: Contributes 5-15% to importance but creates massive multipliers:
- Central districts (Q1, Q3, Q5): 3-5x price multiplier
- Suburban (Q7, Q12): 1x baseline
- Rural: 0.9-1x baseline

### Sample Predictions (100m², 3 bed, 2 bath, 2 floors)

| Location | Price | Region | Notes |
|----------|-------|--------|-------|
| Quận 1, HCMC | 48,194 tr | South | Highest - Premium |
| Quận 3, HCMC | 38,368 tr | South | Near-central |
| Quận 5, HCMC | 31,283 tr | South | Central |
| Quận 10, HCMC | 18,243 tr | South | Moderately central |
| Quận 7, HCMC | 9,663 tr | South | Suburban baseline |
| Đà Nẵng | 9,457 tr | Central | Tier-2 city |
| Nha Trang | 9,100 tr | Central | Lowest - Tier-3 city |

**Statistics**: Highest (48,194) / Lowest (9,100) = **5.3x difference**

See [sample_predictions_30_districts.csv](sample_predictions_30_districts.csv) for 30-district comparison.

### Feature Importance Breakdown

```
Area..................40.1%  ████████████████████
Floors................19.2%  █████████
Bedrooms..............7.2%   ███
Bathrooms.............5.3%   ██
District.............5-15%   ██-██████████
Frontage..............3.9%   ██
```

**Practical Examples**:
- +10m² area → +4,000-5,000 tr VND
- +1 floor → +1,000-2,000 tr VND
- Add frontage → +200-300 tr VND
- Q7 to Q1 change → +38,531 tr VND (most impactful!)

### Model Accuracy & Confidence Intervals

**R² = 62.88%**: Explains ~63% of price variation  
**Error Margin = ±4,826 tr VND** (68% confidence)

**Example for 48,194 tr VND prediction**:
- 68% confidence: 43,368 - 53,020 tr VND (±4,826)
- 95% confidence: 38,542 - 57,846 tr VND (±9,652)

See [CALCULATION_METHOD.md](CALCULATION_METHOD.md) for detailed algorithm documentation.

## Environment Variables

Default values (if .env not set):
- `MODEL_PATH`: backend/artifacts/house_price_model.joblib
- `CORS_ORIGINS`: http://localhost:3000
- `PORT`: 8000

## Training Custom Model

Retrain RandomForest model with all 235 districts:

```bash
.\.venv\Scripts\python.exe backend/scripts/retrain_model.py \
  --data "backend/data/catalyst_dataset.csv" \
  --output "backend/artifacts/house_price_model.joblib" \
  --estimators 100 \
  --max-depth 25
```

## Documentation & Support

- **📊 Detailed Calculation Formula**: [CALCULATION_METHOD.md](CALCULATION_METHOD.md) - Full explanation of how RandomForest calculates predictions
- **📈 Sample Predictions**: [sample_predictions_30_districts.csv](sample_predictions_30_districts.csv) - 30-district sample with prices across Vietnam
- **🔗 API Documentation**: http://localhost:8000/docs (when running)
- **⚙️ Backend Setup**: See backend/README.md
