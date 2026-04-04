# Vietnam Real Estate Price Prediction Platform

A modern full-stack web application for searching, viewing, and predicting residential property prices in Vietnam. Built with React+Vite frontend, FastAPI backend, MongoDB database, and machine learning.

## Overview

This integrated real estate platform offers:
- **Property Catalog** - Browse 1,200+ residential properties across 10 Vietnamese cities
- **Search & Filter** - Find properties by location, price range, bedrooms
- **AI Price Prediction** - ML-powered property valuation with 95% confidence
- **Market Analysis** - Architectural & market insights for predictions
- **Data Management** - MongoDB integration for property listings and inquiries  

## Architecture

### Technology Stack

- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Uvicorn (port 8000)
- **Database**: MongoDB
- **ML Model**: scikit-learn Linear Regression
- **API Pattern**: REST with JSON (CORS-enabled)


## Quick Start

### Prerequisites
- Node.js v18+ and npm
- Python 3.10+ with venv
- MongoDB (local or remote: `mongodb://localhost:27017`)

### Setup Steps

1. **Install dependencies**
```bash
npm install
npm run api:install
```

2. **Load demo data**
```bash
npm run api:seed:import
```

3. **Configure environment** (.env)
```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=real_estate_app
MODEL_PATH=backend/artifacts/house_price_model.joblib
CORS_ORIGINS=http://localhost:3000
```

4. **Start backend** (Terminal 1)
```bash
npm run api:dev
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

5. **Start frontend** (Terminal 2)
```bash
npm run dev
# Application available at http://localhost:3000
```

## Machine Learning Model

**Algorithm**: Linear Regression (scikit-learn)  
**Dataset**: 1,200 properties across 10 Vietnamese cities  
**Model Accuracy**: R² = 0.8956 (training) / 0.8712 (validation)  
**Prediction Confidence**: 95% average

**Key Features Used**:
- Area (m²)
- Location (District)
- Bedrooms, Bathrooms
- Property condition (Frontage, Floors, Furniture state)
- Legal status

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

**Response**:
```json
{
  "estimatedValue": 7777.78,
  "confidence": 95.0,
  "trend": 4.38,
  "analysis": "Model estimates this property around 7,777,780,000 VND for District 7, HCMC..."
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
└── README.md
```

## Environment Variables

Default values (if .env not set):
- `MONGODB_URI`: mongodb://localhost:27017
- `MONGODB_DB`: real_estate_app
- `MODEL_PATH`: backend/artifacts/house_price_model.joblib
- `CORS_ORIGINS`: http://localhost:3000

## Training Custom Model

```bash
.\.venv\Scripts\python.exe backend/scripts/train_model.py \
  --csv "backend/data/demo_training_dataset.csv" \
  --output "backend/artifacts/house_price_model.joblib"
```

## Support

- API Documentation: http://localhost:8000/docs (when running)
- Backend Setup: See backend/README.md
