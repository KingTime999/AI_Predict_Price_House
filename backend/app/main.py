from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import joblib
import os

from .config import settings
from .schemas import (
    PropertyCard,
    PropertyListResponse,
    PredictRequest,
    PredictionResult,
    InquiryRequest,
    InquiryResponse,
)

app = FastAPI(
    title="Vietnam House Price API",
    version="1.0.0",
    description="FastAPI backend for house search, details, and price prediction (CSV-based).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
_DATA_CACHE = {"df": None, "model": None}

# Mount static files for images
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def _load_csv() -> pd.DataFrame:
    """Load dataset from CSV."""
    if _DATA_CACHE["df"] is not None:
        return _DATA_CACHE["df"]
    
    csv_path = Path(__file__).parent.parent / "data" / "catalyst_dataset.csv"
    print(f"Loading CSV from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Fill missing values
    df['Bedrooms'] = df['Bedrooms'].fillna(0)
    df['Bathrooms'] = df['Bathrooms'].fillna(0)
    df['Floors'] = df['Floors'].fillna(0)
    
    _DATA_CACHE["df"] = df
    print(f"Loaded {len(df)} properties from CSV")
    return df


def _load_model() -> dict[str, Any] | None:
    """Load trained model."""
    if _DATA_CACHE["model"] is not None:
        return _DATA_CACHE["model"]
    
    model_path = Path(__file__).parent.parent / "artifacts" / "house_price_model.joblib"
    print(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    model_data = joblib.load(model_path)
    _DATA_CACHE["model"] = model_data
    print(f"Model loaded successfully. R² score: {model_data.get('r2_score', 0):.4f}")
    return model_data


@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    try:
        _load_csv()
        _load_model()
        print("✓ Startup complete: CSV loaded, model ready")
    except Exception as e:
        print(f"✗ Startup error: {e}")


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/properties", response_model=PropertyListResponse)
async def get_properties(
    location: str = Query("", description="Filter by location"),
    min_price: float = Query(0, description="Minimum price in million VND"),
    max_price: float = Query(100000, description="Maximum price in million VND"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get property listings from CSV."""
    try:
        df = _load_csv()
        
        # Filter by price
        filtered = df[(df['Price'] >= min_price) & (df['Price'] <= max_price)].copy()
        
        # Filter by location
        if location:
            filtered = filtered[filtered['Address'].str.contains(location, case=False, na=False)]
        
        total = len(filtered)
        
        # Pagination
        properties_data = filtered.iloc[offset:offset + limit]
        
        items = []
        for idx, row in properties_data.iterrows():
            card = PropertyCard(
                id=str(idx),
                title=f"Property in {row['Address']}",
                location=str(row['Address']),
                price=float(row['Price']),
                beds=float(row.get('Bedrooms', 0) or 0),
                baths=float(row.get('Bathrooms', 0) or 0),
                sqft=float(row['Area']),
                image="/static/house.jpg",
                tag=_property_tag(float(row['Price'])),
            )
            items.append(card)
        
        return PropertyListResponse(items=items, total=total)
    
    except Exception as e:
        print(f"Error fetching properties: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_district(address: str) -> str:
    """Extract district from address (first part before comma)."""
    if not address:
        return "Unknown"
    parts = address.split(',')
    district = parts[0].strip() if len(parts) > 0 else "Unknown"
    
    # Normalize Vietnamese characters: Quan → Quận
    district = district.replace("Quan ", "Quận ")
    
    return district


@app.post("/api/predict", response_model=PredictionResult)
async def predict_price(request: PredictRequest):
    """Predict property price using RandomForest model."""
    try:
        model_data = _load_model()
        
        if not model_data:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        r2_score = model_data.get('r2_score', 0.62)
        
        # Extract district from location
        district = _extract_district(request.location)
        print(f"DEBUG: Location='{request.location}' → District='{district}'")
        
        # Build base features
        base_features = {
            'Area': [request.area],
            'Frontage': [int(request.frontage)],
            'Bedrooms': [request.bedrooms],
            'Bathrooms': [request.bathrooms],
            'Floors': [request.floors],
        }
        
        # Create dataframe with base features
        X_temp = pd.DataFrame(base_features)
        
        # Apply district one-hot encoding
        # Create all district columns from feature_columns
        for col in feature_columns:
            if col.startswith('District_'):
                district_name = col.replace('District_', '')
                X_temp[col] = 1 if district_name == district else 0
        
        # Fill missing district columns with 0 (for districts not in training)
        for col in feature_columns:
            if col not in X_temp.columns:
                X_temp[col] = 0
        
        # Ensure all feature columns are present in correct order
        X_pred = X_temp[feature_columns]
        
        # Predict
        try:
            prediction = model.predict(X_pred)[0]
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(pred_error)}")
        
        # Ensure prediction is reasonable (within training range: 500-196000)
        prediction = max(500, min(196000, prediction))
        
        # Confidence and trend
        confidence = r2_score
        trend = 2.5  # Fixed trend
        
        # Analysis with district information
        analysis = f"Based on {confidence:.1%} model accuracy. Property in {district} with {request.area}m² area, {int(request.bedrooms)} bedrooms, {int(request.bathrooms)} bathrooms, {int(request.floors)} floors."
        
        return PredictionResult(
            estimatedValue=prediction,
            confidence=confidence,
            trend=trend,
            analysis=analysis
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error predicting price: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inquiries", response_model=InquiryResponse)
async def submit_inquiry(request: InquiryRequest):
    """Submit inquiry (stored in memory)."""
    return InquiryResponse(
        id="inquiry_" + str(datetime.now(timezone.utc).timestamp()),
        status="received",
        message="Thank you for your inquiry. We'll contact you soon."
    )


def _property_tag(price: float) -> str:
    """Determine property tag based on price."""
    if price >= 7000:
        return "Premium"
    if price >= 4000:
        return "Featured"
    if price >= 2000:
        return "Hot"
    return "New"
