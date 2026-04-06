from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from joblib import load as joblib_load
import pandas as pd

from .config import settings
from .database import get_collection
from .schemas import (
    AgentInfo,
    InquiryRequest,
    InquiryResponse,
    PredictRequest,
    PredictionResult,
    PropertyCard,
    PropertyDetail,
    PropertyListResponse,
)

app = FastAPI(
    title="Vietnam House Price API",
    version="1.0.0",
    description="FastAPI backend for house search, details, and price prediction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_CACHE: dict[str, Any] = {"artifact": None}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_images(seed: str) -> list[str]:
    return [
        f"https://picsum.photos/seed/{seed}-1/1200/800",
        f"https://picsum.photos/seed/{seed}-2/1200/800",
        f"https://picsum.photos/seed/{seed}-3/1200/800",
        f"https://picsum.photos/seed/{seed}-4/1200/800",
    ]


def _property_tag(price: float) -> str:
    if price >= 7000:
        return "Premium"
    if price >= 4000:
        return "Featured"
    if price >= 2000:
        return "Hot"
    return "New"


def _serialize_card(doc: dict[str, Any]) -> PropertyCard:
    listing_id = str(doc.get("listing_id") or doc.get("_id"))
    address = str(doc.get("address") or "Vietnam")
    price = _safe_float(doc.get("price"), 0.0)
    area = _safe_float(doc.get("area"), 0.0)
    beds = _safe_float(doc.get("bedrooms"), 0.0)
    baths = _safe_float(doc.get("bathrooms"), 0.0)
    image = str(doc.get("image_url") or _build_images(listing_id)[0])

    return PropertyCard(
        id=listing_id,
        title=str(doc.get("title") or f"Property in {address}"),
        location=address,
        price=price,
        beds=beds,
        baths=baths,
        sqft=area,
        image=image,
        tag=str(doc.get("tag") or _property_tag(price)),
    )


def _load_model_artifact(force_reload: bool = False) -> dict[str, Any] | None:
    if _MODEL_CACHE["artifact"] is not None and not force_reload:
        return _MODEL_CACHE["artifact"]

    print(f"DEBUG: Model path: {settings.model_path}")
    print(f"DEBUG: Model path exists: {settings.model_path.exists()}")
    if settings.model_path.parent.exists():
        print(f"DEBUG: Parent directory exists")
        print(f"DEBUG: Files in parent: {list(settings.model_path.parent.iterdir())}")
    
    if not settings.model_path.exists():
        print(f"DEBUG: Model path does not exist!")
        _MODEL_CACHE["artifact"] = None
        return None

    artifact = joblib_load(settings.model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise RuntimeError("Invalid model artifact format.")

    _MODEL_CACHE["artifact"] = artifact
    return artifact


def _prediction_analysis(location: str, estimate: float, confidence: float, is_location_known: bool = True) -> str:
    # Convert from millions to VND for display
    estimate_vnd = estimate * 1_000_000
    formatted_vnd = f"{estimate_vnd:,.0f}"
    
    # Add warning if location is unknown
    location_note = ""
    if not is_location_known:
        location_note = (
            f"\n\n⚠️ **Note**: '{location}' is not in our training dataset. "
            "The model trained on 10 Vietnamese districts only: Bien Hoa, Binh Thanh, Cau Giay, Hai Chau, "
            "Nam Tu Liem, Nha Trang, Quan 1, Quan 7, Thu Dau Mot, and Thu Duc. "
            "This prediction may be less reliable. Consider using a known district for more accurate results."
        )
    
    analysis = (
        f"Model estimates this property around {formatted_vnd} VND ({estimate:,.2f} tr VND) for {location}. "
        f"Confidence is {confidence:.1f}% based on validation error and feature quality. "
        f"This prediction is calculated using linear regression on 1,200 residential properties across 10 cities. "
        f"Consider checking legal status, frontage, and nearby transaction history before final pricing.{location_note}"
    )
    return analysis


def _fallback_prediction(payload: PredictRequest) -> PredictionResult:
    frontage = payload.frontage if payload.frontage else 0
    floors = payload.floors if payload.floors else 2
    estimated = max(
        300.0,
        payload.area * 45 + payload.bedrooms * 280 + payload.bathrooms * 240 + frontage * 55 + floors * 140,
    )
    trend = max(1.0, min(10.5, 3.6 + (payload.area - 60) / 160))
    confidence = 63.0
    analysis = (
        "Model artifact not found yet, using baseline estimate. "
        "Run training script with your Kaggle CSV for production-grade predictions."
    )

    return PredictionResult(
        estimatedValue=round(estimated, 2),
        confidence=confidence,
        trend=round(trend, 2),
        analysis=analysis,
    )


@app.on_event("startup")
def startup_event() -> None:
    print("Starting up - loading model artifact...")
    result = _load_model_artifact()
    print(f"Model load result: {result is not None}")
    if result is not None:
        print(f"Model metrics: {result.get('metrics', {})}")
    else:
        print("Model artifact is None!")


@app.get("/api/health")
def health() -> dict[str, Any]:
    properties_col = get_collection("properties")
    artifact = _load_model_artifact()

    return {
        "status": "ok",
        "modelLoaded": artifact is not None,
        "propertyCount": properties_col.count_documents({}),
    }


@app.get("/api/properties", response_model=PropertyListResponse)
def list_properties(
    location: str | None = None,
    minPrice: float | None = None,
    maxPrice: float | None = None,
    minBeds: float | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
) -> PropertyListResponse:
    properties_col = get_collection("properties")
    query: dict[str, Any] = {}

    if location:
        query["address"] = {"$regex": location, "$options": "i"}

    if minPrice is not None or maxPrice is not None:
        price_query: dict[str, float] = {}
        if minPrice is not None:
            price_query["$gte"] = minPrice
        if maxPrice is not None:
            price_query["$lte"] = maxPrice
        query["price"] = price_query

    if minBeds is not None:
        query["bedrooms"] = {"$gte": minBeds}

    cursor = (
        properties_col.find(query)
        .sort("price", -1)
        .skip(skip)
        .limit(limit)
    )

    items = [_serialize_card(doc) for doc in cursor]
    total = properties_col.count_documents(query)

    return PropertyListResponse(items=items, total=total)


@app.get("/api/properties/featured", response_model=list[PropertyCard])
def featured_properties(limit: int = Query(default=3, ge=1, le=12)) -> list[PropertyCard]:
    properties_col = get_collection("properties")
    cursor = properties_col.find({}).sort("price", -1).limit(limit)
    return [_serialize_card(doc) for doc in cursor]


@app.get("/api/properties/{property_id}", response_model=PropertyDetail)
def property_details(property_id: str) -> PropertyDetail:
    properties_col = get_collection("properties")

    doc = properties_col.find_one({"listing_id": property_id})
    if doc is None and ObjectId.is_valid(property_id):
        doc = properties_col.find_one({"_id": ObjectId(property_id)})

    if doc is None:
        raise HTTPException(status_code=404, detail="Property not found")

    card = _serialize_card(doc)
    listing_id = card.id
    images = list(doc.get("images") or _build_images(listing_id))

    similar_cursor = properties_col.find(
        {
            "listing_id": {"$ne": listing_id},
            "address": doc.get("address"),
        }
    ).limit(3)
    similar = [_serialize_card(item) for item in similar_cursor]

    if len(similar) < 3:
        fallback_cursor = properties_col.find({"listing_id": {"$ne": listing_id}}).sort("price", -1).limit(3)
        similar = [_serialize_card(item) for item in fallback_cursor]

    details = {
        "Address": str(doc.get("address") or "N/A"),
        "Frontage": doc.get("frontage"),
        "Access Road": doc.get("access_road"),
        "Floors": doc.get("floors"),
        "House direction": str(doc.get("house_direction") or "N/A"),
        "Balcony direction": str(doc.get("balcony_direction") or "N/A"),
        "Legal status": str(doc.get("legal_status") or "N/A"),
        "Furniture state": str(doc.get("furniture_state") or "N/A"),
    }

    return PropertyDetail(
        **card.model_dump(),
        description=str(
            doc.get("description")
            or (
                "Well-positioned residential property in Vietnam with strong market fundamentals. "
                "Contact our advisory team for transaction history and legal due diligence."
            )
        ),
        images=images,
        agent=AgentInfo(
            name="Nguyen Minh Anh",
            role="Property Consultant",
            image="https://picsum.photos/seed/agent-vn/200/200",
        ),
        similar=similar,
        rawDetails=details,
    )


@app.post("/api/inquiries", response_model=InquiryResponse, status_code=201)
def create_inquiry(payload: InquiryRequest) -> InquiryResponse:
    inquiries_col = get_collection("inquiries")
    record = payload.model_dump()
    record["createdAt"] = datetime.now(timezone.utc)

    inquiries_col.insert_one(record)
    return InquiryResponse(message="Inquiry submitted successfully")


@app.post("/api/predict", response_model=PredictionResult)
def predict_price(payload: PredictRequest) -> PredictionResult:
    artifact = _load_model_artifact()
    if artifact is None:
        return _fallback_prediction(payload)

    pipeline = artifact["pipeline"]
    metrics = artifact.get("metrics", {})
    
    # Check if location is in training data
    KNOWN_LOCATIONS = [
        "Bien Hoa, Dong Nai",
        "Binh Thanh, TP.HCM",
        "Cau Giay, Ha Noi",
        "Hai Chau, Da Nang",
        "Nam Tu Liem, Ha Noi",
        "Nha Trang, Khanh Hoa",
        "Quan 1, TP.HCM",
        "Quan 7, TP.HCM",
        "Thu Dau Mot, Binh Duong",
        "Thu Duc, TP.HCM",
    ]
    
    is_location_known = any(loc.lower() in payload.location.lower() for loc in KNOWN_LOCATIONS)

    frame = pd.DataFrame(
        [
            {
                "Frontage": payload.frontage,
                "Access Road": payload.accessRoad,
                "Area": payload.area,
                "Bedrooms": payload.bedrooms,
                "Bathrooms": payload.bathrooms,
                "Floors": payload.floors,
                "Address": payload.location,
                "Legal status": payload.legalStatus,
                "Furniture state": payload.furnitureState,
            }
        ]
    )

    estimated = float(pipeline.predict(frame)[0])
    estimated = abs(estimated)

    rmse = _safe_float(metrics.get("rmse"), 0.0)
    confidence = 95.0
    if rmse > 0:
        confidence = max(55.0, min(98.0, 100 - (rmse / max(estimated, 1.0)) * 100))
    
    # Reduce confidence if location is unknown
    if not is_location_known:
        confidence = max(55.0, confidence - 25.0)  # Reduce by 25 percentage points

    trend = max(0.8, min(12.0, 3.1 + (payload.area - 60) / 180 + (payload.bedrooms - 2) * 0.35))

    return PredictionResult(
        estimatedValue=round(estimated, 2),
        confidence=round(confidence, 2),
        trend=round(trend, 2),
        analysis=_prediction_analysis(payload.location, estimated, confidence, is_location_known),
    )
