from pydantic import BaseModel, Field


class PropertyCard(BaseModel):
    id: str
    title: str
    location: str
    price: float
    beds: float
    baths: float
    sqft: float
    image: str
    tag: str | None = None


class PropertyListResponse(BaseModel):
    items: list[PropertyCard]
    total: int


class AgentInfo(BaseModel):
    name: str
    role: str
    image: str


class PropertyDetail(PropertyCard):
    description: str
    images: list[str]
    agent: AgentInfo
    similar: list[PropertyCard] = Field(default_factory=list)
    rawDetails: dict[str, str | float | None] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    area: float = Field(..., gt=0)
    location: str = Field(..., min_length=1)
    bedrooms: float = Field(default=1, ge=0)
    bathrooms: float = Field(default=1, ge=0)
    frontage: int = Field(default=0, ge=0, le=1)
    accessRoad: float = Field(default=0, ge=0)
    floors: float = Field(default=1, ge=1)
    legalStatus: str = "Sale contract"
    furnitureState: str = "Fully furnished"


class PredictionResult(BaseModel):
    estimatedValue: float
    confidence: float
    trend: float
    analysis: str


class InquiryRequest(BaseModel):
    name: str = Field(..., min_length=2)
    email: str = Field(..., min_length=5)
    message: str = Field(..., min_length=5)
    propertyId: str | None = None


class InquiryResponse(BaseModel):
    message: str
