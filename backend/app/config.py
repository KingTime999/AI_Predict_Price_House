from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_db: str
    model_path: Path
    cors_origins: list[str]


def _build_settings() -> Settings:
    origins_raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]

    model_path_raw = os.getenv(
        "MODEL_PATH",
        str(ARTIFACTS_DIR / "house_price_model.joblib"),
    )

    return Settings(
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_db=os.getenv("MONGODB_DB", "real_estate_app"),
        model_path=Path(model_path_raw),
        cors_origins=origins,
    )


settings = _build_settings()
