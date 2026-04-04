import argparse
from datetime import datetime, timezone
import re

import pandas as pd
from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo import MongoClient


def parse_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()

    normalized = (
        text.replace("m2", "")
        .replace("m^2", "")
        .replace("m", "")
        .replace(" ", "")
    )
    normalized = normalized.replace(",", ".")

    matches = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    if not matches:
        return None

    try:
        first = float(matches[0])

        has_ty = "tỷ" in lowered or "ty" in lowered or "billion" in lowered
        has_trieu = "triệu" in lowered or "trieu" in lowered or "million" in lowered

        if has_ty and has_trieu and len(matches) >= 2:
            second = float(matches[1])
            return first * 1000 + second

        if has_ty:
            return first * 1000

        return first
    except ValueError:
        return None


def row_to_document(index: int, row: pd.Series) -> dict[str, object]:
    address = str(row.get("Address") or "Unknown area")
    price = parse_number(row.get("Price")) or 0.0
    area = parse_number(row.get("Area")) or 0.0
    bedrooms = parse_number(row.get("Bedrooms")) or 1.0
    bathrooms = parse_number(row.get("Bathrooms")) or 1.0

    listing_id = f"vn-{index + 1:05d}"

    return {
        "listing_id": listing_id,
        "title": f"Property in {address}",
        "address": address,
        "price": price,
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": parse_number(row.get("Floors")),
        "frontage": parse_number(row.get("Frontage")),
        "access_road": parse_number(row.get("Access Road")),
        "house_direction": str(row.get("House direction") or "N/A"),
        "balcony_direction": str(row.get("Balcony direction") or "N/A"),
        "legal_status": str(row.get("Legal status") or "Sale contract"),
        "furniture_state": str(row.get("Furniture state") or "N/A"),
        "description": (
            f"Located in {address}, this listing offers approximately {area:.1f} m2, "
            f"with {bedrooms:.0f} bedrooms and {bathrooms:.0f} bathrooms. "
            "Data imported from Vietnam Housing Dataset 2024."
        ),
        "image_url": f"https://picsum.photos/seed/{listing_id}/1200/800",
        "images": [
            f"https://picsum.photos/seed/{listing_id}-1/1200/800",
            f"https://picsum.photos/seed/{listing_id}-2/1200/800",
            f"https://picsum.photos/seed/{listing_id}-3/1200/800",
            f"https://picsum.photos/seed/{listing_id}-4/1200/800",
        ],
        "createdAt": datetime.now(timezone.utc),
    }


def import_dataset(db: Database, csv_path: str, drop_existing: bool) -> tuple[int, int]:
    properties: Collection = db["properties"]
    frame = pd.read_csv(csv_path)

    if "Address" not in frame.columns or "Price" not in frame.columns:
        raise ValueError("CSV must include at least Address and Price columns.")

    if drop_existing:
        properties.drop()

    properties.create_index("listing_id", unique=True)
    properties.create_index("address")
    properties.create_index("price")

    ops: list[UpdateOne] = []
    for idx, row in frame.iterrows():
        doc = row_to_document(idx, row)
        ops.append(
            UpdateOne(
                {"listing_id": doc["listing_id"]},
                {"$set": doc},
                upsert=True,
            )
        )

    if not ops:
        return 0, 0

    result = properties.bulk_write(ops, ordered=False)
    upserted = result.upserted_count or 0
    modified = result.modified_count or 0
    return upserted, modified


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Vietnam housing CSV into MongoDB")
    parser.add_argument("--csv", required=True, help="Path to vietnam_housing_dataset.csv")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db-name", default="real_estate_app")
    parser.add_argument("--drop", action="store_true", help="Drop properties collection before import")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]

    upserted, modified = import_dataset(db, args.csv, args.drop)
    total = db["properties"].count_documents({})

    print(f"Import finished. Upserted: {upserted}, Modified: {modified}, Total records: {total}")


if __name__ == "__main__":
    main()
