import argparse
from datetime import datetime, timezone
import random

from pymongo import MongoClient, UpdateOne


ADDRESSES = [
    "Quan 1, TP.HCM",
    "Quan 7, TP.HCM",
    "Thu Duc, TP.HCM",
    "Binh Thanh, TP.HCM",
    "Cau Giay, Ha Noi",
    "Nam Tu Liem, Ha Noi",
    "Hai Chau, Da Nang",
    "Nha Trang, Khanh Hoa",
    "Thu Dau Mot, Binh Duong",
    "Bien Hoa, Dong Nai",
]

HOUSE_DIRECTIONS = ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]
BALCONY_DIRECTIONS = ["N/A", "North", "South", "East", "West"]
LEGAL_STATUS = ["Sale contract", "Pink book", "Red book"]
FURNITURE_STATES = ["N/A", "Fully furnished", "Basic furnished", "Unfurnished"]


def build_record(index: int) -> dict[str, object]:
    rnd = random.Random(2026 + index)
    address = rnd.choice(ADDRESSES)

    area = round(rnd.uniform(35, 240), 1)
    bedrooms = rnd.randint(1, 6)
    bathrooms = rnd.randint(1, 5)
    floors = rnd.randint(1, 5)
    frontage = round(rnd.uniform(3.0, 9.5), 1)
    access_road = round(rnd.uniform(3.0, 20.0), 1)

    # Price unit: million VND
    base_price_per_m2 = rnd.uniform(28, 120)
    premium = bedrooms * 120 + bathrooms * 90 + floors * 70
    location_boost = (ADDRESSES.index(address) + 1) * 50
    price = round(area * base_price_per_m2 + premium + location_boost, 2)

    listing_id = f"seed-{index + 1:05d}"

    return {
        "listing_id": listing_id,
        "title": f"Modern Property {index + 1}",
        "address": address,
        "price": price,
        "area": area,
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "floors": float(floors),
        "frontage": frontage,
        "access_road": access_road,
        "house_direction": rnd.choice(HOUSE_DIRECTIONS),
        "balcony_direction": rnd.choice(BALCONY_DIRECTIONS),
        "legal_status": rnd.choice(LEGAL_STATUS),
        "furniture_state": rnd.choice(FURNITURE_STATES),
        "description": (
            f"Sample listing in {address} with {area} m2, {bedrooms} bedrooms, "
            f"{bathrooms} bathrooms and {floors} floors."
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo property data into MongoDB")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db-name", default="real_estate_app")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--drop", action="store_true", help="Drop properties collection before seeding")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]
    properties = db["properties"]

    if args.drop:
        properties.drop()

    properties.create_index("listing_id", unique=True)
    properties.create_index("address")
    properties.create_index("price")

    ops = []
    for i in range(args.count):
        record = build_record(i)
        ops.append(
            UpdateOne(
                {"listing_id": record["listing_id"]},
                {"$set": record},
                upsert=True,
            )
        )

    result = properties.bulk_write(ops, ordered=False)
    total = properties.count_documents({})

    print(
        f"Seed complete. Upserted: {result.upserted_count}, "
        f"Modified: {result.modified_count}, Total: {total}"
    )


if __name__ == "__main__":
    main()
