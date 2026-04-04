import argparse
import json
from pathlib import Path

from pymongo import MongoClient, UpdateOne


def main() -> None:
    parser = argparse.ArgumentParser(description="Import properties JSON seed into MongoDB")
    parser.add_argument("--file", default="backend/data/mongodb_properties_seed.json")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db-name", default="real_estate_app")
    parser.add_argument("--drop", action="store_true", help="Drop properties collection before import")
    args = parser.parse_args()

    seed_path = Path(args.file)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    docs = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(docs, list):
        raise ValueError("Seed JSON must be a list of documents.")

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]
    properties = db["properties"]

    if args.drop:
        properties.drop()

    properties.create_index("listing_id", unique=True)
    properties.create_index("address")
    properties.create_index("price")

    ops: list[UpdateOne] = []
    for doc in docs:
        listing_id = doc.get("listing_id")
        if not listing_id:
            continue

        ops.append(
            UpdateOne(
                {"listing_id": listing_id},
                {"$set": doc},
                upsert=True,
            )
        )

    if not ops:
        print("No valid documents to import.")
        return

    result = properties.bulk_write(ops, ordered=False)
    total = properties.count_documents({})

    print(
        f"Import complete. Upserted: {result.upserted_count}, "
        f"Modified: {result.modified_count}, Total: {total}"
    )


if __name__ == "__main__":
    main()
