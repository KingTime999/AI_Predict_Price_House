from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from .config import settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri)
    return _client


def get_database() -> Database:
    return get_client()[settings.mongodb_db]


def get_collection(name: str) -> Collection:
    return get_database()[name]
