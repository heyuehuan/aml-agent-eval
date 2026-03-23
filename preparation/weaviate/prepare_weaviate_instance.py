"""Prepare a Weaviate Cloud instance for the AML agent evaluation.

Reads credentials from a .env file at the repository root and:
1. Connects to the Weaviate Cloud instance.
2. Creates the collection (if it does not already exist).
3. Batch-imports all entities from internal_kb_watchlist.json.

Document field mapping (JSON → Weaviate property):
    source                      → source          (watchlist/source name)
    name                        → title           (document name)
    description                 → text            (document content, vectorised)
    source + "|" + entity_id   → document_id     (stable unique identifier)

Required environment variables (set in the root-level .env file):
    WEAVIATE_URL        - REST endpoint URL from the Weaviate Cloud console
    WEAVIATE_API_KEY    - Admin API key from the Weaviate Cloud console

Optional environment variables:
    COLLECTION_NAME        - Weaviate collection to create/populate (default: ComprehensiveWatchList)
    WATCHLIST_JSON_PATH    - Absolute or repo-relative path to internal_kb_watchlist.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth

# ---------------------------------------------------------------------------
# Locate and load .env from the repository root (two levels above this file)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    sys.exit(
        "ERROR: WEAVIATE_URL and WEAVIATE_API_KEY must be set in the .env file "
        f"at {REPO_ROOT / '.env'}"
    )

COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ComprehensiveWatchList")

# Default JSON path: data directory at repo root
_DEFAULT_JSON = REPO_ROOT / "data" / "internal_kb_watchlist.json"
WATCHLIST_JSON_PATH = Path(os.getenv("WATCHLIST_JSON_PATH", str(_DEFAULT_JSON)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_entities(json_path: Path) -> list[dict]:
    """Load and normalise entity records from the JSON watchlist file."""
    if not json_path.exists():
        sys.exit(f"ERROR: Watchlist JSON not found: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)

    normalised = []
    for obj in records:
        normalised.append(
            {
                "source": obj["source"],
                "document_id": f"{obj['source']}|{obj['entity_id']}",
                "title": obj["name"],
                "text": obj["description"],
            }
        )
    return normalised


def _create_collection(client: weaviate.WeaviateClient, name: str) -> None:
    """Create the Weaviate collection if it does not already exist."""
    if client.collections.exists(name):
        print(f"Collection '{name}' already exists – skipping creation.")
        return

    client.collections.create(
        name,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
        ],
        vector_config=[
            Configure.Vectors.text2vec_weaviate(
                name="text_vector",
                source_properties=["text"],
                model="Snowflake/snowflake-arctic-embed-l-v2.0",
            )
        ],
    )
    print(f"Collection '{name}' created.")


def _batch_import(client: weaviate.WeaviateClient, name: str, entities: list[dict]) -> None:
    """Batch-import entities into the collection."""
    collection = client.collections.get(name)

    with collection.batch.fixed_size(batch_size=200) as batch:
        for entity in entities:
            batch.add_object(properties=entity)
            if batch.number_errors > 10:
                print("Batch import stopped: too many errors.")
                break

    failed = collection.batch.failed_objects
    if failed:
        print(f"Failed imports: {len(failed)}")
        print(f"First failure: {failed[0]}")
    else:
        print(f"Successfully imported {len(entities)} entities into '{name}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to Weaviate: {WEAVIATE_URL}")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

    try:
        if not client.is_ready():
            sys.exit("ERROR: Weaviate cluster is not ready.")
        print("Weaviate cluster is ready.")

        _create_collection(client, COLLECTION_NAME)

        print(f"Loading entities from: {WATCHLIST_JSON_PATH}")
        entities = _load_entities(WATCHLIST_JSON_PATH)
        print(f"Loaded {len(entities)} entities.")

        _batch_import(client, COLLECTION_NAME, entities)
    finally:
        client.close()


if __name__ == "__main__":
    main()
