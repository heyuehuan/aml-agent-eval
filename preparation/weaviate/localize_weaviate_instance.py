"""Download a Weaviate collection to a local JSON file.

Exports every object in the collection including:
  - uuid       : Weaviate-assigned UUID
  - metadata   : creation/update timestamps
  - properties : all stored properties (source, document_id, title, text, …)
  - vectors    : named vector(s) stored on the object

Output file (JSONL by default, one object per line):
    data/localized_weaviate/<COLLECTION_NAME>.jsonl

Required environment variables (set in the root-level .env file):
    WEAVIATE_URL        - REST endpoint URL from the Weaviate Cloud console
    WEAVIATE_API_KEY    - Admin API key from the Weaviate Cloud console

Optional environment variables:
    COLLECTION_NAME     - Collection to download (default: ComprehensiveWatchList)
    LOCAL_OUTPUT_PATH   - Override the output file path
    FETCH_BATCH_SIZE    - Objects per page (default: 500)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# ---------------------------------------------------------------------------
# Config
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
FETCH_BATCH_SIZE: int = int(os.getenv("FETCH_BATCH_SIZE", "500"))

_DEFAULT_OUTPUT = REPO_ROOT / "data" / "localized_weaviate" / f"{COLLECTION_NAME}.jsonl"
LOCAL_OUTPUT_PATH = Path(os.getenv("LOCAL_OUTPUT_PATH", str(_DEFAULT_OUTPUT)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(value: object) -> object:
    """Make a value JSON-serialisable (handles datetime, UUID, etc.)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__str__"):
        return str(value)
    return value


def _export_collection(client: weaviate.WeaviateClient, name: str, output: Path) -> int:
    """Stream all objects from *name* into *output* (JSONL). Returns total count."""
    if not client.collections.exists(name):
        sys.exit(f"ERROR: Collection '{name}' does not exist in this Weaviate instance.")

    collection = client.collections.get(name)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    after_uuid = None  # cursor for pagination

    with output.open("w", encoding="utf-8") as fh:
        while True:
            response = collection.query.fetch_objects(
                limit=FETCH_BATCH_SIZE,
                after=after_uuid,
                include_vector=True,
                return_metadata=MetadataQuery(
                    creation_time=True,
                    last_update_time=True,
                ),
            )

            if not response.objects:
                break

            for obj in response.objects:
                record = {
                    "uuid": str(obj.uuid),
                    "metadata": {
                        "creation_time": _serialize(
                            obj.metadata.creation_time if obj.metadata else None
                        ),
                        "last_update_time": _serialize(
                            obj.metadata.last_update_time if obj.metadata else None
                        ),
                    },
                    "properties": {
                        k: _serialize(v) for k, v in obj.properties.items()
                    },
                    "vectors": {
                        vec_name: list(vec)
                        for vec_name, vec in (obj.vector or {}).items()
                    },
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

            after_uuid = response.objects[-1].uuid

            if total % 1000 == 0:
                print(f"  … exported {total} objects so far")

    return total


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

        print(f"Exporting collection '{COLLECTION_NAME}' → {LOCAL_OUTPUT_PATH}")
        total = _export_collection(client, COLLECTION_NAME, LOCAL_OUTPUT_PATH)
        print(f"Done. Exported {total} objects to {LOCAL_OUTPUT_PATH}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
