"""Configuration management for the AML agent."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(verbose=True)

# Project root: directory containing this package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite database configuration."""

    database_path: str
    mode: str = "ro"

    def build_uri(self) -> str:
        """Build a SQLAlchemy connection URI."""
        return f"sqlite:///{self.database_path}?mode={self.mode}"


@dataclass(frozen=True)
class WeaviateConfig:
    """Weaviate Cloud configuration."""

    url: str = ""
    api_key: str | None = None
    collection_name: str = "ComprehensiveWatchList"


@dataclass
class Configs:
    """Central configuration for the AML agent."""

    # LLM
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    planner_model: str = "gemini-2.5-flash"
    worker_model: str = "gemini-2.5-flash"

    # Database
    db: DatabaseConfig | None = None

    # Weaviate
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)

    # Data paths
    transactions_csv: str = str(_PROJECT_ROOT / "data" / "transactions.csv")

    @classmethod
    def from_env(cls) -> "Configs":
        """Load configuration from environment variables."""
        db_path = os.getenv(
            "AML_DB_PATH",
            str(_PROJECT_ROOT / "aml_agent" / "data" / "aml_transactions.db"),
        )
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
            planner_model=os.getenv("AML_PLANNER_MODEL", os.getenv("DEFAULT_PLANNER_MODEL", "gemini-2.5-flash")),
            worker_model=os.getenv("AML_WORKER_MODEL", os.getenv("DEFAULT_WORKER_MODEL", "gemini-2.5-flash")),
            db=DatabaseConfig(database_path=db_path),
            weaviate=WeaviateConfig(
                url=os.getenv("WEAVIATE_URL", ""),
                api_key=os.getenv("WEAVIATE_API_KEY"),
                collection_name=os.getenv("WEAVIATE_COLLECTION", "ComprehensiveWatchList"),
            ),
        )
