"""AML Investigation Agent.

Factory module providing ``create_aml_agent()`` which builds a Google ADK
``LlmAgent`` configured for AML investigations using:
- Read-only SQL tools for transaction database queries
- Internal knowledge base / watchlist search
- External web search via Gemini Grounded Search

The agent produces well-formatted reports with standardized citations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.genai.types import GenerateContentConfig, ThinkingConfig

from aml_agent.config import Configs
from aml_agent.prompts import AGENT_INSTRUCTIONS
from aml_agent.tools.kb_search import KBSearchTool
from aml_agent.tools.sql_database import ReadOnlySqlDatabase
from aml_agent.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# AGENT_INSTRUCTIONS is imported from aml_agent.prompts — edit that file to tune agent behavior.


def create_aml_agent(
    name: str = "AmlInvestigationAgent",
    *,
    configs: Configs | None = None,
    temperature: float | None = None,
    timeout_sec: int | None = None,
    before_model_callback: Callable[..., Any] | None = None,
    after_model_callback: Callable[..., Any] | None = None,
) -> LlmAgent:
    """Create a configured AML investigation agent.

    Parameters
    ----------
    name : str
        Name assigned to the agent.
    configs : Configs | None
        Configuration object. If None, loads from environment.
    temperature : float | None
        Sampling temperature for model generation.
    timeout_sec : int | None
        Timeout for model calls in seconds.

    Returns
    -------
    LlmAgent
        Configured AML investigation agent with SQL, KB, and web search tools.
    """
    if configs is None:
        configs = Configs.from_env()

    # Ensure database exists
    db_path = Path(configs.db.database_path) if configs.db else None
    if db_path and not db_path.exists():
        logger.info("Database not found at %s, building...", db_path)
        from aml_agent.data.build_db import build_database
        build_database(db_path=str(db_path))

    # Initialize tools
    db = ReadOnlySqlDatabase(
        connection_uri=configs.db.build_uri() if configs.db else f"sqlite:///{_PROJECT_ROOT}/aml_agent/data/aml_transactions.db?mode=ro",
        agent_name=name,
    )

    kb = KBSearchTool(
        weaviate_config=configs.weaviate,
        num_results=5,
        snippet_length=500,
    )

    web = WebSearchTool(
        api_key=configs.google_api_key,
        model_name=configs.worker_model,
    )

    return LlmAgent(
        name=name,
        description="Conducts AML investigations using transaction analysis, watchlist screening, and web search.",
        model=configs.planner_model,
        instruction=AGENT_INSTRUCTIONS,
        tools=[
            FunctionTool(db.get_schema_info),
            FunctionTool(db.execute),
            FunctionTool(kb.search_knowledgebase),
            FunctionTool(kb.get_entity_by_id),
            FunctionTool(web.web_search),
        ],
        generate_content_config=GenerateContentConfig(
            temperature=temperature,
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
    )


def _get_root_agent() -> LlmAgent:
    """Lazy factory for ADK discovery."""
    return create_aml_agent()


# ADK discovery: module-level root_agent (deferred)
class _LazyAgent:
    """Proxy that creates the agent on first attribute access."""

    def __init__(self):
        self._agent = None

    def _ensure(self):
        if self._agent is None:
            self._agent = create_aml_agent()
        return self._agent

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


root_agent = _LazyAgent()
