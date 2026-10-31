"""Application settings, loaded from environment variables (and the repo-root .env locally).

In Docker/ECS there is no .env file: the variables are injected into the container, and
the same names work. Never put real secrets in code; AWS secrets come from Secrets Manager.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# agent_host/app/config.py -> repo root (when run from a checkout). In a container, set
# absolute paths (e.g. CHROMA_PATH=/data/chroma) instead of relying on this.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # --- vector store ---
    vector_store: Literal["chroma", "bedrock_kb"] = "chroma"
    chroma_path: Path = Path("./chroma_db")
    chroma_collection: str = "filings"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bedrock_kb_id: str = ""

    # --- AWS ---
    aws_region: str = "ap-south-1"

    # --- MCP servers (base URLs; the client appends /mcp) ---
    sec_edgar_url: str = "http://localhost:8001"
    market_data_url: str = "http://localhost:8002"
    portfolio_url: str = "http://localhost:8003"
    notify_slack_url: str = "http://localhost:8004"

    # --- LLM (Step 14) ---
    # "anthropic" uses the Anthropic API (needs ANTHROPIC_API_KEY in the environment/.env).
    # "bedrock" uses AWS Bedrock via your AWS profile (needs BEDROCK_MODEL_ID).
    llm_provider: Literal["anthropic", "bedrock"] = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    bedrock_model_id: str = ""
    anthropic_api_key: str = ""
    llm_timeout_seconds: float = 90.0

    # --- Agent behaviour ---
    max_iterations: int = 8  # cap on planner->tool loops per request
    demo_user_id: int = 1  # whose portfolio to read until real auth exists
    slack_channel_name: str = "#finance-reports"
    mcp_timeout_seconds: float = 60.0

    @field_validator("chroma_path")
    @classmethod
    def _resolve_chroma_path(cls, value: Path) -> Path:
        # Relative paths resolve against the repo root, so ingestion (run from the root)
        # and the agent host (run from agent_host/) always share one database.
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @model_validator(mode="after")
    def _check_backend(self) -> Settings:
        if self.vector_store == "bedrock_kb" and not self.bedrock_kb_id:
            raise ValueError("BEDROCK_KB_ID must be set when VECTOR_STORE=bedrock_kb")
        if self.llm_provider == "bedrock" and not self.bedrock_model_id:
            raise ValueError("BEDROCK_MODEL_ID must be set when LLM_PROVIDER=bedrock")
        if self.max_iterations < 1:
            raise ValueError("MAX_ITERATIONS must be at least 1")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
