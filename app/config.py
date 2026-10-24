# file: app/config.py

import os
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_ENV: Literal["development", "production"] = "development"
    USE_MOCK_MODELS: bool = False
    
    OPENCODE_API_KEY: str = Field(..., description="API key for OpenCode")
    OPENCODE_BASE_URL: str = "https://opencode.ai/inference/openai/v1"
    
    # Switched to OpenAI models to bypass DeepSeek API timeouts
    OPENCODE_MODEL_FAST: str = "gpt-4o-mini"
    OPENCODE_MODEL_HEAVY: str = "gpt-4o"
    
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str | None = None
    
    MODEL_TIMEOUT_S: float = 20.0
    QUERY_DEADLINE_S: float = 30.0  
    
    DATA_DIR: str = "data"
    MAX_UPLOAD_MB: int = 50
    
    CHUNK_TARGET_CHARS: int = 450
    CHUNK_MAX_CHARS: int = 1800
    
    GATE_MODE: Literal["hard", "soft"] = "hard"
    LEX_WEIGHT: float = 0.5
    CAND_Z_FLOOR: float = 0.0
    CAND_MIN: int = 20
    CAND_MAX_FACT: int = 30
    CAND_MAX_LIST: int = 80   
    DEDUPE_COSINE: float = 0.92 
    SIBLING_COSINE: float = 0.90
    
    ADJ_MAX_CANDIDATES: int = 100 
    ADJ_SHARD_SIZE: int = 40
    ADJ_MAX_SHARDS: int = 2         
    CONTEXT_TOKEN_BUDGET: int = 10000 
    MAX_SELECTED_CHUNKS: int = 70    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode='after')
    def validate_safety_and_models(self) -> "Settings":
        if self.USE_MOCK_MODELS and self.APP_ENV == "production":
            raise RuntimeError("CRITICAL: USE_MOCK_MODELS cannot be true in production.")
        if self.OPENCODE_MODEL_FAST.endswith("-free") or self.OPENCODE_MODEL_HEAVY.endswith("-free"):
            raise ValueError("Model rejected (cannot end with '-free').")
        return self

settings = Settings()