import asyncio
import json
import time
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError
from langfuse.openai import AsyncOpenAI

from app.config import settings

T = TypeVar("T", bound=BaseModel)

class LLMUnavailable(Exception):
    """Raised when all LLM call attempts (including fallbacks) fail."""
    pass

# Initialize Langfuse-wrapped OpenAI client
client = AsyncOpenAI(
    base_url=settings.OPENCODE_BASE_URL,
    api_key=settings.OPENCODE_API_KEY
)

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return text

async def _execute_with_retry(
    model: str, 
    schema: type[T], 
    system: str, 
    user: str, 
    name: str, 
    fallback_used: bool
) -> tuple[T, dict]:
    # Inject JSON schema into the system prompt
    schema_json = json.dumps(schema.model_json_schema())
    system_prompt = f"{system}\n\nRespond strictly with JSON matching this schema:\n{schema_json}"
    
    for attempt in range(2):
        start_time = time.perf_counter()
        
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=10000,
                    name=f"{name}_att{attempt+1}"
                ),
                timeout=settings.MODEL_TIMEOUT_S
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty or null message content")
                
            clean_content = _strip_code_fences(content)
            parsed_result = schema.model_validate_json(clean_content)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            tokens = response.usage.total_tokens if response.usage else 0
            
            meta = {
                "model": model,
                "tokens": tokens,
                "duration_ms": duration_ms,
                "fallback_used": fallback_used
            }
            
            return parsed_result, meta
            
        except asyncio.TimeoutError:
            # Never retry on timeout
            raise
        except (ValidationError, ValueError):
            if attempt == 1:
                # Max retries reached for validation failure
                raise

async def structured_call(
    tier: Literal["FAST", "HEAVY"], 
    schema: type[T], 
    system: str, 
    user: str, 
    *, 
    name: str
) -> tuple[T, dict]:
    primary_model = settings.OPENCODE_MODEL_FAST if tier == "FAST" else settings.OPENCODE_MODEL_HEAVY
    fallback_model = settings.OPENCODE_MODEL_HEAVY if tier == "FAST" else settings.OPENCODE_MODEL_FAST
    
    try:
        return await _execute_with_retry(
            model=primary_model,
            schema=schema,
            system=system,
            user=user,
            name=name,
            fallback_used=False
        )
    except Exception as primary_exc:
        # If models are the same, don't attempt a redundant fallback
        if primary_model != fallback_model:
            try:
                return await _execute_with_retry(
                    model=fallback_model,
                    schema=schema,
                    system=system,
                    user=user,
                    name=f"{name}_fallback",
                    fallback_used=True
                )
            except Exception as fallback_exc:
                raise LLMUnavailable(
                    f"Both primary ({primary_model}) and fallback ({fallback_model}) models failed. "
                    f"Last error: {fallback_exc}"
                ) from fallback_exc
                
        raise LLMUnavailable(f"Model {primary_model} failed and no fallback available: {primary_exc}") from primary_exc

# requirements:
# langfuse>=2.0.0
# pydantic>=2.0.0
# openai>=1.0.0

# contract:
# input: structured_call(tier="FAST", schema=DocCard, system="Extract data.", user="<delim>Text</delim>", name="doccard_extract")
# output: (DocCard(doc_id="1", ...), {"model": "deepseek-v4.1-flash", "tokens": 150, "duration_ms": 1200.5, "fallback_used": False})