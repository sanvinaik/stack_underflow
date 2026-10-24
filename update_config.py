from pathlib import Path

cfg_path = Path("app/config.py")
text = cfg_path.read_text(encoding="utf-8")
text = text.replace("CAND_Z_FLOOR: float = 0.5", "CAND_Z_FLOOR: float = 0.0")
text = text.replace("DEDUPE_COSINE: float = 0.90", "DEDUPE_COSINE: float = 0.92")
text = text.replace("ADJ_MAX_SHARDS: int = 1", "ADJ_MAX_SHARDS: int = 2")
text = text.replace("CONTEXT_TOKEN_BUDGET: int = 8000", "CONTEXT_TOKEN_BUDGET: int = 10000")
text = text.replace("MAX_SELECTED_CHUNKS: int = 25", "MAX_SELECTED_CHUNKS: int = 70")

cfg_path.write_text(text, encoding="utf-8")
print("config.py updated.")
