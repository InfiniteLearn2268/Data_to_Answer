"""
config.py
---------
Every tunable limit lives here and can be overridden from .env, so nothing
is hard-coded in the other files.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# --- storage -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data")).resolve()   # one sub-folder per session

# --- upload ------------------------------------------------------------------
CHUNK_BYTES = _int("CHUNK_BYTES", 8 * 1024 * 1024)                 # browser sends the file in 8 MB pieces
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 20 * 1024 ** 3)        # 20 GB per file
ALLOWED_EXT = (".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet")

# --- DuckDB resources --------------------------------------------------------
DUCKDB_MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "3GB")  # spills to disk beyond this
DUCKDB_THREADS = _int("DUCKDB_THREADS", max(1, (os.cpu_count() or 2)))
MAX_CONCURRENT_INGESTS = _int("MAX_CONCURRENT_INGESTS", 2)
PROFILE_MAX_COLUMNS = _int("PROFILE_MAX_COLUMNS", 60)               # columns profiled for the LLM schema

# --- query safety ------------------------------------------------------------
QUERY_TIMEOUT_S = _int("QUERY_TIMEOUT_S", 60)           # a single query is killed after this
MAX_FETCH_ROWS = _int("MAX_FETCH_ROWS", 1000)           # hard cap on rows pulled from the database
MAX_ROWS_TO_LLM = _int("MAX_ROWS_TO_LLM", 100)          # rows shown to the model
MAX_CHARS_TO_LLM = _int("MAX_CHARS_TO_LLM", 8000)       # size cap on the JSON shown to the model

# --- agent -------------------------------------------------------------------
MAX_TURNS = _int("MAX_TURNS", 6)
MAX_COMPLETION_TOKENS = _int("MAX_COMPLETION_TOKENS", 2048)
CLARIFY_TIMEOUT_S = _int("CLARIFY_TIMEOUT_S", 600)      # a waiting thread gives up after 10 min
MAX_CONCURRENT_QUESTIONS = _int("MAX_CONCURRENT_QUESTIONS", 8)

# --- sessions ----------------------------------------------------------------
SESSION_TTL_S = _int("SESSION_TTL_S", 6 * 3600)         # idle sessions (and their files) are deleted

# --- web ---------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

# Optional fallback database (e.g. your Postgres) -- kept out of source code.
DEFAULT_DB_URI = os.environ.get("DEFAULT_DB_URI", "")
DEFAULT_DB_TABLE = os.environ.get("DEFAULT_DB_TABLE", "orders")
