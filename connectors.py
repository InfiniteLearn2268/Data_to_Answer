"""
connectors.py
-------------
ONE common interface (`BaseConnector`) that every backend implements, so the
LLM agent never has to know whether it is talking to DuckDB, Postgres,
MongoDB, Elasticsearch or Spark. Each connector exposes:

    .query_language   -> str, the dialect the LLM must write ("sql", ...)
    .dialect          -> str, sqlglot dialect used by the guardrail
    .get_schema()     -> str, text the LLM reads to ground its queries
    .run_query(query) -> list[dict], at most MAX_FETCH_ROWS rows
    .last_truncated   -> bool, True if the last result was cut at that cap

WHAT CHANGED vs. the original
  * NEW DuckDBConnector -- the default for uploaded files (fast on 50M+ rows).
  * Every SQL connector now enforces limits IN THE DATABASE, not in Python:
      - the query is wrapped as  SELECT * FROM (<query>) LIMIT n  so the
        engine never builds more than n rows (no more fetchall() of millions)
      - a per-query timeout kills runaway queries
      - the connection is read-only
  * DuckDB queries run with external file access disabled, so a generated
    query can't read arbitrary files from the server.
"""

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    MAX_FETCH_ROWS,
    QUERY_TIMEOUT_S,
)


def _strip_semicolon(query: str) -> str:
    return query.strip().rstrip(";").strip()


def _wrap_with_limit(query: str, n: int) -> str:
    """Make the DATABASE stop after n rows (we ask for n+1 so we can tell
    whether the result was truncated)."""
    return f"SELECT * FROM ({_strip_semicolon(query)}) AS _agent_q LIMIT {n + 1}"


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseConnector(ABC):
    """Every backend-specific connector must implement these methods."""

    query_language: str = "unknown"
    dialect: str = "unknown"          # sqlglot dialect name, used by the SQL guardrail
    last_truncated: bool = False

    @abstractmethod
    def get_schema(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run_query(self, query: Any) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. Optional."""


# ---------------------------------------------------------------------------
# 0. DuckDB connector -- uploaded files (CSV / Excel / JSON / Parquet)
# ---------------------------------------------------------------------------
class DuckDBConnector(BaseConnector):
    query_language = "sql"
    dialect = "duckdb"

    def __init__(self, config: dict):
        self.db_path = config["db_path"]
        self.target_table = config["target"]

    def get_schema(self) -> str:
        # ingest.py stored a profile (types, ranges, distinct values) next to
        # the database, so this costs nothing even for 50M rows.
        profile = Path(self.db_path).with_suffix(".profile.txt")
        if profile.exists():
            return profile.read_text(encoding="utf-8")
        import duckdb
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            lines = [f"Table: {self.target_table}"]
            for name, ctype, *_ in con.execute(f'DESCRIBE "{self.target_table}"').fetchall():
                lines.append(f"  - {name} ({ctype})")
            return "\n".join(lines)
        finally:
            con.close()

    def run_query(self, query: str) -> list[dict]:
        import duckdb

        # A fresh read-only connection per query: nothing can modify the data,
        # and concurrent questions from different sessions never interfere.
        con = duckdb.connect(
            self.db_path,
            read_only=True,
            config={
                "memory_limit": DUCKDB_MEMORY_LIMIT,
                "threads": DUCKDB_THREADS,
                # (temp_directory can't be set once external access is off; DuckDB
                #  spills next to the database file by default, which is what we want)
                "enable_external_access": False,   # no read_csv('/etc/passwd') etc.
            },
        )
        timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)   # kills the query after the timeout
        timer.start()
        try:
            cur = con.execute(_wrap_with_limit(query, MAX_FETCH_ROWS))
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except duckdb.InterruptException:
            raise TimeoutError(
                f"Query exceeded {QUERY_TIMEOUT_S}s and was cancelled. "
                f"Use filters or aggregation (GROUP BY / COUNT / SUM) to make it cheaper."
            )
        finally:
            timer.cancel()
            con.close()
        self.last_truncated = len(rows) > MAX_FETCH_ROWS
        return [dict(zip(columns, r)) for r in rows[:MAX_FETCH_ROWS]]


# ---------------------------------------------------------------------------
# 1. SQL connector (Postgres / MySQL / Snowflake / BigQuery via SQLAlchemy)
# ---------------------------------------------------------------------------
class SQLConnector(BaseConnector):
    query_language = "sql"

    def __init__(self, config: dict):
        from sqlalchemy import create_engine, inspect

        uri = config["connection_uri"]
        connect_args = {}
        if uri.startswith("postgresql"):
            # Enforced by the database itself: read-only transactions and a
            # server-side timeout. Even a bypassed guardrail can't write.
            connect_args["options"] = (
                f"-c statement_timeout={QUERY_TIMEOUT_S * 1000} -c default_transaction_read_only=on"
            )
        self.engine = create_engine(uri, pool_pre_ping=True, connect_args=connect_args)
        self.target_table = config["target"]
        self.inspector = inspect(self.engine)
        self.dialect = {"postgresql": "postgres"}.get(self.engine.dialect.name, self.engine.dialect.name)

    def get_schema(self) -> str:
        columns = self.inspector.get_columns(self.target_table)
        lines = [f"Table: {self.target_table}"]
        for col in columns:
            lines.append(f"  - {col['name']} ({col['type']})")
        return "\n".join(lines)

    def run_query(self, query: str) -> list[dict]:
        from sqlalchemy import text

        with self.engine.connect() as conn:
            result = conn.execute(text(_wrap_with_limit(query, MAX_FETCH_ROWS)))
            columns = list(result.keys())
            rows = result.fetchmany(MAX_FETCH_ROWS + 1)      # never fetchall()
        self.last_truncated = len(rows) > MAX_FETCH_ROWS
        return [dict(zip(columns, r)) for r in rows[:MAX_FETCH_ROWS]]

    def close(self) -> None:
        self.engine.dispose()


# ---------------------------------------------------------------------------
# 2. NoSQL connector (MongoDB-style document store) -- unchanged scaffolding
# ---------------------------------------------------------------------------
class NoSQLConnector(BaseConnector):
    query_language = "mongo-json"

    def __init__(self, config: dict):
        from pymongo import MongoClient

        client = MongoClient(config["connection_uri"])
        self.db = client[config["database"]]
        self.collection = self.db[config["target"]]

    def get_schema(self) -> str:
        sample_docs = list(self.collection.find().limit(3))
        fields = sorted({key for doc in sample_docs for key in doc.keys()})
        return f"Collection: {self.collection.name}\nObserved fields: {fields}\nSample doc: {sample_docs[0] if sample_docs else '{}'}"

    def run_query(self, query: dict) -> list[dict]:
        cursor = self.collection.find(query).limit(MAX_FETCH_ROWS)
        return [{k: v for k, v in doc.items() if k != "_id"} for doc in cursor]


# ---------------------------------------------------------------------------
# 3. Vector / hybrid search connector -- unchanged scaffolding
# ---------------------------------------------------------------------------
class VectorConnector(BaseConnector):
    query_language = "es-dsl"

    def __init__(self, config: dict):
        from elasticsearch import Elasticsearch

        self.client = Elasticsearch(config["connection_uri"])
        self.index = config["target"]
        self.text_field = config["text_field"]
        self.embedding_field = config["embedding_field"]

    def get_schema(self) -> str:
        mapping = self.client.indices.get_mapping(index=self.index)
        fields = list(mapping[self.index]["mappings"]["properties"].keys())
        return f"Index: {self.index}\nFields: {fields}\nText field: {self.text_field}\nVector field: {self.embedding_field}"

    def run_query(self, query: dict) -> list[dict]:
        response = self.client.search(index=self.index, body=query, size=20)
        return [hit["_source"] for hit in response["hits"]["hits"]]


# ---------------------------------------------------------------------------
# 4. Big-data connector (Spark SQL over Parquet/Delta) -- unchanged scaffolding
# ---------------------------------------------------------------------------
class BigDataConnector(BaseConnector):
    query_language = "spark-sql"
    dialect = "spark"

    def __init__(self, config: dict):
        from pyspark.sql import SparkSession

        self.spark = (
            SparkSession.builder.master(config["connection_uri"]).appName("agentic-retrieval").getOrCreate()
        )
        self.view_name = config["view_name"]
        df = self.spark.read.parquet(config["target"])
        df.createOrReplaceTempView(self.view_name)

    def get_schema(self) -> str:
        df = self.spark.table(self.view_name)
        lines = [f"View: {self.view_name}"]
        for field in df.schema.fields:
            lines.append(f"  - {field.name} ({field.dataType})")
        return "\n".join(lines)

    def run_query(self, query: str) -> list[dict]:
        result_df = self.spark.sql(query)
        return [row.asDict() for row in result_df.limit(MAX_FETCH_ROWS).collect()]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_CONNECTOR_REGISTRY = {
    "duckdb": DuckDBConnector,
    "sql": SQLConnector,
    "nosql": NoSQLConnector,
    "vector": VectorConnector,
    "bigdata": BigDataConnector,
}


def get_connector(config: dict) -> BaseConnector:
    """Single entry point -- reads config['type'] and builds the connector."""
    backend_type = config["type"]
    connector_cls = _CONNECTOR_REGISTRY.get(backend_type)
    if connector_cls is None:
        raise ValueError(f"Unknown dataset type '{backend_type}'. Valid options: {list(_CONNECTOR_REGISTRY)}")
    return connector_cls(config)
