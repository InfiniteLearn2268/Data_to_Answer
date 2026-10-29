"""
query_agent.py
---------------
The agent brain (ReAct loop). Given a question and a Connector it:

  1. builds a grounded system prompt (schema profile + query language + rules)
  2. calls the LLM (Groq, OpenAI-compatible) with two tools:
       - run_query              -> fetch real data
       - ask_user_clarification -> pause and ask the human
  3. validates every query with a code-level guardrail, runs it, and feeds a
     SIZE-LIMITED result (or the error) back to the model
  4. repeats up to MAX_TURNS, then returns the final natural-language answer

WHAT CHANGED vs. the original (behaviour of the loop itself is the same)
  * Guardrail rewritten on the sqlglot AST:
      - exactly ONE statement (the old parse_one() silently ignored a second one,
        so "SELECT 1 LIMIT 1; DROP TABLE x" slipped through on Postgres)
      - the statement must be a SELECT / set-operation, and no write/DDL node
        may appear anywhere inside it (incl. CTEs and SELECT ... INTO)
      - file-reading / admin functions are blocked
      - the weak `"limit" in query` substring test is gone: row limits are now
        enforced inside the database by the connector (see connectors.py)
  * Results are capped before the LLM sees them (rows, characters, long cells),
    so a big result can no longer blow the token limit.
  * EVERY tool call in a turn is answered. The original handled only
    tool_calls[0]; the API rejects the next request when a call has no reply.
  * Bad tool arguments / API hiccups are fed back instead of crashing.
"""

import json
import os
import re

from openai import OpenAI

from config import (
    MAX_CHARS_TO_LLM,
    MAX_COMPLETION_TOKENS,
    MAX_ROWS_TO_LLM,
    MAX_TURNS,
)
from connectors import BaseConnector

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# Default clarification handler: a terminal prompt (main.py passes its own).
# ---------------------------------------------------------------------------
def _default_clarify(question: str) -> str:
    print(f"\n[clarification needed] {question}")
    return input("Your answer: ").strip()


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------
_BLOCKED_FUNCTIONS = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|lo_import|lo_export|dblink|pg_sleep|"
    r"read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|read_blob|glob|"
    r"sniff_csv|parquet_scan|csv_scan|query_table|load_extension|install_extension)\s*\(",
    re.IGNORECASE,
)


def _validate_query(query_language: str, query, dialect: str = None) -> None:
    """Raises ValueError if the query looks unsafe. Runs BEFORE execution no
    matter what the LLM was told -- prompts can be ignored, code cannot."""
    if query_language not in ("sql", "spark-sql"):
        # nosql / vector queries are JSON filter objects, not executable code.
        return

    import sqlglot
    from sqlglot import exp

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(query, read=dialect) if s is not None]
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"SQL could not be parsed: {exc}")

    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed (guardrail).")
    stmt = statements[0]

    set_op = getattr(exp, "SetOperation", exp.Union)
    if not isinstance(stmt, (exp.Select, set_op, exp.Subquery)):
        raise ValueError("Only SELECT queries are permitted (guardrail).")

    forbidden_names = ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Merge", "Command",
                       "Copy", "Pragma", "Set", "Use", "Attach", "Detach", "Truncate", "Grant", "Transaction")
    forbidden = tuple(getattr(exp, n) for n in forbidden_names if hasattr(exp, n))
    for node in stmt.find_all(*forbidden):
        raise ValueError(f"'{type(node).__name__}' operations are not permitted (guardrail).")
    for node in stmt.find_all(exp.Select):
        if node.args.get("into"):
            raise ValueError("SELECT ... INTO is not permitted (guardrail).")

    if _BLOCKED_FUNCTIONS.search(query):
        raise ValueError("That function is not permitted (guardrail).")


# ---------------------------------------------------------------------------
# Result shaping: what the LLM is allowed to see
# ---------------------------------------------------------------------------
def _shape_result(rows: list, db_truncated: bool) -> str:
    """Caps rows, long cells and total characters so the model's context
    (and the provider's tokens-per-minute limit) can't be overrun."""
    def trim(value):
        if isinstance(value, str) and len(value) > 200:
            return value[:200] + "…"
        return value

    total = len(rows)
    shown = [{k: trim(v) for k, v in row.items()} for row in rows[:MAX_ROWS_TO_LLM]]

    def render(n: int) -> str:
        payload = {"rows": shown[:n], "rows_returned": f"{total}+" if db_truncated else total}
        if n < total or db_truncated:
            payload["note"] = (
                f"Only the first {n} rows are shown. The full result is larger. Use aggregation "
                f"(COUNT/SUM/AVG/GROUP BY), a narrower filter, or ORDER BY ... LIMIT for precise answers."
            )
        return json.dumps(payload, default=str)

    n = len(shown)
    text = render(n)
    while len(text) > MAX_CHARS_TO_LLM and n > 1:
        n = max(1, n // 2)
        text = render(n)
    return text


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
def _build_tool_schema(query_language: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_query",
                "description": (
                    f"Execute a {query_language} query against the connected database "
                    f"and return the results. Call this whenever you need real data "
                    f"to answer the user's question, AND you have enough information "
                    f"to write a precise, unambiguous query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"A valid {query_language} query as a string. "
                                           f"For mongo-json or es-dsl, pass the JSON body as a string.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user_clarification",
                "description": (
                    "Call this INSTEAD of run_query when the user's question is genuinely "
                    "ambiguous -- for example, it references a filter (like a region, category, "
                    "or date range) without specifying which value to use, and the dataset has "
                    "multiple possible values for it. Ask one clear, specific question that would "
                    "let you write a precise query once answered. Do not guess a value on your own "
                    "when a real ambiguity exists."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The clarifying question to ask the user, in plain language.",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
    ]


def _tool_message(tool_call, payload: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call.id, "content": payload}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def ask(
    question: str,
    connector: BaseConnector,
    client: OpenAI,
    clarify_fn=_default_clarify,
) -> str:
    """Runs the full agent loop for one question and returns the final answer."""

    schema_text = connector.get_schema()
    tools = _build_tool_schema(connector.query_language)

    system_prompt = f"""You are a data-retrieval agent. You answer questions using
ONLY the connected dataset below -- never invent field/table names, values, or numbers.

Query language to use: {connector.query_language}
Schema (with value ranges and, for small categorical columns, the full list of values):
{schema_text}

Rules:
- The table can hold tens of millions of rows. NEVER select raw rows to "look through" them.
  Push the work into the database: COUNT, SUM, AVG, MIN, MAX, GROUP BY, ORDER BY ... LIMIT.
- Always call the `run_query` tool to fetch real data before answering,
  UNLESS the question is out of scope (see below).
- Use the exact column names and category values shown in the schema above.
- If a question is ambiguous -- e.g. it mentions a filter like a region, category,
  product, or date range without saying which one, and the dataset has multiple
  possible values -- call `ask_user_clarification` and ask a specific question
  BEFORE calling run_query. Do not guess or pick an arbitrary value yourself.
- Once the user has answered your clarifying question, use that answer to write
  a precise query.
- OUT-OF-SCOPE QUESTIONS: if the question is unrelated to this dataset -- general
  knowledge (e.g. "what's the capital of France"), casual conversation (e.g. "how
  are you", "tell me a joke"), or something about columns/entities that simply do
  not exist in the schema above -- do NOT call run_query and do NOT guess an
  answer. Instead, reply directly with plain text: politely say you can only
  answer questions about this connected dataset, and briefly mention (in one
  sentence) what the dataset actually contains, based on the schema above, so
  the user knows what they can ask instead.
- Write a single SELECT statement per call (a WITH ... SELECT is fine).
- If a query fails, read the error message and try a corrected query.
- If a result says it was truncated, refine the query (aggregate or filter) instead of guessing.
- Give a concise, factual final answer grounded only in the returned rows.
- A result of 0 records, a count of 0, or an empty result set is a valid and meaningful answer.
- NEVER retry a successful query merely because its result is 0 or empty.
- If the query executed successfully and the returned result is 0/empty, accept that result and answer the user directly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_COMPLETION_TOKENS,
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_unset=True))

        tool_calls = message.tool_calls
        if not tool_calls:
            # No tool call -> final answer (real data answer OR out-of-scope text).
            return message.content or ""

        clarified_this_turn = False

        # Answer EVERY tool call -- the API requires a reply for each id.
        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                messages.append(_tool_message(tool_call, json.dumps({"error": "Tool arguments were not valid JSON."})))
                continue

            # ---- Branch 1: ask the human --------------------------------
            if name == "ask_user_clarification":
                if clarified_this_turn:
                    messages.append(_tool_message(tool_call, json.dumps(
                        {"error": "Skipped: ask only one clarifying question at a time."})))
                    continue
                clarified_this_turn = True
                clarifying_question = args.get("question", "Could you clarify your question?")
                print(f"[turn {turn}] asking for clarification:\n{clarifying_question}\n")
                user_answer = clarify_fn(clarifying_question)
                messages.append(_tool_message(tool_call, json.dumps({"user_response": user_answer})))
                continue

            # ---- Branch 2: run a query ----------------------------------
            if name != "run_query":
                messages.append(_tool_message(tool_call, json.dumps({"error": f"Unknown tool '{name}'."})))
                continue

            raw_query = args.get("query", "")
            print(f"[turn {turn}] proposed {connector.query_language} query:\n{raw_query}\n")
            try:
                _validate_query(connector.query_language, raw_query, getattr(connector, "dialect", None))
                executable = json.loads(raw_query) if connector.query_language in ("mongo-json", "es-dsl") else raw_query
                rows = connector.run_query(executable)
                tool_output = _shape_result(rows, getattr(connector, "last_truncated", False))
            except Exception as exc:
                # Feed the error back -- the "self-repair" step.
                tool_output = json.dumps({"error": str(exc)[:1500]})
            messages.append(_tool_message(tool_call, tool_output))

    return "Agent could not produce a valid query within the turn limit."
