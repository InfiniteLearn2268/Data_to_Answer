# DataMuse v2 — Ask questions about data of any size

Upload a file (CSV, TSV, Excel, JSON, JSON-Lines, Parquet) of several GB, or point it at your
Postgres, and ask questions in plain English.

```
Browser (static/index.html)  --chunked upload / polling-->  FastAPI (main.py)
                                                               |
                                      query_agent.py  <-->  Groq LLM
                                                               |
                                      connectors.py  -->  DuckDB file (per session)  |  Postgres
```

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env        # put your NEW Groq key in it
python main.py              # then open http://127.0.0.1:8000
```
One command, one process: the server also serves the website (Streamlit is no longer needed).
Run **one** uvicorn worker (sessions live in memory); threads handle the concurrency.

## Measured results (2 vCPU / 7 GB RAM sandbox, scripted LLM)
| Data | Upload (local) | Ingest | Typical query | Server peak RAM |
|---|---|---|---|---|
| 200k rows / 11 MB | instant | 0.5 s | 0.2 s | – |
| 10M rows / 528 MB | 1 s | 5.6 s | 0.2–0.4 s | – |
| **50M rows / 2.7 GB** | 5 s | **25 s** | 0.2–1.2 s (GROUP BY / top-N) | **873 MB** |

* 50M rows also works with DuckDB limited to 500 MB RAM (high-cardinality GROUP BY 2.9 s, DISTINCT counts 4.2 s).
* Over a real network, upload time is limited by the user's bandwidth, not by the server.

## What was wrong before, and how it is solved
| Problem | Fix |
|---|---|
| Streamlit's 200 MB upload cap; whole file in RAM twice | Page slices the file into 8 MB chunks (3 in parallel, auto-retry), server streams them to disk |
| Upload/ingest timeouts (120 s / 90 s) and frozen server | Ingestion runs in a background pool; page polls progress with stage + elapsed time |
| pandas → SQLite: slow, row-store full scans | DuckDB: parallel CSV/Parquet/JSON reader, columnar engine, spills to disk |
| One shared `uploaded_data.db` — users overwrote each other | One DuckDB file per session |
| `"limit" in query` check; unbounded `fetchall()` | Row cap enforced inside the database (`SELECT * FROM (...) LIMIT n`), `fetchmany` |
| No query timeout | 60 s timeout (DuckDB `interrupt`, Postgres `statement_timeout`) |
| Huge results sent to the LLM | Rows, cell length and total characters are capped before the model sees them |
| `parse_one` ignored a 2nd statement (`SELECT 1; DROP …`) | Guardrail on the full sqlglot AST: exactly one SELECT, no write/DDL node anywhere, no file functions; DuckDB opened read-only with external file access off; Postgres session read-only |
| Only `tool_calls[0]` answered → API 400 on parallel calls | Every tool call gets a reply |
| Thread leak on unanswered clarification; sessions never freed | 10 min clarification timeout; idle sessions + files deleted (6 h), raw upload deleted after ingest, orphans wiped at startup |
| LLM had only column names | Ingest builds a profile (types, ranges, distinct values of small columns) so filters are correct and ambiguity is detectable |
| Passwords in `dataset_config.py` | Read from `.env` |

## Files
`main.py` server · `ingest.py` file → DuckDB · `connectors.py` backends (DuckDB, Postgres, plus your
Mongo/Elastic/Spark scaffolding, still untested) · `query_agent.py` agent + guardrail · `config.py` all limits ·
`static/index.html` website · `tests/` (fake LLM, row generator, HTTP + browser tests).

## Tests
```bash
python tests/test_units.py                       # guardrail, shaping, all file formats, bad files
python tests/make_big_csv.py 10000000 big.csv
python tests/run_with_fake_llm.py &              # real server, scripted model (no API key needed)
python tests/e2e_http.py big.csv                 # chunked upload + 11 question scenarios
python tests/browser_test.py big.csv .           # real Chromium: upload, clarification, refresh (needs playwright)
```

## Honest limits
* Sessions are in memory, so restarting the server clears them (their files are wiped too). For several
  server processes, move session state to Redis.
* Ingest needs disk of roughly the raw file plus the database (the database was ~15% of the CSV size here).
* One question is limited to 1000 rows from the engine and ~100 rows shown to the LLM: by design,
  the model should aggregate, not read rows.
* LLM-written SQL can still be subtly wrong; sanity-check important numbers.
* Excel is read in a streaming way, but the format itself caps at ~1M rows.
* For data that already lives in a warehouse, don't upload — set `DEFAULT_DB_URI` (Postgres) or add a connector.
* The real Groq model was not called in my tests (no key was used); the pipeline was tested with a scripted model
  that enforces the same tool-reply rules as the real API.
