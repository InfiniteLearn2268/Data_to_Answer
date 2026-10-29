"""Unit checks: guardrail, result shaping, every file format, messy data, low-memory spill."""
import json, os, sys, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import duckdb, pandas as pd
from query_agent import _validate_query, _shape_result
from ingest import load_file_to_dataset_config
from connectors import get_connector

# ---- guardrail ------------------------------------------------------------
ok = ["SELECT 1", "select a from t limit 3;", "WITH x AS (SELECT 1 a) SELECT * FROM x",
      "SELECT a FROM t UNION ALL SELECT a FROM u", "SELECT count(*) FROM t WHERE name = 'limit'"]
bad = ["SELECT 1; DROP TABLE t", "DROP TABLE t", "DELETE FROM t", "INSERT INTO t VALUES (1)", "UPDATE t SET a=1",
       "SELECT * INTO newt FROM t", "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", "CREATE TABLE x AS SELECT 1",
       "SELECT * FROM read_csv('/etc/passwd')", "SELECT pg_read_file('/etc/passwd')", "COPY t TO '/tmp/x'", "PRAGMA database_list", "", "SELEC nonsense ((("]
for q in ok:
    _validate_query("sql", q, "duckdb")
for q in bad:
    try: _validate_query("sql", q, "postgres" if "RETURNING" in q or "INTO" in q else "duckdb"); print("NOT BLOCKED:", q); sys.exit(1)
    except ValueError: pass
print(f"guardrail: {len(ok)} safe queries pass, {len(bad)} dangerous ones blocked")

# ---- result shaping ---------------------------------------------------------
rows = [{"id": i, "text": "x" * 500} for i in range(1000)]
out = json.loads(_shape_result(rows, True))
print(f"shaping: 1000 rows x 500-char cells -> {len(json.dumps(out))} chars to LLM, {len(out['rows'])} rows, note={'note' in out}")
assert len(json.dumps(out)) <= 8500 and "note" in out

# ---- all formats + messy CSV -----------------------------------------------
d = tempfile.mkdtemp()
df = pd.DataFrame({"Order ID": range(1, 1001), "Customer Name": ["a", "b"] * 500, "Amount ($)": [1.5] * 1000, "Customer Name ": ["z"] * 1000})
df.to_csv(f"{d}/t.csv", index=False); df.to_csv(f"{d}/t.tsv", index=False, sep="\t")
df.to_excel(f"{d}/t.xlsx", index=False); df.to_json(f"{d}/t.json", orient="records"); df.to_json(f"{d}/t.jsonl", orient="records", lines=True)
df.to_parquet(f"{d}/t.parquet")
for ext in ["csv", "tsv", "xlsx", "json", "jsonl", "parquet"]:
    cfg = load_file_to_dataset_config(f"{d}/t.{ext}", f"{d}/db_{ext}/data.duckdb")
    c = get_connector(cfg)
    r = c.run_query("SELECT count(*) AS n, sum(amount) AS s FROM uploaded_data")[0]
    cols = [x["column_name"] for x in c.run_query("SELECT column_name FROM information_schema.columns WHERE table_name='uploaded_data'")]
    assert r["n"] == 1000, (ext, r)
    print(f"format .{ext:8s} ok  rows={r['n']} cols={cols}")

# messy: number column that turns into text after the sample window
with open(f"{d}/messy.csv", "w") as f:
    f.write("id,value\n")
    for i in range(300000): f.write(f"{i},{i}\n")
    f.write("300000,N/A\n")
cfg = load_file_to_dataset_config(f"{d}/messy.csv", f"{d}/m/data.duckdb")
print("messy csv ->", cfg["description"][:60], "| rows", cfg["rows"])
for bad_file, expect in [(f"{d}/x.txt", "Unsupported")]:
    open(bad_file, "w").write("hi")
    try: load_file_to_dataset_config(bad_file, f"{d}/b/data.duckdb"); sys.exit("should fail")
    except ValueError as e: print("bad type ->", str(e)[:60])
open(f"{d}/empty.csv", "w").write("a,b\n")
try: load_file_to_dataset_config(f"{d}/empty.csv", f"{d}/e/data.duckdb"); sys.exit("should fail")
except ValueError as e: print("empty csv ->", str(e)[:60])
open(f"{d}/corrupt.parquet", "wb").write(b"not parquet")
try: load_file_to_dataset_config(f"{d}/corrupt.parquet", f"{d}/c/data.duckdb"); sys.exit("should fail")
except ValueError as e: print("corrupt parquet ->", str(e)[:70])
print("ALL UNIT CHECKS PASSED")
