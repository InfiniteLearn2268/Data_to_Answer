"""python make_big_csv.py ROWS out.csv  -- generates a realistic sales CSV fast, using DuckDB."""
import sys, duckdb
rows, out = int(sys.argv[1]), sys.argv[2]
con = duckdb.connect()
con.execute(f"""
COPY (
  SELECT i AS "Order ID",
         DATE '2020-01-01' + CAST(i % 1800 AS INTEGER) AS "Order Date",
         (['North','South','East','West','Central'])[1 + CAST(hash(i) % 5 AS INTEGER)] AS region,
         (['Books','Toys','Games','Garden','Tools','Food','Health','Sports'])[1 + CAST(hash(i*7) % 8 AS INTEGER)] AS category,
         CAST(hash(i*13) % 500000 AS INTEGER) AS customer_id,
         ROUND(5 + (hash(i*31) % 100000) / 100.0, 2) AS amount,
         1 + CAST(hash(i*3) % 9 AS INTEGER) AS quantity,
         (['shipped','pending','returned','cancelled'])[1 + CAST(hash(i*5) % 4 AS INTEGER)] AS status
  FROM range({rows}) t(i)
) TO '{out}' (HEADER, DELIMITER ',')
""")
print("wrote", out)
