"""
make_test_files.py -- creates big test files for DataMuse in every supported format.
Same columns as your sample: order_id, customer_name, region, category, amount, order_date

Setup (once):
    pip install duckdb xlsxwriter xlwt

Run:
    python make_test_files.py               # full size (needs ~6 GB free disk, takes ~2 minutes)
    python make_test_files.py --small       # 1/20 of the size, for a quick check
    python make_test_files.py --out my_folder
"""
import argparse
import os
import time

import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="test_data")
ap.add_argument("--small", action="store_true", help="use 1/20 of the row counts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
k = 20 if args.small else 1

NAMES = ["Vikram", "Priya", "Ravi", "Anita", "Meera", "Divya", "Simran", "Karan", "Arjun", "Neha", "Rohan", "Kavya",
         "Aditya", "Isha", "Manish", "Pooja", "Sanjay", "Tanvi", "Yash", "Zoya", "Nikhil", "Riya", "Amit", "Sneha",
         "Deepak", "Lata", "Harsh", "Ishaan", "Jaya", "Kunal"]
CATEGORIES = ["Electronics", "Clothing", "Furniture", "Groceries", "Books", "Sports"]
REGIONS = ["North", "South", "East", "West"]
lit = lambda xs: "[" + ",".join("'" + x + "'" for x in xs) + "]"


def select(n, date_as_text=False):
    date = "DATE '2024-01-01' + CAST(hash(i * 5) % 1034 AS INTEGER)"
    if date_as_text:
        date = f"CAST({date} AS VARCHAR)"
    return f"""SELECT i + 1 AS order_id,
        {lit(NAMES)}[1 + CAST(hash(i * 11) % {len(NAMES)} AS INTEGER)] AS customer_name,
        {lit(REGIONS)}[1 + CAST(hash(i * 3) % {len(REGIONS)} AS INTEGER)] AS region,
        {lit(CATEGORIES)}[1 + CAST(hash(i * 7) % {len(CATEGORIES)} AS INTEGER)] AS category,
        500 + CAST(hash(i * 13) % 9500 AS INTEGER) AS amount,
        {date} AS order_date
        FROM range({n}) t(i)"""


con = duckdb.connect()


def done(name, n, t):
    path = os.path.join(args.out, name)
    print(f"  {name:20s} {n:>11,} rows  {os.path.getsize(path) / 1e6:8.1f} MB  ({time.time() - t:.0f}s)")


def copy(name, n, options):
    t = time.time()
    con.execute(f"COPY ({select(n)}) TO '{os.path.join(args.out, name)}' {options}")
    done(name, n, t)


print(f"Writing to ./{args.out}/")
copy("sales_10m.tsv", 10_000_000 // k, "(HEADER, DELIMITER '\t')")
copy("sales_10m.jsonl", 10_000_000 // k, "(FORMAT JSON)")
copy("sales_5m.json", 5_000_000 // k, "(FORMAT JSON, ARRAY true)")
copy("sales_50m.parquet", 50_000_000 // k, "(FORMAT PARQUET)")

# ---- Excel (.xlsx): written row by row, so memory stays low -----------------
t = time.time()
import xlsxwriter

n = 1_048_575 // k
cur = con.execute(select(n, date_as_text=True))
cols = [d[0] for d in cur.description]
wb = xlsxwriter.Workbook(os.path.join(args.out, "sales_1m.xlsx"), {"constant_memory": True})
ws = wb.add_worksheet("data")
ws.write_row(0, 0, cols)
r = 1
while True:
    batch = cur.fetchmany(50_000)
    if not batch:
        break
    for row in batch:
        ws.write_row(r, 0, row)
        r += 1
wb.close()
done("sales_1m.xlsx", n, t)

# ---- old Excel (.xls): the format allows at most 65,536 rows including the header
t = time.time()
import xlwt

n = 65_534 // k
rows = con.execute(select(n, date_as_text=True)).fetchall()
wb = xlwt.Workbook()
ws = wb.add_sheet("data")
for j, name in enumerate(cols):
    ws.write(0, j, name)
for i, row in enumerate(rows, 1):
    for j, v in enumerate(row):
        ws.write(i, j, v)
wb.save(os.path.join(args.out, "sales_65k.xls"))
done("sales_65k.xls", n, t)
print("Done. Upload any of these files to DataMuse.")