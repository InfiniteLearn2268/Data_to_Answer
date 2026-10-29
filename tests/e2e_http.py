"""End-to-end test over real HTTP against a running server: chunked upload,
ingestion progress, then a battery of questions. Usage: python e2e_http.py file.csv"""
import sys, time, os, json, math, requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
path = sys.argv[1]
size = os.path.getsize(path)
cfg = requests.get(f"{BASE}/config").json()
CH = cfg["chunk_bytes"]

t0 = time.time()
init = requests.post(f"{BASE}/upload/init", json={"filename": os.path.basename(path), "size": size}).json()
sid, n = init["session_id"], init["chunks"]
def send(i):
    with open(path, "rb") as f:
        f.seek(i * CH); data = f.read(CH)
    r = requests.put(f"{BASE}/upload/{sid}/chunk", params={"index": i}, data=data); r.raise_for_status()
with ThreadPoolExecutor(3) as ex: list(ex.map(send, range(n)))
t_up = time.time() - t0
print(f"upload  {size/1e6:,.0f} MB in {t_up:.1f}s ({size/1e6/t_up:,.0f} MB/s), {n} chunks")

requests.post(f"{BASE}/upload/{sid}/complete").raise_for_status()
t1 = time.time(); last = None
while True:
    s = requests.get(f"{BASE}/session/{sid}").json()
    if s["stage"] != last: print("   stage:", s["stage"], f"({time.time()-t1:.0f}s)"); last = s["stage"]
    if s["phase"] in ("ready", "error"): break
    time.sleep(0.5)
print(f"ingest  {time.time()-t1:.1f}s -> {s['phase']}: {s.get('error') or s['description']}")
assert s["phase"] == "ready"

def ask(q, answer=None):
    t = time.time()
    requests.post(f"{BASE}/ask", json={"session_id": sid, "question": q}).raise_for_status()
    while True:
        d = requests.get(f"{BASE}/status/{sid}").json()
        if d["status"] == "needs_clarification":
            print(f"   (clarify asked: {d['question']}  -> answering {answer!r})")
            requests.post(f"{BASE}/clarify", json={"session_id": sid, "answer": answer}).raise_for_status()
        elif d["status"] != "processing":
            break
        time.sleep(0.2)
    out = (d["answer"] or d["error"] or "")
    print(f"ask     [{time.time()-t:5.1f}s] {q!r}\n          -> {out[:260]}")
    return d

ask("how many rows are there")
ask("total amount by region")
ask("who are the top customers")
ask("an ambiguous question about sales in a region", answer="North")
ask("tell me a joke")
ask("two calls please")
ask("show me everything")
ask("drop the table")
ask("readfile please")
if os.environ.get("SLOW"): ask("slow query")
ask("how many rows are there")   # table must still exist after the DROP attempt
requests.delete(f"{BASE}/session/{sid}")
print("done")
