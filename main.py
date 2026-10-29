"""
main.py
-------
FastAPI server. Serves the website AND the API, so there is nothing else to
run:   python main.py     (or:  uvicorn main:app --port 8000)

FLOW (all calls return instantly; the browser polls for progress)
    GET  /                        the website (static/index.html)
    GET  /config                  limits the page needs (chunk size, max size, ...)

  Upload a file of ANY size -- the browser slices it into 8 MB chunks:
    POST /upload/init             {filename, size}        -> {session_id, chunk_bytes}
    PUT  /upload/{id}/chunk?index=N   (raw bytes)         -> {received}
    POST /upload/{id}/complete                            -> starts ingestion in the background
    GET  /session/{id}            poll: phase = uploading | ingesting | ready | error (+ stage text)

  Or use the configured default database:
    POST /session/default

  Ask questions (same job pattern as before):
    POST /ask      {session_id, question}   -> {"status": "processing"}
    GET  /status/{id}   -> processing | needs_clarification | done | error
    POST /clarify  {session_id, answer}
    DELETE /session/{id}                    free the session's files now

WHAT CHANGED vs. the original main.py -- each item fixes a problem found in review
  * Chunked, streaming upload straight to disk: the server never holds the file
    in RAM (old: `await file.read()` of the whole file), no 200 MB Streamlit cap,
    and a failed chunk is simply re-sent (resumable).
  * Ingestion runs in a worker pool, not inside the request: the server stays
    responsive and there is no 120 s upload timeout.
  * One DuckDB file per session: users can't overwrite each other's data.
  * Questions run in a bounded thread pool; a clarification nobody answers
    times out instead of leaking a thread forever.
  * Idle sessions and their files are deleted automatically; the raw upload is
    deleted right after ingestion; orphaned folders are wiped at startup.
  * The Groq key is optional at import time, so the server can start and show
    a clear error instead of crashing with KeyError.
"""

import math
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

import config
from connectors import get_connector
from dataset_config import DATASET_CONFIG
from ingest import load_file_to_dataset_config
from query_agent import ask

SESSIONS_DIR = config.DATA_DIR / "sessions"
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# LLM client (built lazily; tests may replace `main.client`)
# ---------------------------------------------------------------------------
client: Optional[OpenAI] = None


def get_client():
    global client
    if client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file and restart the server.")
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", max_retries=4, timeout=90)
    return client


# ---------------------------------------------------------------------------
# Session store (in memory; metadata only -- the data lives in per-session files)
# ---------------------------------------------------------------------------
sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()

ingest_pool = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_INGESTS, thread_name_prefix="ingest")
ask_pool = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_QUESTIONS, thread_name_prefix="ask")


def _new_session(sid: str, **extra) -> dict:
    now = time.time()
    s = {
        "id": sid,
        "created": now,
        "last_active": now,
        "dir": SESSIONS_DIR / sid,
        # upload / ingestion lifecycle
        "phase": "awaiting_upload",     # awaiting_upload | uploading | ingesting | ready | error
        "stage": "",
        "phase_error": None,
        "ingest_started": None,
        "filename": None,
        "size": 0,
        "chunks_expected": 0,
        "chunks_received": set(),
        "file_lock": threading.Lock(),
        # data
        "connector": None,
        "backend_type": None,
        "description": "",
        # current question
        "status": "idle",               # idle | processing | needs_clarification | done | error
        "question": None,
        "answer": None,
        "error": None,
        "clarify_event": None,
        "clarify_answer": None,
    }
    s.update(extra)
    return s


def _get(sid: str) -> dict:
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Upload a file first.")
    s["last_active"] = time.time()
    return s


def _destroy_session(sid: str) -> None:
    with sessions_lock:
        s = sessions.pop(sid, None)
    if s is None:
        return
    ev = s.get("clarify_event")
    if ev is not None:
        ev.set()                      # release a waiting thread
    try:
        if s.get("connector") is not None:
            s["connector"].close()
    except Exception:
        pass
    shutil.rmtree(s["dir"], ignore_errors=True)


def _janitor() -> None:
    """Deletes idle sessions (and their database files) after SESSION_TTL_S."""
    while True:
        time.sleep(60)
        now = time.time()
        for sid, s in list(sessions.items()):
            busy = s["status"] in ("processing", "needs_clarification") or s["phase"] == "ingesting"
            if not busy and now - s["last_active"] > config.SESSION_TTL_S:
                _destroy_session(sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    shutil.rmtree(SESSIONS_DIR, ignore_errors=True)    # folders from a previous run are orphans
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_janitor, daemon=True).start()
    yield


app = FastAPI(title="DataMuse API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class InitUpload(BaseModel):
    filename: str
    size: int


class AskRequest(BaseModel):
    session_id: str
    question: str


class ClarifyRequest(BaseModel):
    session_id: str
    answer: str


# ---------------------------------------------------------------------------
# Website + config
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    return {
        "chunk_bytes": config.CHUNK_BYTES,
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "allowed_ext": list(config.ALLOWED_EXT),
        "default_available": DATASET_CONFIG is not None,
        "llm_configured": bool(os.environ.get("GROQ_API_KEY")) or client is not None,
    }


# ---------------------------------------------------------------------------
# Upload (chunked) + ingestion
# ---------------------------------------------------------------------------
@app.post("/upload/init")
def upload_init(req: InitUpload):
    ext = os.path.splitext(req.filename)[1].lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {', '.join(config.ALLOWED_EXT)}")
    if req.size <= 0:
        raise HTTPException(400, "The file is empty.")
    if req.size > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Limit is {config.MAX_UPLOAD_BYTES / 1024 ** 3:.0f} GB.")

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(SESSIONS_DIR).free
    if free < req.size * 2.2:     # raw upload + the database built from it
        raise HTTPException(507, "The server does not have enough free disk space for this file.")

    sid = uuid.uuid4().hex
    s = _new_session(sid)
    s["dir"].mkdir(parents=True, exist_ok=True)
    s["filename"] = req.filename
    s["size"] = req.size
    s["chunks_expected"] = math.ceil(req.size / config.CHUNK_BYTES)
    s["upload_path"] = str(s["dir"] / f"upload{ext}")
    with open(s["upload_path"], "wb") as f:
        f.truncate(req.size)       # pre-size the file so chunks can land at any offset
    s["phase"] = "uploading"
    with sessions_lock:
        sessions[sid] = s
    return {"session_id": sid, "chunk_bytes": config.CHUNK_BYTES, "chunks": s["chunks_expected"]}


def _write_chunk(s: dict, index: int, body: bytes) -> None:
    with s["file_lock"]:
        with open(s["upload_path"], "r+b") as f:
            f.seek(index * config.CHUNK_BYTES)
            f.write(body)
        s["chunks_received"].add(index)


@app.put("/upload/{sid}/chunk")
async def upload_chunk(sid: str, index: int, request: Request):
    s = _get(sid)
    if s["phase"] != "uploading":
        raise HTTPException(409, "This session is not accepting uploads.")
    if not 0 <= index < s["chunks_expected"]:
        raise HTTPException(400, "Chunk index out of range.")
    is_last = index == s["chunks_expected"] - 1
    expected_len = s["size"] - index * config.CHUNK_BYTES if is_last else config.CHUNK_BYTES
    body = await request.body()                      # at most one chunk (8 MB) in memory
    if len(body) != expected_len:
        raise HTTPException(400, f"Chunk {index} should be {expected_len} bytes, got {len(body)}.")
    await asyncio.to_thread(_write_chunk, s, index, body)
    return {"received": len(s["chunks_received"]), "expected": s["chunks_expected"]}


def _ingest_job(sid: str) -> None:
    s = sessions.get(sid)
    if s is None:
        return
    s["ingest_started"] = time.time()
    db_path = str(s["dir"] / "data.duckdb")

    def progress(stage: str) -> None:
        s["stage"] = stage

    try:
        cfg = load_file_to_dataset_config(s["upload_path"], db_path, progress=progress, original_name=s["filename"])
        s["connector"] = get_connector(cfg)
        s["backend_type"] = cfg["type"]
        s["description"] = cfg["description"]
        s["stage"] = "Ready"
        s["phase"] = "ready"
    except Exception as exc:
        s["phase_error"] = str(exc)
        s["phase"] = "error"
    finally:
        try:
            os.remove(s["upload_path"])              # free disk: the database replaces the raw upload
        except OSError:
            pass


@app.post("/upload/{sid}/complete")
def upload_complete(sid: str):
    s = _get(sid)
    if s["phase"] != "uploading":
        raise HTTPException(409, "Upload already completed or session not uploading.")
    missing = s["chunks_expected"] - len(s["chunks_received"])
    if missing:
        raise HTTPException(400, f"{missing} chunk(s) missing. Re-send them, then call complete again.")
    s["phase"] = "ingesting"
    s["stage"] = "Queued"
    ingest_pool.submit(_ingest_job, sid)
    return {"phase": "ingesting"}


@app.post("/session/default")
def create_default_session():
    if DATASET_CONFIG is None:
        raise HTTPException(404, "No default database is configured (set DEFAULT_DB_URI in .env).")
    try:
        connector = get_connector(DATASET_CONFIG)
        connector.get_schema()
    except Exception as exc:
        raise HTTPException(400, f"Couldn't connect to the default database: {exc}")
    sid = uuid.uuid4().hex
    s = _new_session(sid, phase="ready", stage="Ready", connector=connector,
                     backend_type=DATASET_CONFIG["type"], description=DATASET_CONFIG.get("description", ""))
    s["filename"] = "Default database"
    with sessions_lock:
        sessions[sid] = s
    return {"session_id": sid, "backend_type": s["backend_type"], "description": s["description"]}


@app.get("/session/{sid}")
def session_info(sid: str):
    s = _get(sid)
    out = {
        "phase": s["phase"],
        "stage": s["stage"],
        "description": s["description"],
        "filename": s["filename"],
        "backend_type": s["backend_type"],
        "error": s["phase_error"],
    }
    if s["phase"] == "uploading":
        out["chunks_received"] = len(s["chunks_received"])
        out["chunks_expected"] = s["chunks_expected"]
    if s["phase"] == "ingesting" and s["ingest_started"]:
        out["elapsed_s"] = round(time.time() - s["ingest_started"], 1)
    return out


@app.delete("/session/{sid}")
def delete_session(sid: str):
    _destroy_session(sid)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def _run_ask(sid: str, question: str) -> None:
    s = sessions.get(sid)
    if s is None:
        return

    def clarify_fn(clarifying_question: str) -> str:
        ev = threading.Event()
        s["clarify_event"] = ev                       # create the event BEFORE announcing the question
        s["question"] = clarifying_question
        s["status"] = "needs_clarification"
        if not ev.wait(timeout=config.CLARIFY_TIMEOUT_S):
            raise TimeoutError("No answer to the clarifying question was received in time.")
        if sid not in sessions:
            raise RuntimeError("Session was closed.")
        answer = s["clarify_answer"]
        s["clarify_answer"], s["question"], s["clarify_event"] = None, None, None
        s["status"] = "processing"
        return answer

    try:
        result = ask(question=question, connector=s["connector"], client=get_client(), clarify_fn=clarify_fn)
        s["answer"] = result
        s["status"] = "done"
    except Exception as exc:
        s["error"] = str(exc)
        s["status"] = "error"
    finally:
        s["last_active"] = time.time()


@app.post("/ask")
def ask_question(req: AskRequest):
    s = _get(req.session_id)
    if s["phase"] != "ready":
        raise HTTPException(409, "The dataset isn't ready yet.")
    if s["status"] in ("processing", "needs_clarification"):
        raise HTTPException(409, "This session is already processing a question.")
    if not req.question.strip():
        raise HTTPException(400, "Please type a question.")
    s["status"], s["answer"], s["error"], s["question"] = "processing", None, None, None
    ask_pool.submit(_run_ask, req.session_id, req.question.strip())
    return {"status": "processing"}


@app.get("/status/{sid}")
def get_status(sid: str):
    s = _get(sid)
    return {"status": s["status"], "question": s["question"], "answer": s["answer"], "error": s["error"]}


@app.post("/clarify")
def clarify(req: ClarifyRequest):
    s = _get(req.session_id)
    ev = s["clarify_event"]
    if s["status"] != "needs_clarification" or ev is None:
        raise HTTPException(409, "This session isn't waiting for clarification right now.")
    s["clarify_answer"] = req.answer
    ev.set()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # ONE worker on purpose: sessions live in this process's memory.
    uvicorn.run("main:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)),
                workers=1, timeout_keep_alive=75)
