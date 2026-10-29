"""Starts the real server with the scripted fake LLM plugged in."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import uvicorn
import main
from fake_llm import FakeClient

main.client = FakeClient()
uvicorn.run(main.app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), workers=1, log_level="warning")
