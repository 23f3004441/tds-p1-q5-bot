"""
TDS P1 Q5 — Data Analyst Telegram Bot (webhook, FastAPI).

Receives Telegram updates at POST /webhook, runs each message through the
agent (agent.py), replies via the Telegram Bot API, and logs every step as
JSONL (gcs_logger.py) to a public URL used as `log_url` in final answers.

Env vars required (set on Render):
    BOT_TOKEN         - from @BotFather
    AIPIPE_TOKEN      - from aipipe.org/login
    GITHUB_TOKEN      - personal access token with repo write access
    GITHUB_REPO       - "owner/repo" that will hold the run log
Optional:
    AIPIPE_BASE_URL   (default https://aipipe.org/openai/v1)
    AIPIPE_MODEL      (default gpt-4o-mini)
    GITHUB_LOG_PATH   (default logs/run.jsonl)
    GITHUB_BRANCH     (default main)
"""
import asyncio
import json
import os

import httpx
from fastapi import BackgroundTasks, FastAPI, Request

from agent import run_turn
from github_logger import log_event

BOT_TOKEN = os.environ["BOT_TOKEN"]
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

app = FastAPI()

# in-memory per-chat state: {chat_id: {"history": [...], "ns": {...}}}
CHATS: dict = {}
MAX_HISTORY = 30
OVERALL_TURN_TIMEOUT = 150  # seconds — hard cap on one full agent turn (leaves room for a
                             # forced best-guess fallback call if real attempts are exhausted)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(req: Request, background_tasks: BackgroundTasks):
    update = await req.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return {"ok": True}
    chat_id = msg["chat"]["id"]
    text = msg["text"]
    background_tasks.add_task(handle_message, chat_id, text)
    return {"ok": True}


async def handle_message(chat_id: int, text: str):
    state = CHATS.setdefault(chat_id, {"history": [], "ns": {}})
    state["history"].append({"role": "user", "content": text})
    log_event({"chat_id": chat_id, "role": "user", "content": text})

    try:
        reply = await asyncio.wait_for(
            run_turn(chat_id, state["history"], state["ns"]), timeout=OVERALL_TURN_TIMEOUT
        )
    except asyncio.TimeoutError:
        reply = json.dumps({"error": "turn_timeout", "log_url": ""})
        log_event({"chat_id": chat_id, "role": "error", "content": "overall turn timeout"})
    except Exception as e:
        reply = json.dumps({"error": f"{type(e).__name__}: {e}"})
        log_event({"chat_id": chat_id, "role": "error", "content": str(e)})

    state["history"].append({"role": "assistant", "content": reply})
    state["history"][:] = state["history"][-MAX_HISTORY:]
    log_event({"chat_id": chat_id, "role": "assistant", "content": reply})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": reply})
            if r.status_code != 200:
                log_event({"chat_id": chat_id, "role": "error",
                           "content": f"sendMessage failed {r.status_code}: {r.text[:300]}"})
    except Exception as e:
        log_event({"chat_id": chat_id, "role": "error", "content": f"sendMessage exception: {e}"})
