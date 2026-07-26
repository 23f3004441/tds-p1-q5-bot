"""Append-only JSONL run log, pushed to a file in a public GitHub repo via
the Contents API. `log_url` is the raw.githubusercontent.com URL to that
file — public, wget-able, no GCP/billing/KYC involved.

Env vars:
    GITHUB_TOKEN         - personal access token with repo write access
    GITHUB_REPO          - "owner/repo", e.g. "yumnairfan14/tds-p1-q5-bot"
    GITHUB_LOG_PATH        (default "logs/run.jsonl")
    GITHUB_BRANCH           (default "main")
"""
import base64
import json
import os
import threading
import time

import httpx

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO")  # "owner/repo"
GITHUB_LOG_PATH = os.environ.get("GITHUB_LOG_PATH", "logs/run.jsonl")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

LOCAL_LOG = "/tmp/run.jsonl"
_lock = threading.Lock()
_sha = None  # cached blob sha for the log file, refreshed as needed


def _api_url() -> str:
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_LOG_PATH}"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def _fetch_sha(client: httpx.Client):
    global _sha
    r = client.get(_api_url(), headers=_headers(), params={"ref": GITHUB_BRANCH})
    if r.status_code == 200:
        _sha = r.json().get("sha")
    else:
        _sha = None  # file doesn't exist yet, or repo not reachable


def _push(client: httpx.Client, content_bytes: bytes):
    global _sha
    payload = {
        "message": "update run log",
        "content": base64.b64encode(content_bytes).decode(),
        "branch": GITHUB_BRANCH,
    }
    if _sha:
        payload["sha"] = _sha
    r = client.put(_api_url(), headers=_headers(), json=payload)
    if r.status_code in (200, 201):
        _sha = r.json()["content"]["sha"]
    elif r.status_code in (409, 422):
        # sha out of date — refresh and retry once
        _fetch_sha(client)
        payload["sha"] = _sha
        r2 = client.put(_api_url(), headers=_headers(), json=payload)
        if r2.status_code in (200, 201):
            _sha = r2.json()["content"]["sha"]
        else:
            print(f"GitHub log push failed (retry): {r2.status_code} {r2.text}")
    else:
        print(f"GitHub log push failed: {r.status_code} {r.text}")


def log_event(event: dict) -> None:
    event = {"ts": time.time(), **event}
    line = json.dumps(event, default=str)
    with _lock:
        with open(LOCAL_LOG, "a") as f:
            f.write(line + "\n")
        if GITHUB_TOKEN and GITHUB_REPO:
            try:
                with httpx.Client(timeout=20) as client:
                    if _sha is None:
                        _fetch_sha(client)
                    with open(LOCAL_LOG, "rb") as f:
                        _push(client, f.read())
            except Exception as e:
                print(f"GitHub log push error: {e}")


def public_log_url() -> str | None:
    if GITHUB_REPO:
        return f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_LOG_PATH}"
    return None
