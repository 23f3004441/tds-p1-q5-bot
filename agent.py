"""ReAct-style agent: calls AIPipe (OpenAI-compatible) chat completions with
two tools — python_exec (fetch/compute) and final_answer (submit). Every
tool call and model step is logged as JSONL via gcs_logger.log_event.
"""
import contextlib
import io
import json
import os
import re
import traceback

import httpx

from github_logger import log_event, public_log_url

AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "gpt-4o-mini")

MAX_STEPS = 8
MAX_TOOL_OUTPUT = 4000

SYSTEM_PROMPT = """You are a data-analyst agent replying inside a Telegram chat, as part of an \
automated grading exchange. Each incoming user message is either:

1. Context / data for a task (may include inline data, a public dataset reference such as \
MOSPI, or setup for a later step). Reply with a SHORT plain-text acknowledgement (a few words), \
and do NOT output JSON yet.

2. A final request that explicitly says something like "Reply with ONLY ... JSON object ...". \
When you see this, you MUST actually solve the task first (use the python_exec tool to fetch \
data with `requests`, load/clean it with `pandas`/`numpy`, and compute the real answer — never \
guess or fabricate numbers), then call the final_answer tool with the computed value.

Rules for final_answer:
- Call it with `value` = a JSON-encoded STRING of just the answer payload requested inside the \
message (e.g. '{"state": "Assam"}' or '[1, 2, 3]' or '"42"'). Do NOT include an outer \
{"answer":..., "log_url":...} wrapper yourself — that is added automatically by the system.
- The value must match the exact shape/keys the message asked for, nothing extra.
- Only call final_answer once, only when the current message is the "reply with ONLY" request.

The python_exec tool runs Python with requests/pandas/numpy/json/re/math/statistics pre-imported, \
persistent across calls in this conversation (reuse variables). Use print() to see output. \
Prefer real computation over assumptions — fetch and inspect data before answering.
"""

PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Execute Python code to fetch data (requests), analyze it (pandas/numpy), and "
            "compute results. State persists across calls within this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"],
        },
    },
}

FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the final computed answer for the current 'reply with ONLY JSON' request.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "JSON-encoded string of just the answer payload, e.g. '{\"state\": \"Assam\"}'",
                }
            },
            "required": ["value"],
        },
    },
}

TOOLS = [PYTHON_EXEC_TOOL, FINAL_ANSWER_TOOL]


def exec_python(code: str, ns: dict) -> str:
    for name in ("requests", "pandas", "numpy", "json", "re", "math", "statistics"):
        if name not in ns:
            try:
                ns[name] = __import__(name)
            except ImportError:
                pass
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, ns)
    except Exception:
        buf.write("\n" + traceback.format_exc())
    out = buf.getvalue()
    return out[-MAX_TOOL_OUTPUT:] if len(out) > MAX_TOOL_OUTPUT else out


def wants_final_json(text: str) -> bool:
    t = text.lower()
    return "reply with only" in t or "reply with exactly" in t or ("only" in t and "json" in t)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coerce_value(raw: str):
    """Best-effort parse of a value string into JSON; falls back to a JSON substring match."""
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return cleaned  # give up, send raw string


async def call_aipipe(messages):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{AIPIPE_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
            json={
                "model": AIPIPE_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        )
        r.raise_for_status()
        return r.json()


async def run_turn(chat_id, history, ns, max_steps: int = MAX_STEPS) -> str:
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    is_final_turn = wants_final_json(last_user)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    for _ in range(max_steps):
        data = await call_aipipe(messages)
        choice = data["choices"][0]["message"]
        tool_calls = choice.get("tool_calls")

        if not tool_calls:
            text = (choice.get("content") or "").strip()
            if not is_final_turn:
                return text[:300] or "On it."
            # model answered without calling final_answer — salvage what we can
            value = _coerce_value(text)
            return json.dumps({"answer": value, "log_url": public_log_url() or ""})

        messages.append(choice)
        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if fn == "final_answer":
                value = _coerce_value(str(args.get("value", "")))
                log_event({"chat_id": chat_id, "role": "final_answer", "value": value})
                return json.dumps({"answer": value, "log_url": public_log_url() or ""})

            elif fn == "python_exec":
                code = args.get("code", "")
                result = exec_python(code, ns)
                log_event({"chat_id": chat_id, "role": "tool", "tool": "python_exec", "code": code, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result or "(no output)"})

            else:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"unknown tool {fn}"})

    if is_final_turn:
        return json.dumps({"answer": {"error": "max_steps_exceeded"}, "log_url": public_log_url() or ""})
    return "Still working on it."
