"""ReAct-style agent: calls AIPipe (OpenAI-compatible) chat completions with
two tools — python_exec (fetch/compute) and final_answer (submit). Every
tool call and model step is logged as JSONL via gcs_logger.log_event.
"""
import asyncio
import concurrent.futures
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

MAX_STEPS = 6
PYTHON_EXEC_TIMEOUT = 30  # seconds — hard wall-clock cap per python_exec call

# Dedicated thread pool so a hung/slow python_exec call (e.g. a stalled download)
# never blocks the single-worker asyncio event loop that also has to accept new
# incoming Telegram webhook requests.
_EXEC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)
MAX_TOOL_OUTPUT = 4000

SYSTEM_PROMPT = """You are a data-analyst agent replying inside a Telegram chat, as part of an \
automated grading exchange. Each incoming user message is either:

1. Context / data for a task (may include inline data, a public dataset reference such as \
MOSPI, or setup for a later step). Reply with a SHORT plain-text acknowledgement (a few words), \
and do NOT output JSON yet.

2. A final request that explicitly says something like "Reply with ONLY ... JSON object ...". \
When you see this, you MUST actually solve the task first (use the python_exec tool to fetch \
data with `requests`, load/clean it with `pandas`/`numpy`, and compute the real answer), then \
call the final_answer tool with the computed value.

NEVER invent, simulate, or guess data values to answer with. If a source is a PDF, download it \
with `requests.get(url).content` then extract its text/tables with `pdfplumber` \
(`pdfplumber.open(io.BytesIO(resp.content))`, then `.pages[i].extract_text()` or `.extract_table()`). \
If a source is an Excel/CSV file, use `pandas.read_excel(io.BytesIO(resp.content))` or \
`pandas.read_csv(io.BytesIO(resp.content))`. If a source is an HTML page, use \
`pandas.read_html(resp.text)` or parse with `requests` + regex/string search on `resp.text`. \
Try multiple approaches and multiple candidate URLs before giving up. Only if every real avenue \
is genuinely exhausted (e.g. you truly cannot reach the source after several attempts) should you \
fall back to your best general knowledge — and even then, do not invent fabricated numeric tables; \
give your single best real-world estimate instead of a made-up dataset.

Rules for final_answer:
- Call it with `value` = a JSON-encoded STRING of ONLY the answer payload requested inside the \
message (e.g. '{"state": "Assam"}' or '[1, 2, 3]' or '"42"'). Do NOT include an outer \
{"answer":..., "log_url":...} wrapper yourself, even if the user's message shows that wrapper as \
part of the example shape — that wrapper is added automatically by the system. If the message says \
'Reply with ONLY {"answer": {"state": "X"}, "log_url": "..."}', your `value` should be just \
'{"state": "X"}' — nothing about "answer" or "log_url" belongs in what you pass.
- The value must match the exact shape/keys the message asked for, nothing extra.
- Only call final_answer once, only when the current message is the "reply with ONLY" request.

The python_exec tool runs Python with requests/pandas/numpy/pdfplumber/json/re/math/statistics/io \
pre-imported, persistent across calls in this conversation (reuse variables). Use print() to see \
output. Always prefer real computation over assumptions — fetch and inspect data before answering.
"""

PYTHON_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Execute Python code to fetch data (requests), extract PDFs (pdfplumber) or "
            "Excel/CSV (pandas), analyze it (pandas/numpy), and compute results. State "
            "persists across calls within this conversation."
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


class _RequestsWithDefaultTimeout:
    """Thin wrapper around `requests` that defaults timeout=20s on get/post/request
    calls if the model doesn't specify one — a hard backstop against indefinite hangs."""
    _DEFAULT_TIMEOUT = 20

    def __init__(self, real_requests):
        self._real = real_requests

    def __getattr__(self, name):
        return getattr(self._real, name)

    def get(self, *args, **kwargs):
        kwargs.setdefault("timeout", self._DEFAULT_TIMEOUT)
        return self._real.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        kwargs.setdefault("timeout", self._DEFAULT_TIMEOUT)
        return self._real.post(*args, **kwargs)

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", self._DEFAULT_TIMEOUT)
        return self._real.request(*args, **kwargs)


def exec_python(code: str, ns: dict) -> str:
    for name in ("requests", "pandas", "numpy", "json", "re", "math", "statistics", "io", "pdfplumber"):
        if name not in ns:
            try:
                mod = __import__(name)
                if name == "requests":
                    mod = _RequestsWithDefaultTimeout(mod)
                ns[name] = mod
            except ImportError:
                pass

    buf = io.StringIO()

    def _print(*args, **kwargs):
        kwargs.pop("file", None)
        print(*args, file=buf, **kwargs)

    ns["print"] = _print  # thread-safe: doesn't touch the process-global sys.stdout

    try:
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


def _unwrap_double_wrap(value):
    """If the model ignored instructions and wrapped its own {"answer":..., "log_url":...}
    around the payload, unwrap it so we don't nest it again."""
    if isinstance(value, dict) and set(value.keys()) == {"answer", "log_url"}:
        return value["answer"]
    return value


async def call_aipipe(messages):
    async with httpx.AsyncClient(timeout=45) as client:
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
            value = _unwrap_double_wrap(_coerce_value(text))
            return json.dumps({"answer": value, "log_url": public_log_url() or ""})

        messages.append(choice)
        for tc in tool_calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if fn == "final_answer":
                value = _unwrap_double_wrap(_coerce_value(str(args.get("value", ""))))
                log_event({"chat_id": chat_id, "role": "final_answer", "value": value})
                return json.dumps({"answer": value, "log_url": public_log_url() or ""})

            elif fn == "python_exec":
                code = args.get("code", "")
                loop = asyncio.get_running_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(_EXEC_POOL, exec_python, code, ns),
                        timeout=PYTHON_EXEC_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    result = (
                        f"TIMEOUT: code did not finish within {PYTHON_EXEC_TIMEOUT}s "
                        "(likely a slow/hung network request — try a smaller request, "
                        "add timeout=15 to requests calls, or try a different source)."
                    )
                log_event({"chat_id": chat_id, "role": "tool", "tool": "python_exec", "code": code, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result or "(no output)"})

            else:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": f"unknown tool {fn}"})

    if is_final_turn:
        return json.dumps({"answer": {"error": "max_steps_exceeded"}, "log_url": public_log_url() or ""})
    return "Still working on it."
