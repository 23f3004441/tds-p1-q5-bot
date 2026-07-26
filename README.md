# Data Analyst Telegram Bot — TDS P1 Q5

FastAPI webhook bot. On each Telegram message: logs it, runs it through an
AIPipe-backed agent (with a `python_exec` tool for fetching/computing real
answers), replies via Telegram, and pushes a JSONL run log to a file in a
public GitHub repo — its `raw.githubusercontent.com` URL is used as `log_url`.
No GCP involved.

## 1. Create the bot (physical step — do this yourself)
- Telegram → `@BotFather` → `/newbot` → pick a username ending in `bot`. Done: `tdsp1bot`
- Copy the token it gives you (`BOT_TOKEN`). Done

## 2. AIPipe token
- https://aipipe.org/login → copy token → `AIPIPE_TOKEN`. Done

## 3. GitHub personal access token (for the log)
- GitHub → Settings → Developer settings → Personal access tokens →
  Fine-grained tokens → Generate new token.
- Scope it to the repo you'll create in step 4, with **Contents: Read and write** permission.
- Copy it → `GITHUB_TOKEN`.

## 4. Create a public GitHub repo
- Create a new public repo (e.g. `tds-p1-q5-bot`).
- Upload `main.py`, `agent.py`, `github_logger.py`, `requirements.txt`, `README.md`
  via "Add file → Upload files".
- `GITHUB_REPO` = `<your-username>/tds-p1-q5-bot`.
- The bot creates `logs/run.jsonl` in this same repo automatically on first message.

## 5. Deploy on Render (Web Service, free tier)
- New → Web Service → connect the repo.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `BOT_TOKEN`
  - `AIPIPE_TOKEN`
  - `GITHUB_TOKEN`
  - `GITHUB_REPO`
  - optional: `AIPIPE_MODEL` (default `gpt-4o-mini`), `AIPIPE_BASE_URL`,
    `GITHUB_LOG_PATH` (default `logs/run.jsonl`), `GITHUB_BRANCH` (default `main`)

## 6. Point Telegram at your deployed URL
```
curl -F "url=https://<your-render-app>.onrender.com/webhook" \
     https://api.telegram.org/bot<BOT_TOKEN>/setWebhook
```
Verify: `curl https://api.telegram.org/bot<BOT_TOKEN>/getWebhookInfo`

## 7. Keep it warm
Add `https://<your-render-app>.onrender.com/` to UptimeRobot (same pattern
you used for GA5) so Render doesn't spin the service down before/during grading.

## 8. Sanity test
Message your bot on Telegram:
> Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY this JSON object and nothing else: {"answer": {"state": "<state name>"}, "log_url": "<url>"}

It should reply with exactly one JSON object, and
`https://raw.githubusercontent.com/<owner>/<repo>/main/logs/run.jsonl` should
be `wget`-able and growing after each message.

## 9. Register
In the portal, submit: `<your GitHub repo URL>, <your bot username>`
