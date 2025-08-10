# techpulse-agent

> **TL;DR**
> Pull tech news from RSS → generate short, neutral **AI summaries** → save a **.docx** → upload to Google Drive in `Daily Tech News → DDMMYYYY → 1..24`.
> De-duplicates links with SQLite. Works with **OpenAI**, **Ollama** (local), or **no LLM** (extractive fallback).

---

## Features

* **Agentic summaries**: OpenAI or Ollama; falls back to clean extractive summary.
* **Clean input**: HTML + entities decoded; feed artifacts stripped.
* **Structured Drive path**: `Daily Tech News / DDMMYYYY / 1..24`.
* **No duplicates**: URL-normalized, SQLite-backed de-dup.
* **Config-driven**: Feeds in YAML; secrets via `.env`.
* **Headless**: Works great on cron.

---

## Prerequisites

* Python **3.11+**
* Optional: **Conda** (recommended) or `venv`
* For Drive upload (choose one):

  * **Service Account** (recommended for unattended): share a Drive folder with the service account email.
  * **OAuth Desktop** (dev only): test-user consent; tokens in `.secrets/`.

---

## Quickstart (TL;DR)

```bash
# 1) Create env
conda create -n techpulse python=3.11 -y
conda activate techpulse

# 2) Install deps
pip install -r requirements.txt

# 3) Feeds
mkdir -p src/configs
cat > src/configs/feeds.yaml << 'EOF'
feeds:
  - https://www.theverge.com/rss/index.xml
  - https://techcrunch.com/tag/artificial-intelligence/feed/
  - https://www.technologyreview.com/feed/
  - https://aws.amazon.com/blogs/aws/feed/
  - https://cloud.google.com/blog/rss/
EOF

# 4) Secrets
mkdir -p .secrets
# Put Google service account JSON here:
# .secrets/service_account.json

# 5) .env
cat > .env << 'EOF'
FEEDS_FILE=src/configs/feeds.yaml
NEWS_DB_PATH=.secrets/news_seen.sqlite

# Drive (Service Account preferred)
GOOGLE_SERVICE_ACCOUNT_JSON=.secrets/service_account.json
GOOGLE_DRIVE_FOLDER_ID=        # parent folder id you shared with the service account (optional)
TOP_FOLDER_NAME=Daily Tech News

# AI Summaries (pick one or neither)
# OPENAI_API_KEY=sk-...
# AI_MODEL=gpt-4o-mini
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.1
EOF

# 6) Run
python cli.py
```

On first run you’ll see:

* Console summaries
* A DOCX created in `tmp_docs/`
* Upload to Drive under `Daily Tech News / DDMMYYYY / 1..24`

---

## Project Structure

```
techpulse-agent/
├─ cli.py
├─ requirements.txt
├─ .env                       # not committed
├─ .secrets/                  # not committed (service_account.json, token.json, sqlite)
│  ├─ service_account.json
│  ├─ token.json
│  └─ news_seen.sqlite
├─ src/
│  ├─ agent/
│  │  ├─ run.py               # fetch → dedupe → summarize → docx → upload
│  │  ├─ feeds.py             # RSS/Atom parsing
│  │  ├─ summarize.py         # OpenAI/Ollama/extractive
│  │  ├─ writer_docx.py       # .docx builder
│  │  ├─ drive.py             # Google Drive (Service Account)
│  │  └─ storage.py           # SQLite de-dup
│  └─ configs/
│     └─ feeds.yaml
└─ tmp_docs/
```

**.gitignore** should include:

```
.env
.secrets/
credentials/
*.pem
*.key
token.json
```

---

## Configuration

Edit `.env` (or set as real env vars):

| Var                           | Purpose                             | Example                         |
| ----------------------------- | ----------------------------------- | ------------------------------- |
| `FEEDS_FILE`                  | Path to YAML feed list              | `src/configs/feeds.yaml`        |
| `NEWS_DB_PATH`                | SQLite file for de-dup              | `.secrets/news_seen.sqlite`     |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Service account JSON                | `.secrets/service_account.json` |
| `GOOGLE_DRIVE_FOLDER_ID`      | Parent folder id (optional)         | `1AbC...xyz`                    |
| `TOP_FOLDER_NAME`             | Top folder name in Drive            | `Daily Tech News`               |
| `OPENAI_API_KEY`              | Use OpenAI for summaries (optional) | `sk-...`                        |
| `AI_MODEL`                    | OpenAI model                        | `gpt-4o-mini`                   |
| `OLLAMA_HOST`                 | Local LLM                           | `http://localhost:11434`        |
| `OLLAMA_MODEL`                | Ollama model                        | `llama3.1`                      |

**YAML feeds** (`src/configs/feeds.yaml`):

```yaml
feeds:
  - https://www.theverge.com/rss/index.xml
  - https://techcrunch.com/tag/artificial-intelligence/feed/
```

---

## Running

```bash
conda activate techpulse
python cli.py
```

* Each run:

  * Fetches fresh items
  * Skips URLs seen before (even with UTM tracking variants)
  * Produces 2–4 sentence summaries (OpenAI → Ollama → extractive)
  * Writes a single `.docx`
  * Uploads it to Google Drive into `Daily Tech News / DDMMYYYY / 1..24`

---

## Scheduling (cron)

IST example: run at 5 minutes past every hour.

```bash
crontab -e
# adjust paths to your machine
5 * * * * cd /path/to/techpulse-agent && /Users/you/miniconda3/envs/techpulse/bin/python cli.py >> agent.log 2>&1
```

---

## Troubleshooting

* **HTML codes like `[&#8230;]` in text**
  We decode entities and strip feedburner ellipses automatically in `summarize.py`. If you still see artifacts, ensure you’re running the latest code.

* **OAuth app blocked**
  Use **Service Account** flow (recommended). Share your Drive parent folder with the service account email, then set `GOOGLE_SERVICE_ACCOUNT_JSON` (and optionally `GOOGLE_DRIVE_FOLDER_ID`).

* **Feeds file not found**
  Set `FEEDS_FILE` correctly or keep it at `src/configs/feeds.yaml`.

* **RequestsDependencyWarning (chardet/charset\_normalizer)**
  Run: `pip install charset-normalizer` (or `chardet`).

* **No new items**
  The de-dup DB remembers URLs. Clear old entries (`.secrets/news_seen.sqlite`) or adjust pruning in `storage.py`.

---

## Roadmap

* Topic tags per feed (e.g., ai/devops/security)
* Native Google Docs output
* Summarization knobs (length/tone) per feed
* Slack/Telegram delivery hooks

---

## License

MIT. Use it, tweak it, ship it.
