import os, yaml
from dataclasses import dataclass
from datetime import datetime
from dateutil import tz
from pathlib import Path

IST = tz.gettz("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <repo root>

@dataclass
class Settings:
    feeds_file: str = os.getenv("FEEDS_FILE", str(PROJECT_ROOT / "src" / "configs" / "feeds.yaml"))
    google_credentials_path: str = os.getenv("GOOGLE_CREDENTIALS_PATH", str(PROJECT_ROOT / ".secrets" / "credentials.json"))
    google_token_path: str = os.getenv("GOOGLE_TOKEN_PATH", str(PROJECT_ROOT / ".secrets" / "token.json"))
    top_folder_name: str = os.getenv("TOP_FOLDER_NAME", "Daily Tech News")
    root_drive_parent_id: str | None = os.getenv("GOOGLE_DRIVE_FOLDER_ID") or None
    news_db_path: str = os.getenv("NEWS_DB_PATH", str(PROJECT_ROOT / ".secrets" / "news_seen.sqlite"))

def now_ist() -> datetime:
    return datetime.now(IST)

def load_feeds_from_yaml(path: str) -> list[str]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    feeds = data.get("feeds", [])
    return [u.strip() for u in feeds if isinstance(u, str) and u.strip()]
