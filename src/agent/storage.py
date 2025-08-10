import sqlite3, time, hashlib, urllib.parse
from pathlib import Path
from typing import Optional

# Strip common tracking params so ?utm_source=... doesn't create "new" URLs
TRACKING_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id","gclid","fbclid"}

def _normalize_url(url: str) -> str:
    try:
        p = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        q = [(k,v) for (k,v) in q if k.lower() not in TRACKING_PARAMS]
        new_query = urllib.parse.urlencode(q)
        return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, new_query, p.fragment))
    except Exception:
        return url

def _hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen(
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            first_seen_ts INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_url ON seen(url)")
    conn.commit()
    return conn

def is_seen(conn, url: str) -> bool:
    url = _normalize_url(url)
    h = _hash(url)
    cur = conn.execute("SELECT 1 FROM seen WHERE url_hash=? LIMIT 1", (h,))
    return cur.fetchone() is not None

def mark_seen(conn, url: str):
    url = _normalize_url(url)
    h = _hash(url)
    conn.execute(
        "INSERT OR IGNORE INTO seen(url_hash, url, first_seen_ts) VALUES (?,?,?)",
        (h, url, int(time.time()))
    )
    conn.commit()

def prune_older_than(conn, days: int = 30):
    cutoff = int(time.time()) - days*24*3600
    conn.execute("DELETE FROM seen WHERE first_seen_ts < ?", (cutoff,))
    conn.commit()
