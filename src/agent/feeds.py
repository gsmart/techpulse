from typing import List, Dict, Any, Optional
from datetime import datetime
from dateutil import parser as dateparser, tz
import feedparser
from .config import IST

def _to_ist(dt) -> Optional[datetime]:
    if not dt: return None
    if dt.tzinfo is None: return dt.replace(tzinfo=tz.UTC).astimezone(IST)
    return dt.astimezone(IST)

def fetch_items(feed_urls: List[str], window_minutes: int) -> List[Dict[str, Any]]:
    items = []
    for url in feed_urls:
        d = feedparser.parse(url)
        for e in d.entries:
            link = e.get("link") or ""
            if not link: continue
            published = None
            for k in ("published", "updated", "created"):
                if e.get(k):
                    try:
                        published = _to_ist(dateparser.parse(e[k])); break
                    except Exception: pass
            items.append({
                "title": (e.get("title") or "").strip(),
                "link": link,
                "source": d.feed.get("title", url),
                "published": published,
                "summary_seed": (e.get("summary") or e.get("description") or ""),
            })
    items.sort(key=lambda x: x["published"] or datetime(2000,1,1,tzinfo=IST), reverse=True)
    return items
