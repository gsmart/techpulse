import os, requests, mimetypes
from pathlib import Path
from typing import Optional

TG_API = "https://api.telegram.org"

def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v else None

# ---------- Telegram ----------
def tg_send_text_old(text: str, parse_mode: str = "Markdown") -> bool:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    # Telegram max message length ≈ 4096 chars
    text = text[:4000]
    url = f"{TG_API}/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode}, timeout=30)
    return r.ok

def tg_send_text(text, markdown=False):
    import os, requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    payload = {"chat_id": chat, "text": text}
    if markdown:
        payload["parse_mode"] = "Markdown"
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json=payload,
        timeout=15
    )
    return r.ok

def tg_send_file(file_path: str, caption: str = "", parse_mode: str = "Markdown") -> bool:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"{TG_API}/bot{token}/sendDocument"
    p = Path(file_path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    with open(p, "rb") as f:
        files = {"document": (p.name, f, mime)}
        data = {"chat_id": chat_id, "caption": caption[:990], "parse_mode": parse_mode}
        r = requests.post(url, data=data, files=files, timeout=120)
    return r.ok

# ---------- Discord ----------
def discord_send_text(content: str) -> bool:
    wh = _env("DISCORD_WEBHOOK_URL")
    if not wh:
        return False
    # Discord limit ≈ 2000 chars
    payload = {"content": content[:1990]}
    r = requests.post(wh, json=payload, timeout=30)
    return r.ok

def discord_send_embed(title: str, url: str, description: str = "") -> bool:
    wh = _env("DISCORD_WEBHOOK_URL")
    if not wh:
        return False
    embed = {
        "title": title[:256],
        "url": url,
        "description": description[:3900],
    }
    payload = {"embeds": [embed]}
    r = requests.post(wh, json=payload, timeout=30)
    return r.ok

def discord_send_file(file_path: str, message: str = "") -> bool:
    wh = _env("DISCORD_WEBHOOK_URL")
    if not wh:
        return False
    p = Path(file_path)
    with open(p, "rb") as f:
        files = {"file": (p.name, f)}
        data = {"content": message[:1990]} if message else {}
        r = requests.post(wh, data=data, files=files, timeout=120)
    return r.ok
