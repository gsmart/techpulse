import os
from .config import Settings, load_feeds_from_yaml, now_ist
from .feeds import fetch_items
from .summarize import summarize_item
from .writer_docx import write_docx
from .drive import ensure_hierarchy, upload_docx
from .storage import connect, is_seen, mark_seen, prune_older_than
from .notify import tg_send_text, tg_send_file, discord_send_text, discord_send_embed, discord_send_file


def _build_message(fname: str, web_link: str, items: list[dict]) -> str:
    # compact TL;DR post
    lines = [f"*Tech Pulse — {now_ist().strftime('%Y-%m-%d %H:00 IST')}*",
             f"[Open in Drive]({web_link})", ""]
    for i, it in enumerate(items[:6], 1):  # preview top 6
        lines.append(f"{i}. *{it['title']}* — {it['source']}")
    if len(items) > 6:
        lines.append(f"...and {len(items)-6} more.")
    return "\n".join(lines)

def run_once(max_items: int = 12):
    cfg = Settings()
    feeds = load_feeds_from_yaml(cfg.feeds_file)

    conn = connect(cfg.news_db_path)
    prune_older_than(conn, days=30)

    collected = fetch_items(feeds, window_minutes=180)
    fresh = []
    for it in collected:
        if not it.get("link") or is_seen(conn, it["link"]):
            continue
        fresh.append(it)
        if len(fresh) >= max_items:
            break
    if not fresh:
        print("[i] No new items this run.")
        return []

    for it in fresh:
        it["summary"] = summarize_item(it)

    for it in fresh:
        print(f"- {it['title']} ({it['source']})")
        print(f"  {it['summary']}")
        print(f"  {it['link']}\n")

    outdir = "./tmp_docs"
    os.makedirs(outdir, exist_ok=True)
    fname = f"Tech Pulse — {now_ist().strftime('%Y-%m-%d %H 00 IST')}.docx"
    local_path = os.path.join(outdir, fname)
    write_docx(fresh, local_path)
    print(f"[✓] Saved DOCX: {local_path}")

    # Upload to Drive (optional but recommended)
    web_link = None
    try:
        folder_id = ensure_hierarchy(cfg)
        uploaded = upload_docx(cfg, folder_id, local_path, fname)
        web_link = uploaded.get("webViewLink")
        print(f"[✓] Uploaded: {web_link}")
    finally:
        for it in fresh:
            mark_seen(conn, it["link"])

    # ---------- Notifications ----------
    msg = _build_message(fname, uploaded.get("webViewLink", "") if 'uploaded' in locals() else "", fresh)

    # Telegram
    tg_ok_text = tg_send_text(msg, markdown=False)
    # tg_ok_file = tg_send_file(local_path, caption=fname, markdown=False)
    tg_ok_file = tg_send_file(local_path, caption=fname)
    print(f"[TG] text={tg_ok_text} file={tg_ok_file}")

    # Discord
    dc_ok_text = discord_send_text(msg.replace("*", ""))      # strip Markdown *bold* for Discord text
    if web_link:
        discord_send_embed(title=fname, url=web_link, description="Tech news TL;DR")
    discord_send_file(local_path, "Tech Pulse docx")

    return fresh