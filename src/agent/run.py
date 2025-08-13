# src/agent/run.py
import os
from .config import Settings, load_feeds_from_yaml, now_ist
from .feeds import fetch_items
from .summarize import _hf_summarize_many, summarize_item, _clean_entities, _strip_html, _extractive
from .writer_docx import write_docx
from .drive import ensure_hierarchy, upload_docx
from .storage import connect, is_seen, mark_seen, prune_older_than
from .notify import tg_send_text, tg_send_file
import time

def _build_message(fname: str, web_link: str, items: list[dict]) -> str:
    lines = [
        f"Tech Pulse — {now_ist().strftime('%Y-%m-%d %H:00 IST')}",
        f"Drive: {web_link}" if web_link else "Drive: (upload skipped)"
    ]
    for i, it in enumerate(items[:6], 1):
        lines.append(f"{i}. {it['title']} — {it['source']}")
    if len(items) > 6:
        lines.append(f"...and {len(items)-6} more.")
    return "\n".join(lines)

def run_once(max_items: int = 12, window_minutes: int = 180):
    cfg = Settings()
    feeds = load_feeds_from_yaml(cfg.feeds_file)

    conn = connect(cfg.news_db_path)
    # Keep DB small; use env DEDUP_DAYS if present
    days = int(os.getenv("DEDUP_DAYS", "30"))
    prune_older_than(conn, days=days)

    # fetch + de-dup
    t0 = time.perf_counter()
    collected = fetch_items(feeds, window_minutes=window_minutes)
    t_fetch = time.perf_counter()
    fresh = []
    for it in collected:
        if not it.get("link") or is_seen(conn, it["link"]):
            continue
        fresh.append(it)
        if len(fresh) >= max_items:
            break

    if not fresh:
        print("[i] No new items this run.")
        if os.getenv("NOTIFY_TELEGRAM") == "1":
            tg_send_text(f"No new items at {now_ist().strftime('%Y-%m-%d %H:%M IST')}")
        return []

    # summarize
    # for it in fresh:
    #     it["summary"] = summarize_item(it)
    # summarize (batched for HF)
    # summarize (batched for HF; falls back to item-wise summarize_item for other backends)
    from .summarize import _hf_summarize_many, summarize_item, _clean_entities, _strip_html, _extractive

    backend = (os.getenv("SUMMARY_BACKEND", "") or "").lower().strip()
    if backend == "hf":
        # Build cleaned seeds once (match summarize_item behavior)
        seeds = []
        for it in fresh:
            seed = (it.get("content_seed") or "") + "\n" + (it.get("summary_seed") or "")
            seed = _clean_entities(seed.strip())
            title = _clean_entities(it.get("title", ""))
            if len(seed) < 40:
                seed = (seed + " " + title).strip()
            seeds.append(_strip_html(seed))

        # Gate abstractive by length: too short or too long → extractive
        min_chars = int(os.getenv("HF_GATE_ABS_MIN_CHARS", "160"))  # NEW
        max_chars = int(os.getenv("HF_GATE_ABS_MAX_CHARS", "1200"))  # keep/adjust

        abstractive_idx, abstractive_in = [], []
        summaries = [""] * len(seeds)

        for i, s in enumerate(seeds):
            L = len(s)
            if min_chars <= L <= max_chars:
                abstractive_idx.append(i)
                abstractive_in.append(s)
            else:
                summaries[i] = _extractive(s)

        # One HF call for all remaining items (robust function already falls back if needed)
        if abstractive_in:
            outs = _hf_summarize_many(abstractive_in)
            if not outs or len(outs) != len(abstractive_in):
                # HF failed or size mismatch → extractive for those
                for i in abstractive_idx:
                    summaries[i] = _extractive(seeds[i])
            else:
                for j, i in enumerate(abstractive_idx):
                    summaries[i] = outs[j]

        for i, it in enumerate(fresh):
            it["summary"] = summaries[i]
    else:
        # keep existing path for OpenAI/Ollama/Auto
        for it in fresh:
            it["summary"] = summarize_item(it)

    t_sum = time.perf_counter()

    # console preview
    for it in fresh:
        print(f"- {it['title']} ({it['source']})\n  {it['summary']}\n  {it['link']}\n")

    # save docx
    outdir = "./tmp_docs"
    os.makedirs(outdir, exist_ok=True)
    fname = f"Tech Pulse — {now_ist().strftime('%Y-%m-%d %H 00 IST')}.docx"
    local_path = os.path.join(outdir, fname)
    write_docx(fresh, local_path)
    t_docx = time.perf_counter()
    print(f"[✓] Saved DOCX: {local_path}")

    # upload to drive
    web_link = ""
    try:
        folder_id = ensure_hierarchy(cfg)
        uploaded = upload_docx(cfg, folder_id, local_path, fname)
        t_upload = time.perf_counter()
        web_link = uploaded.get("webViewLink", "")
        print(f"[✓] Uploaded: {web_link}")
        print(
            f"[TIMING] fetch={t_fetch - t0:.2f}s sum={t_sum - t_fetch:.2f}s docx={t_docx - t_sum:.2f}s upload={t_upload - t_docx:.2f}s total={t_upload - t0:.2f}s")

    finally:
        # mark seen after generating the doc (so retries don't lose items)
        for it in fresh:
            mark_seen(conn, it["link"])

    # telegram (optional)
    if os.getenv("NOTIFY_TELEGRAM") == "1":
        msg = _build_message(fname, web_link, fresh)
        ok1 = tg_send_text(msg)            # plain text (no markdown)
        ok2 = tg_send_file(local_path, caption=fname)
        print(f"[TG] text={ok1} file={ok2}")

    return fresh
