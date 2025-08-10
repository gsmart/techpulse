import html
import re, os, requests

def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(s.split())

def _clean_entities(s: str) -> str:
    # Decode HTML entities
    s = html.unescape(s or "")
    # Remove trailing feedburner-style ellipsis markers
    s = re.sub(r"\[&#8230;]$", "", s.strip())
    return s

def _extractive(text: str, max_chars: int = 480) -> str:
    text = _strip_html(text or "")
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "..."

def _openai_summarize(title: str, source: str, text: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Summarize the news item in 2–4 concise, neutral sentences for busy engineers. "
            "Include concrete details (features, versions, dates, benchmarks) when present. Avoid hype.\n\n"
            f"Title: {title}\nSource: {source}\nText:\n{text[:5000]}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def _ollama_summarize(title: str, source: str, text: str) -> str | None:
    host = os.getenv("OLLAMA_HOST")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    if not host:
        return None
    try:
        r = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Summarize the news item in 2–4 concise, neutral sentences with concrete facts.\n\n"
                        f"Title: {title}\nSource: {source}\nText:\n{text[:4000]}"
                    ),
                }],
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None

def summarize_item(item: dict) -> str:
    # Pick best available text
    seed = (item.get("content_seed") or "") + "\n" + (item.get("summary_seed") or "")
    seed = _clean_entities(seed.strip())  # <-- decode & remove artifacts
    title = _clean_entities(item.get("title", ""))
    source = _clean_entities(item.get("source", ""))

    # Try OpenAI → Ollama → extractive
    s = _openai_summarize(title, source, seed)
    if not s:
        s = _ollama_summarize(title, source, seed)
    if not s:
        s = _extractive(seed)
    return s
