# src/agent/summarize.py

import html
import os
import re
import requests
from typing import Optional, List, Tuple
import time
import math
import json
import requests as _req


# ---------- helpers / config ----------

def _endsentence(s: str) -> str:
    s = s.strip()
    return s if s.endswith(('.', '!', '?')) else (s + '.' if s else s)

def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name, "").strip().lower())
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default

def _backend_enabled(name: str, default: bool) -> bool:
    # name in {OPENAI, OLLAMA, HF, TEXTRANK}
    if _env_bool(f"DISABLE_{name}", False):
        return False
    if os.getenv(f"ENABLE_{name}") is not None:
        return _env_bool(f"ENABLE_{name}")
    return default

print(
    "[SUM] config:",
    "backend=", (os.getenv("SUMMARY_BACKEND") or "").lower() or "auto",
    "enable_openai=", _backend_enabled("OPENAI", True),
    "enable_ollama=", _backend_enabled("OLLAMA", True),
    "enable_hf=", _backend_enabled("HF", _env_bool("USE_HF_SUMMARY", False)),
    "enable_textrank=", _backend_enabled("TEXTRANK", False),
)

# ---------- cleaning ----------

def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s or "", flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(s.split())

def _clean_entities(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\[&#8230;]$", "", s.strip())  # strip feedburner ellipsis
    return s

# ---------- simple extractive fallback ----------

def _extractive(text: str, max_chars: int = 480) -> str:
    text = _strip_html(text or "")
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "..."

# ---------- OpenAI ----------

def _openai_summarize(title: str, source: str, text: str) -> Optional[str]:
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
            f"Title: {title}\nSource: {source}\nText:\n{(text or '')[:5000]}"
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

# ---------- Ollama ----------

def _ollama_summarize(title: str, source: str, text: str) -> Optional[str]:
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
                        f"Title: {title}\nSource: {source}\nText:\n{(text or '')[:4000]}"
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

# ---------- HF Transformers (CPU / Apple MPS) ----------

_HF_PIPE = None
_HF_WARMED = False

def _hf_get_pipe():
    """Initialize a local summarization pipeline. Uses MPS if available (Apple Silicon), else CPU.
       - Respects HF_DEVICE={mps|cpu} and USE_CPU_ONLY
       - Safe dynamic-quant guard (CPU only)
       - One-time warmup (prevents first-item spike)
    """
    global _HF_PIPE, _HF_WARMED
    if _HF_PIPE is not None:
        return _HF_PIPE
    try:
        import os
        import torch
        from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

        # Tame CPU oversubscription (helps on Intel/macOS; harmless otherwise)
        try:
            n_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
            torch.set_num_threads(n_threads)
        except Exception:
            pass

        model_id = os.getenv("HF_SUMMARY_MODEL", "sshleifer/distilbart-cnn-6-6")

        # ---- device selection (no overrides later) ----
        force = (os.getenv("HF_DEVICE", "").strip().lower() or None)  # 'mps' | 'cpu' | None
        use_cpu_only = os.getenv("USE_CPU_ONLY", "0") == "1"
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if force == "mps":
            use_mps = True
            use_cpu_only = False
        elif force == "cpu":
            use_mps = False
            use_cpu_only = True
        else:
            use_mps = (not use_cpu_only) and mps_avail

        print(f"[HF] devpick mps_avail={mps_avail} USE_CPU_ONLY={use_cpu_only} HF_DEVICE={force or 'auto'}")

        # ---- load model/tokenizer ----
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if use_mps else None),
        )

        # ---- optional dynamic quantization (CPU only, guarded) ----
        if (not use_mps) and os.getenv("HF_DYN_QUANT", "0") == "1":
            try:
                engines = getattr(torch.backends.quantized, "supported_engines", [])
                if isinstance(engines, (list, tuple)) and "qnnpack" in engines:
                    torch.backends.quantized.engine = "qnnpack"
                if getattr(torch.backends.quantized, "engine", "none") != "none":
                    import torch.nn as nn
                    from torch.ao.quantization import quantize_dynamic
                    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                    print(f"[HF] dynamic quantization enabled: {torch.backends.quantized.engine}")
                else:
                    print("[HF] dynamic quantization skipped: no supported engine")
            except Exception as qe:
                print(f"[HF] dynamic quantization disabled: {qe}")

        # ---- move to device and build pipeline ----
        if use_mps:
            model.to("mps")
            device = 0     # pipeline uses 0 for GPU/MPS
        else:
            device = -1    # CPU

        _HF_PIPE = pipeline("summarization", model=model, tokenizer=tok, device=device)
        print(f"[HF] device={'mps' if use_mps else 'cpu'} model={model_id}")

        # ---- one-time warmup to avoid first-item latency spike ----
        if not _HF_WARMED:
            _ = _HF_PIPE("Hello.")
            _HF_WARMED = True

        return _HF_PIPE
    except Exception as e:
        import traceback
        print("[HF] init failed:", repr(e))
        traceback.print_exc()
        return None


def _chunk(text: str, max_chars: int = 2400) -> List[str]:
    """Split text to avoid model max length; sentence-aware-ish."""
    t = _strip_html(text or "")
    if len(t) <= max_chars:
        return [t]
    parts = re.split(r"(?<=[.!?])\s+", t)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > max_chars:
            if cur:
                chunks.append(cur.strip())
                cur = ""
        cur += (" " if cur else "") + p
    if cur:
        chunks.append(cur.strip())
    return chunks

def _finalize_summary(text: str) -> str:
    """Trim to a complete sentence and normalize spacing."""
    s = " ".join((text or "").split())
    # Take up to the last sentence terminator if present
    ends = list(re.finditer(r'[.?!](?:["\')\]]+)?', s))
    if ends:
        return s[:ends[-1].end()]
    return (s + ".") if s and s[-1] not in ".?!" else s

def _polish_summary(text: str) -> str:
    """Light cleanup for readability: fix spaces before punctuation, trim, cap, limit to 4 sentences."""
    s = " ".join((text or "").split())
    # remove stray spaces before punctuation: "word ." -> "word."
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    # fix spaces around parentheses/brackets
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    # ensure single space after sentence punctuation
    s = re.sub(r"([.!?])([^\s\W])", r"\1 \2", s)

    # sentence cap + limit to 4 sentences
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p.strip()]
    parts = parts[:4]  # keep at most 4
    s = " ".join(parts)

    # capitalize first letter
    if s:
        s = s[0].upper() + s[1:]
    return _finalize_summary(s)


# REPLACE your existing _hf_summarize with this:

def _hf_summarize_old(title: str, source: str, text: str) -> Optional[str]:
    pipe = _hf_get_pipe()
    if pipe is None:
        return None
    try:
        max_new   = int(os.getenv("HF_MAX_NEW_TOKENS", "32"))   # smaller = faster
        min_new   = int(os.getenv("HF_MIN_NEW_TOKENS", "8"))
        num_beams = int(os.getenv("HF_NUM_BEAMS", "1"))         # greedy
        max_chars = int(os.getenv("CHUNK_MAX_CHARS", "2200"))   # keep 2200 since you’re mostly 1 chunk
        batch_sz  = int(os.getenv("HF_BATCH_SIZE", "8"))

        chunks = _chunk(text, max_chars=max_chars)
        if not chunks:
            return None

        import time
        t0 = time.perf_counter()

        gen_kwargs = dict(
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=False,
            num_beams=num_beams,
            length_penalty=1.0,
            truncation=True,
            no_repeat_ngram_size=3,
        )

        if len(chunks) == 1:
            # Avoid list overhead for the common case
            out = pipe(chunks[0], **gen_kwargs)[0]["summary_text"].strip()
            outs = [_finalize_summary(out)]
        else:
            # Only batch when it actually helps
            outs_raw = pipe(chunks, batch_size=batch_sz, **gen_kwargs)
            outs = [_finalize_summary(o["summary_text"].strip()) for o in outs_raw]

        joined = _finalize_summary(" ".join(outs)).strip() or _extractive(text)
        print(f"[HF] summarization_time_sec={time.perf_counter() - t0:.3f} chunks={len(chunks)}{' (batched)' if len(chunks)>1 else ''}")
        return joined
    except Exception as e:
        import traceback
        print("[HF] error:", repr(e))
        traceback.print_exc()
        return None

# Keep the single-item helper for other call sites if needed
def _hf_summarize_older(title: str, source: str, text: str) -> Optional[str]:
    res = _hf_summarize_many([_strip_html(text or "")])
    return (res[0] if res else None) or _extractive(text or "")

# New: summarize many texts in ONE call
def _hf_summarize_many(texts: List[str]) -> Optional[List[str]]:
    """Summarize many texts in ONE call. Prefer hot remote server; fallback to local; always polish."""
    if not texts:
        return []

    # 1) Try remote server (no cold start)
    outs = _hf_summarize_many_remote(texts)
    if outs is not None:
        return [_polish_summary(o) for o in outs]

    # 2) Local fallback (will show [HF] device=... in logs)
    pipe = _hf_get_pipe()
    if pipe is None:
        return [_polish_summary(_extractive(t)) for t in texts]

    import time
    t0 = time.perf_counter()

    max_new   = int(os.getenv("HF_MAX_NEW_TOKENS", "64"))
    min_new   = int(os.getenv("HF_MIN_NEW_TOKENS", "24"))
    num_beams = int(os.getenv("HF_NUM_BEAMS", "3"))
    batch_sz  = int(os.getenv("HF_BATCH_SIZE", "12"))
    no_rep    = int(os.getenv("HF_NO_REPEAT", "4"))
    rep_pen   = float(os.getenv("HF_REP_PENALTY", "1.10"))
    len_pen   = float(os.getenv("HF_LENGTH_PENALTY", "1.05"))

    gen_kwargs = dict(
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        do_sample=False,
        num_beams=num_beams,
        length_penalty=len_pen,
        truncation=True,
        no_repeat_ngram_size=no_rep,
        repetition_penalty=rep_pen,
        early_stopping=True,
    )

    try:
        outs_raw = pipe(texts, batch_size=batch_sz, **gen_kwargs)
        outs = [_polish_summary(o["summary_text"].strip()) for o in outs_raw]
        print(f"[HF] summarization_time_sec={time.perf_counter() - t0:.3f} items={len(texts)} (single call)")
        return outs
    except Exception as e:
        print(f"[HF] error during batch summarize: {e}")
        return [_polish_summary(_extractive(t)) for t in texts]




def _hf_summarize(title: str, source: str, text: str) -> Optional[str]:
    res = _hf_summarize_many([_strip_html(text or "")])
    return (res[0] if res else None) or _extractive(text or "")

def _hf_summarize_many_remote(texts: List[str]) -> Optional[List[str]]:
    """Call the hot HF server if HF_REMOTE_URL is set."""
    url = os.getenv("HF_REMOTE_URL", "").strip()
    if not url:
        return None
    try:
        payload = {
            "texts": texts,
            "max_new_tokens": int(os.getenv("HF_MAX_NEW_TOKENS", "64")),
            "min_new_tokens": int(os.getenv("HF_MIN_NEW_TOKENS", "24")),
            "num_beams": int(os.getenv("HF_NUM_BEAMS", "3")),
            "batch_size": int(os.getenv("HF_BATCH_SIZE", "12")),
            "no_repeat_ngram_size": int(os.getenv("HF_NO_REPEAT", "4")),
            "repetition_penalty": float(os.getenv("HF_REP_PENALTY", "1.10")),
            "length_penalty": float(os.getenv("HF_LENGTH_PENALTY", "1.05")),
        }
        r = requests.post(url.rstrip("/") + "/summarize_many", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        outs = data.get("summaries", None)
        if not isinstance(outs, list):
            print(f"[HF REMOTE] bad response: {data}")
            return None
        print(f"[HF REMOTE] ok items={len(texts)}")
        return outs
    except Exception as e:
        print(f"[HF REMOTE] failed: {e}")
        return None


# ---------- TextRank (optional, CPU-only) ----------

def _textrank_summarize(text: str, sentences: int = 3) -> Optional[str]:
    if os.getenv("USE_TEXTRANK", "0") != "1":
        return None
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        clean = _strip_html(text or "")
        parser = PlaintextParser.from_string(clean, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        sents = summarizer(parser.document, sentences)
        out = " ".join(str(s) for s in sents).strip()
        return out or None
    except Exception:
        return None

# ---------- main entry ----------

def _timed(callable_fn, *args, label: str = "", **kwargs) -> Tuple[Optional[str], float]:
    t0 = time.perf_counter()
    result = callable_fn(*args, **kwargs)
    t1 = time.perf_counter()
    if label:
        print(f"[{label}] summarization_time_sec={t1 - t0:.3f}")
    return result, (t1 - t0)

def summarize_item(item: dict) -> str:
    # Build clean seed
    seed = (item.get("content_seed") or "") + "\n" + (item.get("summary_seed") or "")
    seed = _clean_entities(seed.strip())
    title = _clean_entities(item.get("title", ""))
    source = _clean_entities(item.get("source", ""))

    if len(seed) < 40:
        seed = (seed + " " + title).strip()

    # Hard override: SUMMARY_BACKEND=openai|ollama|hf|textrank|auto
    backend = (os.getenv("SUMMARY_BACKEND", "") or "").lower().strip()
    if backend == "openai":
        return _openai_summarize(title, source, seed) or _extractive(seed) or (title or "")
    if backend == "ollama":
        return _ollama_summarize(title, source, seed) or _extractive(seed) or (title or "")
    if backend == "hf":
        return _hf_summarize(title, source, seed) or _extractive(seed) or (title or "")
    if backend == "textrank":
        return _textrank_summarize(seed) or _extractive(seed) or (title or "")

    # Auto pipeline with feature flags (defaults keep your old behavior)
    hf_default = _env_bool("USE_HF_SUMMARY", False)
    enabled = {
        "openai":   _backend_enabled("OPENAI", True),
        "ollama":   _backend_enabled("OLLAMA", True),
        "hf":       _backend_enabled("HF", hf_default),
        "textrank": _backend_enabled("TEXTRANK", False),
    }

    if enabled["openai"]:
        s, _ = _timed(_openai_summarize, title, source, seed, label="OPENAI")
        if s: return s
    if enabled["ollama"]:
        s, _ = _timed(_ollama_summarize, title, source, seed, label="OLLAMA")
        if s: return s
    if enabled["hf"]:
        s, _ = _timed(_hf_summarize, title, source, seed, label="HF")
        if s: return s
    if enabled["textrank"]:
        s, _ = _timed(_textrank_summarize, seed, label="TEXTRANK")
        if s: return s

    # Final fallback
    t0 = time.perf_counter()
    s = _extractive(seed)
    print(f"[FALLBACK] summarization_time_sec={time.perf_counter() - t0:.3f}")
    return s
