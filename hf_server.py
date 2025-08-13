# hf_server.py
import os, time
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

MODEL_ID = os.getenv("HF_SUMMARY_MODEL", "sshleifer/distilbart-cnn-6-6")
FORCE = (os.getenv("HF_DEVICE", "").strip().lower() or None)  # 'mps'|'cpu'|None
USE_CPU_ONLY = os.getenv("USE_CPU_ONLY", "0") == "1"
MPS_AVAIL = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if FORCE == "mps":
    USE_MPS = True;  USE_CPU_ONLY = False
elif FORCE == "cpu":
    USE_MPS = False; USE_CPU_ONLY = True
else:
    USE_MPS = (not USE_CPU_ONLY) and MPS_AVAIL

try:
    n_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
    torch.set_num_threads(n_threads)
except Exception:
    pass

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=(torch.float16 if USE_MPS else None))
if USE_MPS:
    model.to("mps"); DEVICE = 0
else:
    DEVICE = -1

pipe = pipeline("summarization", model=model, tokenizer=tok, device=DEVICE)
print(f"[HF SERVER] device={'mps' if USE_MPS else 'cpu'} model={MODEL_ID}")

# one-time warmup
_ = pipe("Hello.")

class SummarizeManyIn(BaseModel):
    texts: List[str]
    max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "64"))
    min_new_tokens: int = int(os.getenv("HF_MIN_NEW_TOKENS", "24"))
    num_beams: int = int(os.getenv("HF_NUM_BEAMS", "3"))
    batch_size: int = int(os.getenv("HF_BATCH_SIZE", "12"))
    no_repeat_ngram_size: int = int(os.getenv("HF_NO_REPEAT", "4"))
    repetition_penalty: float = float(os.getenv("HF_REP_PENALTY", "1.10"))
    length_penalty: float = float(os.getenv("HF_LENGTH_PENALTY", "1.05"))

class SummarizeManyOut(BaseModel):
    summaries: List[str]

app = FastAPI()

@app.post("/summarize_many", response_model=SummarizeManyOut)
def summarize_many(inp: SummarizeManyIn):
    t0 = time.perf_counter()
    outs = pipe(
        inp.texts,
        batch_size=inp.batch_size,
        max_new_tokens=inp.max_new_tokens,
        min_new_tokens=inp.min_new_tokens,
        do_sample=False,
        num_beams=inp.num_beams,
        length_penalty=inp.length_penalty,
        truncation=True,
        no_repeat_ngram_size=inp.no_repeat_ngram_size,
        repetition_penalty=inp.repetition_penalty,
        early_stopping=True,
    )
    summaries = [o["summary_text"].strip() for o in outs]
    print(f"[HF SERVER] time={time.perf_counter()-t0:.3f}s items={len(inp.texts)}")
    return SummarizeManyOut(summaries=summaries)

# run: uvicorn hf_server:app --host 127.0.0.1 --port 8765 --workers 1
