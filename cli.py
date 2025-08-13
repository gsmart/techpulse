# cli.py
from dotenv import load_dotenv
load_dotenv()  # load .env first

import os, sys, time, argparse
from src.agent.run import run_once

CHOICES = ("openai", "ollama", "hf", "auto")

def _env(key: str) -> str | None:
    v = os.getenv(key)
    return v.strip() if v else None

def _choose_backend_interactive() -> str:
    print("\nChoose summarization backend:")
    print("  1) OpenAI (cloud)")
    print("  2) Ollama  (local LLM)")
    print("  3) HF      (Transformers on CPU/MPS)")
    print("  4) Auto    (OpenAI → Ollama → HF → fallback)\n")
    while True:
        sel = input("Enter 1/2/3/4: ").strip()
        mapping = {"1": "openai", "2": "ollama", "3": "hf", "4": "auto"}
        if sel in mapping:
            return mapping[sel]
        print("Invalid choice, please use 1/2/3/4.")

def _resolve_backend(args_backend: str | None, interactive: bool) -> str:
    # precedence:
    #   1) --backend if provided
    #   2) if --interactive (or FORCE_INTERACTIVE=1): prompt
    #   3) SUMMARY_BACKEND env if valid
    #   4) prompt if TTY
    #   5) 'auto'
    if args_backend:
        return args_backend
    if interactive or os.getenv("FORCE_INTERACTIVE") == "1":
        return _choose_backend_interactive()
    env = (_env("SUMMARY_BACKEND") or "").lower()
    if env in CHOICES:
        return env
    if sys.stdin.isatty():
        return _choose_backend_interactive()
    return "auto"

def _validate_backend(b: str) -> str:
    if b == "openai" and not _env("OPENAI_API_KEY"):
        print("[warn] OPENAI_API_KEY missing; switching to 'hf'.")
        return "hf"
    if b == "ollama" and not _env("OLLAMA_HOST"):
        print("[warn] OLLAMA_HOST missing; switching to 'hf'.")
        return "hf"
    return b

def main():
    ap = argparse.ArgumentParser(description="TechPulse — hourly TL;DR generator")
    ap.add_argument("--backend", choices=CHOICES, help="openai | ollama | hf | auto")
    ap.add_argument("--interactive", action="store_true", help="force interactive backend selection")
    ap.add_argument("--max-items", type=int, default=12, help="limit items per run (default: 12)")
    ap.add_argument("--window-minutes", type=int, default=180, help="lookback window (default: 180)")
    args = ap.parse_args()

    backend = _validate_backend(_resolve_backend(args.backend, args.interactive))
    os.environ["SUMMARY_BACKEND"] = backend  # used by summarize.py

    print(f"\n[TechPulse] backend={backend} max_items={args.max_items} window={args.window_minutes}m")

    t0 = time.perf_counter()
    try:
        run_once(max_items=args.max_items, window_minutes=args.window_minutes)
    finally:
        dt = time.perf_counter() - t0
        print(f"[TechPulse] total_time_sec={dt:.2f}\n")

if __name__ == "__main__":
    main()
