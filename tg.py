import os, requests
from dotenv import load_dotenv

load_dotenv()  # <-- loads .env from the current directory

token = os.getenv("TELEGRAM_BOT_TOKEN")
chat = os.getenv("TELEGRAM_CHAT_ID")

print(f"TOKEN={token}, CHAT={chat}")  # debug print

r = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    json={"chat_id": chat, "text": "Hello from TechPulse 👋"},
    timeout=15
)
print(r.status_code, r.text)
