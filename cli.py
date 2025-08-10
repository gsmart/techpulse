from dotenv import load_dotenv
load_dotenv()  # loads .env before anything else

from src.agent.run import run_once

if __name__ == "__main__":
    run_once()
