"""
Central configuration for the dubizzle Car Assistant backend.
All values can be overridden via environment variables / .env file.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Dataset & storage paths ---
CARS_XLSX_PATH = os.getenv("CARS_XLSX_PATH", str(DATA_DIR / "cars_dataset.xlsx"))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "app.db"))
LEADS_CSV_PATH = os.getenv("LEADS_CSV_PATH", str(DATA_DIR / "leads.csv"))

# --- LLM (Google Gemini, via LiteLLM) ---
# Get a free key at https://aistudio.google.com/app/apikey and put it in .env as GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")

# --- Business rules ---
VIEWING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
VIEWING_OPEN_HOUR = 8   # 8am
VIEWING_CLOSE_HOUR = 20  # 8pm

# Competitor names the assistant must never discuss / recommend
COMPETITOR_NAMES = [
    "carswitch", "yallamotor", "opensooq", "autotrader",
    "cars24", "carfirst", "facebook marketplace",
]

AGENT_NAME = "Dubi"  # the assistant's persona name
