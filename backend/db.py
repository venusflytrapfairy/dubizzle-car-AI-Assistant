"""
Persistent storage module (SQLite).

Tables:
  users            - one row per returning user (long-term memory anchor)
  user_prefs       - structured preferences we've learned about a user (budget,
                     preferred make/body type, liked listings) - updated as the
                     conversation progresses, read back in on every new session
  sessions         - one row per chat session (a user can have many)
  messages         - full turn-by-turn transcript per session (short-term memory,
                     reconstructed into the LLM context on every /chat call)
  session_summaries- a rolling LLM-generated summary of each finished/ongoing
                     session, appended to the user's long-term memory so future
                     sessions can say "last time you were looking at ..."
  bookings         - test-drive / viewing slot bookings
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from backend.config import SQLITE_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at REAL,
    last_seen_at REAL
);

CREATE TABLE IF NOT EXISTS user_prefs (
    user_id TEXT PRIMARY KEY,
    budget_min INTEGER,
    budget_max INTEGER,
    preferred_make TEXT,
    preferred_body_type TEXT,
    liked_listing_ids TEXT,      -- JSON list
    long_term_summary TEXT,      -- rolling natural-language memory
    updated_at REAL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    started_at REAL,
    last_active_at REAL,
    turn_count INTEGER DEFAULT 0,
    questions_asked INTEGER DEFAULT 0,
    mentioned_budget INTEGER DEFAULT 0,
    requested_viewing INTEGER DEFAULT 0,
    flexible_timing INTEGER DEFAULT 0,
    mentioned_urgency INTEGER DEFAULT 0,
    last_search_results TEXT,   -- JSON list of listing dicts currently "in view"
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,             -- 'user' | 'assistant' | 'tool'
    content TEXT,
    created_at REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    session_id TEXT,
    listing_id INTEGER,
    day TEXT,
    time_slot TEXT,
    created_at REAL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- users ----
def get_or_create_user(user_id: str, display_name: Optional[str] = None) -> dict:
    now = time.time()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        is_returning = row is not None
        if row is None:
            conn.execute(
                "INSERT INTO users (user_id, display_name, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
                (user_id, display_name, now, now),
            )
            conn.execute(
                "INSERT INTO user_prefs (user_id, liked_listing_ids, updated_at) VALUES (?, '[]', ?)",
                (user_id, now),
            )
        else:
            conn.execute(
                "UPDATE users SET last_seen_at = ?, display_name = COALESCE(?, display_name) WHERE user_id = ?",
                (now, display_name, user_id),
            )
    return {"user_id": user_id, "is_returning": is_returning}


def get_user_profile(user_id: str) -> Optional[dict]:
    with get_conn() as conn:
        u = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not u:
            return None
        p = conn.execute("SELECT * FROM user_prefs WHERE user_id = ?", (user_id,)).fetchone()
        profile = dict(u)
        if p:
            profile.update(dict(p))
            profile["liked_listing_ids"] = json.loads(p["liked_listing_ids"] or "[]")
        return profile


def update_user_prefs(user_id: str, **fields):
    if not fields:
        return
    with get_conn() as conn:
        existing = conn.execute("SELECT * FROM user_prefs WHERE user_id = ?", (user_id,)).fetchone()
        if existing is None:
            conn.execute("INSERT INTO user_prefs (user_id, liked_listing_ids, updated_at) VALUES (?, '[]', ?)",
                         (user_id, time.time()))
        cols, vals = [], []
        for k, v in fields.items():
            if k == "liked_listing_ids" and isinstance(v, list):
                v = json.dumps(v)
            cols.append(f"{k} = ?")
            vals.append(v)
        vals.append(time.time())
        vals.append(user_id)
        conn.execute(f"UPDATE user_prefs SET {', '.join(cols)}, updated_at = ? WHERE user_id = ?", vals)


def append_liked_listing(user_id: str, listing_id: int):
    profile = get_user_profile(user_id) or {}
    liked = profile.get("liked_listing_ids") or []
    if listing_id not in liked:
        liked.append(listing_id)
    update_user_prefs(user_id, liked_listing_ids=liked)


# ------------------------------------------------------------- sessions ----
def get_or_create_session(session_id: str, user_id: str) -> dict:
    now = time.time()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions (session_id, user_id, started_at, last_active_at) VALUES (?, ?, ?, ?)",
                (session_id, user_id, now, now),
            )
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row)


def touch_session(session_id: str, **increments):
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET last_active_at = ? WHERE session_id = ?", (time.time(), session_id))
        for field, delta in increments.items():
            conn.execute(f"UPDATE sessions SET {field} = {field} + ? WHERE session_id = ?", (delta, session_id))


def set_session_flag(session_id: str, field: str, value=1):
    with get_conn() as conn:
        conn.execute(f"UPDATE sessions SET {field} = ? WHERE session_id = ?", (value, session_id))


def set_last_search_results(session_id: str, results: list):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET last_search_results = ? WHERE session_id = ?",
            (json.dumps(results), session_id),
        )


def get_session(session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


# ------------------------------------------------------------- messages ----
def add_message(session_id: str, role: str, content: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )


def get_messages(session_id: str, limit: int = 40) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ------------------------------------------------------------- bookings ----
def add_booking(user_id: str, session_id: str, listing_id: int, day: str, time_slot: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO bookings (user_id, session_id, listing_id, day, time_slot, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, session_id, listing_id, day, time_slot, time.time()),
        )


def is_slot_taken(listing_id: int, day: str, time_slot: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM bookings WHERE listing_id = ? AND day = ? AND time_slot = ?",
            (listing_id, day, time_slot),
        ).fetchone()
        return row is not None
