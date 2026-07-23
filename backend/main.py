import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import db
from backend.inventory import inventory_store
from backend.llm_agent import handle_chat_turn

app = FastAPI(title="dubizzle Car Assistant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.init_db()


class ChatRequest(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    is_returning_user: bool
    tool_calls: list[dict] = []


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")

    user_info = db.get_or_create_user(req.user_id)
    session_id = req.session_id or str(uuid.uuid4())
    db.get_or_create_session(session_id, req.user_id)

    result = handle_chat_turn(
        user_id=req.user_id,
        session_id=session_id,
        user_message=req.message,
        is_returning=user_info["is_returning"],
    )
    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        is_returning_user=user_info["is_returning"],
        tool_calls=result["tool_calls"],
    )


@app.post("/session/new")
def new_session(user_id: str):
    session_id = str(uuid.uuid4())
    db.get_or_create_session(session_id, user_id)
    return {"session_id": session_id}


@app.get("/users/{user_id}/profile")
def get_profile(user_id: str):
    profile = db.get_user_profile(user_id)
    if profile is None:
        raise HTTPException(404, "user not found")
    return profile


@app.get("/inventory/stats")
def inventory_stats():
    return inventory_store.stats()


@app.get("/inventory/search")
def inventory_search(
    make: str | None = None,
    model: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    body_type: str | None = None,
    keyword: str | None = None,
    limit: int = 10,
):
    return inventory_store.search(
        make=make, model=model, min_year=min_year, max_year=max_year,
        min_price=min_price, max_price=max_price, body_type=body_type,
        keyword=keyword, limit=limit,
    )


@app.get("/leads")
def get_leads():
    import csv
    from backend.config import LEADS_CSV_PATH
    import os

    if not os.path.exists(LEADS_CSV_PATH):
        return []
    with open(LEADS_CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


@app.get("/health")
def health():
    return {"status": "ok"}
