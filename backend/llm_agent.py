"""
The agent orchestration layer: builds the prompt (guardrails + long-term
memory + short-term history), runs the tool-calling loop against the LLM,
and keeps lead scoring in sync.
"""
from __future__ import annotations

import json

from backend import db, memory
from backend.config import AGENT_NAME, COMPETITOR_NAMES, VIEWING_DAYS, VIEWING_OPEN_HOUR, VIEWING_CLOSE_HOUR
from backend.lead_scoring import LeadSignals, upsert_lead_csv
from backend.llm_client import chat_completion
from backend.tools import TOOL_SCHEMAS, execute_tool

MAX_TOOL_ITERATIONS = 5
SUMMARIZE_EVERY_N_TURNS = 4


def _system_prompt(memory_context: str, in_view_reminder: str) -> str:
    competitor_list = ", ".join(COMPETITOR_NAMES)
    return f"""You are {AGENT_NAME}, dubizzle's AI car-shopping assistant for the dubizzle Cars
marketplace in the UAE. You help users search the current inventory, answer questions about
specific listings, and book viewing/test-drive slots.

SCOPE & GUARDRAILS (follow strictly):
- You ONLY discuss: cars in the provided inventory, general car-buying questions relevant to
  this marketplace, booking viewings, and light friendly chit-chat.
- Politely decline anything unrelated (coding help, history questions, general trivia, unrelated
  tasks) and steer the conversation back to cars. Keep the decline brief and warm.
- NEVER mention, compare to, or recommend competing platforms, including: {competitor_list}.
  If asked about a competitor, politely say you can only help with dubizzle's own listings.
- NEVER invent car specs, prices, or availability. Every factual claim about a listing must come
  from a tool call (search_inventory / get_car_details). If a price or spec isn't in the data,
  say it's "not listed" / "price on request" rather than guessing a number.
- Some listings genuinely have no listed price (only ~13% of this inventory states a firm cash
  price) - that's normal, just be transparent about it and offer to note the user's budget for
  the seller instead.

BOOKING RULES:
- Viewings/test drives are available {', '.join(VIEWING_DAYS)}, {VIEWING_OPEN_HOUR}:00-{VIEWING_CLOSE_HOUR}:00 only (no Sundays).
- Always confirm the specific car (listing_id), day, and time with the user before calling book_viewing.
- If a slot is already taken, apologize and offer to find another time.

LEAD QUALIFICATION (do this naturally, don't interrogate):
- As the conversation flows, try to learn: the user's name, budget range, preferred make/body
  type, how flexible their timing is, and whether they're in a hurry to buy. Whenever the user
  reveals any of this, call save_lead_info to record it - even partial info is useful.
- Don't ask more than one qualifying question per turn, and always prioritize actually helping
  them find/explore cars over interrogating them.

MEMORY:
{memory_context}

{in_view_reminder}

STYLE: Friendly, concise, knowledgeable - like a helpful dubizzle showroom assistant. Use AED for
prices. When listing cars, mention listing_id (e.g. "#12"), price, year, and one standout detail,
not the entire description unless asked."""


def _in_view_reminder(session: dict) -> str:
    raw = session.get("last_search_results")
    if not raw:
        return "No cars have been shown yet this session."
    try:
        listings = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not listings:
        return ""
    lines = ["Cars currently 'in view' from the last search (for resolving things like 'the first one'):"]
    for c in listings:
        lines.append(f"  #{c['listing_id']}: {c['year']} {c['make'].title()} {c['model'].title()} - {c['price_display']}")
    return "\n".join(lines)


def _looks_like_question(text: str) -> bool:
    return "?" in text or any(
        text.lower().startswith(w) for w in ("what", "how", "is there", "does", "can", "do you", "which", "when")
    )


def _recompute_lead(user_id: str, session_id: str, is_returning: bool = False):
    profile = db.get_user_profile(user_id) or {}
    session = db.get_session(session_id) or {}
    signals = LeadSignals(
        mentioned_budget=bool(session.get("mentioned_budget")),
        questions_asked=session.get("questions_asked", 0),
        requested_viewing=bool(session.get("requested_viewing")),
        flexible_timing=bool(session.get("flexible_timing")),
        returning_user=is_returning,
        mentioned_urgency=bool(session.get("mentioned_urgency")),
        has_stated_requirements=bool(
            profile.get("preferred_make") or profile.get("preferred_body_type")
            or profile.get("budget_min") or profile.get("budget_max")
        ),
    )
    upsert_lead_csv(
        user_id=user_id,
        name=profile.get("display_name"),
        email=None,
        budget_min=profile.get("budget_min"),
        budget_max=profile.get("budget_max"),
        preferred_make=profile.get("preferred_make"),
        body_type=profile.get("preferred_body_type"),
        viewing_booked=bool(session.get("requested_viewing")),
        signals=signals,
    )


def handle_chat_turn(user_id: str, session_id: str, user_message: str, is_returning: bool) -> dict:
    db.add_message(session_id, "user", user_message)
    db.touch_session(
        session_id,
        turn_count=1,
        questions_asked=1 if _looks_like_question(user_message) else 0,
    )

    session = db.get_session(session_id) or {}
    mem_ctx = memory.build_memory_context(user_id, is_returning)
    in_view = _in_view_reminder(session)
    system_prompt = _system_prompt(mem_ctx, in_view)

    history = db.get_messages(session_id, limit=30)
    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m["role"] if m["role"] in ("user", "assistant") else "user"
        llm_messages.append({"role": role, "content": m["content"]})

    tool_results_for_client = []
    final_text = ""

    for _ in range(MAX_TOOL_ITERATIONS):
        response = chat_completion(llm_messages, tools=TOOL_SCHEMAS)
        choice = response.choices[0]
        msg = choice.message

        if getattr(msg, "tool_calls", None):
            llm_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(tc.function.name, args, user_id=user_id, session_id=session_id)
                tool_results_for_client.append({"tool": tc.function.name, "args": args, "result": result})
                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
            continue  # loop again so the model can use the tool result

        final_text = msg.content or ""
        break

    if not final_text:
        final_text = "Sorry, I had trouble processing that - could you rephrase?"

    db.add_message(session_id, "assistant", final_text)

    _recompute_lead(user_id, session_id, is_returning)

    updated_session = db.get_session(session_id) or {}
    if updated_session.get("turn_count", 0) % SUMMARIZE_EVERY_N_TURNS == 0:
        try:
            memory.summarize_session_and_store(user_id, session_id)
        except Exception:
            pass

    return {
        "reply": final_text,
        "tool_calls": tool_results_for_client,
    }
