"""
Tool (function-calling) definitions exposed to the LLM, and their Python
implementations. Keeping these in one file makes the boundary between
"what the model can ask for" and "what the backend actually does" explicit -
the model can never fabricate inventory data or bookings because every
fact it states must come back through one of these functions.
"""
from __future__ import annotations

import time
from typing import Any

from backend import db
from backend.config import VIEWING_DAYS, VIEWING_OPEN_HOUR, VIEWING_CLOSE_HOUR
from backend.inventory import inventory_store

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_inventory",
            "description": (
                "Search the dubizzle used-car inventory. Use this whenever the user asks "
                "about cars, makes/models, price range, body type (SUV/sedan/coupe/etc.), "
                "year, or features/keywords (e.g. 'sunroof', 'GCC', 'warranty'). Never invent "
                "cars or specs - always call this tool to ground your answer in real listings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "make": {"type": "string", "description": "e.g. 'mercedes-benz', 'toyota'"},
                    "model": {"type": "string", "description": "e.g. 'c-class', 'explorer'"},
                    "min_year": {"type": "integer"},
                    "max_year": {"type": "integer"},
                    "min_price": {"type": "integer", "description": "AED"},
                    "max_price": {"type": "integer", "description": "AED"},
                    "body_type": {"type": "string", "description": "SUV, Sedan, Coupe, Hatchback, Convertible, Pickup, Wagon, Van"},
                    "keyword": {"type": "string", "description": "free text keywords to match in title/description, e.g. 'sunroof GCC'"},
                    "limit": {"type": "integer", "description": "max results, default 5"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_car_details",
            "description": "Get the full details of one specific listing by its listing_id. Use this for follow-up questions about a car already mentioned in the conversation (mileage, warranty, full description, etc.).",
            "parameters": {
                "type": "object",
                "properties": {"listing_id": {"type": "integer"}},
                "required": ["listing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_viewing_availability",
            "description": "Check what days/hours viewings can be booked for. Viewings are always Monday-Saturday, 8am-8pm (no Sundays). Use before booking if the user hasn't picked a specific slot yet.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_viewing",
            "description": "Book a car viewing / test-drive slot for the user. Only call this once you have a listing_id, a day (Monday-Saturday), and a time (an hour between 8am-8pm). Confirm details with the user before calling if unclear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "integer"},
                    "day": {"type": "string", "description": "Monday, Tuesday, ... Saturday"},
                    "time_slot": {"type": "string", "description": "e.g. '10:00 AM', '2:00 PM'"},
                },
                "required": ["listing_id", "day", "time_slot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_lead_info",
            "description": (
                "Record/update what we know about this user as a sales lead: their name, email, "
                "budget range, preferred make/body type, whether their timing is flexible, and "
                "whether they've expressed urgency to buy soon. Call this whenever the user reveals "
                "any of this information during the conversation, even partially - it updates their "
                "profile and lead score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "budget_min": {"type": "integer"},
                    "budget_max": {"type": "integer"},
                    "preferred_make": {"type": "string"},
                    "preferred_body_type": {"type": "string"},
                    "flexible_timing": {"type": "boolean"},
                    "mentioned_urgency": {"type": "boolean", "description": "true if user said things like 'need it this week', 'urgent', 'buying soon'"},
                },
            },
        },
    },
]


def _fmt_car_line(c: dict) -> str:
    bits = [f"#{c['listing_id']}", str(c["year"]), c["make"].title(), c["model"].title()]
    if c.get("trim") and c["trim"].lower() != "other":
        bits.append(c["trim"].title())
    bits.append(f"- {c['price_display']}")
    if c.get("mileage_km"):
        bits.append(f"- {c['mileage_km']:,}km")
    if c.get("body_type") and c["body_type"] != "Unspecified":
        bits.append(f"- {c['body_type']}")
    return " ".join(bits)


def execute_tool(name: str, args: dict[str, Any], *, user_id: str, session_id: str) -> dict:
    """Executes a tool call and returns a JSON-serialisable result dict."""

    if name == "search_inventory":
        results = inventory_store.search(
            make=args.get("make"),
            model=args.get("model"),
            min_year=args.get("min_year"),
            max_year=args.get("max_year"),
            min_price=args.get("min_price"),
            max_price=args.get("max_price"),
            body_type=args.get("body_type"),
            keyword=args.get("keyword"),
            limit=args.get("limit") or 5,
        )
        db.set_last_search_results(session_id, results)
        return {"count": len(results), "listings": results}

    if name == "get_car_details":
        car = inventory_store.get_car(int(args["listing_id"]))
        if car is None:
            return {"error": f"No listing with id {args['listing_id']}"}
        return {"listing": car}

    if name == "check_viewing_availability":
        return {
            "days": VIEWING_DAYS,
            "hours": f"{VIEWING_OPEN_HOUR}:00 - {VIEWING_CLOSE_HOUR}:00",
            "note": "No viewings on Sunday.",
        }

    if name == "book_viewing":
        listing_id = int(args["listing_id"])
        day = str(args["day"]).strip().title()
        time_slot = str(args["time_slot"]).strip()

        if day not in VIEWING_DAYS:
            return {"error": f"'{day}' is not a bookable day. Viewings run Monday-Saturday only."}
        car = inventory_store.get_car(listing_id)
        if car is None:
            return {"error": f"No listing with id {listing_id}"}
        if db.is_slot_taken(listing_id, day, time_slot):
            return {"error": f"That slot ({day} {time_slot}) for listing #{listing_id} is already booked. Please suggest another time."}

        db.add_booking(user_id, session_id, listing_id, day, time_slot)
        db.set_session_flag(session_id, "requested_viewing", 1)
        db.append_liked_listing(user_id, listing_id)
        return {
            "confirmed": True,
            "listing_id": listing_id,
            "car_title": car["title"],
            "day": day,
            "time_slot": time_slot,
            "message": f"Viewing confirmed for {car['title']} on {day} at {time_slot}.",
        }

    if name == "save_lead_info":
        prefs_update = {}
        if args.get("budget_min") is not None:
            prefs_update["budget_min"] = args["budget_min"]
        if args.get("budget_max") is not None:
            prefs_update["budget_max"] = args["budget_max"]
        if args.get("preferred_make"):
            prefs_update["preferred_make"] = args["preferred_make"]
        if args.get("preferred_body_type"):
            prefs_update["preferred_body_type"] = args["preferred_body_type"]
        if prefs_update:
            db.update_user_prefs(user_id, **prefs_update)

        if args.get("budget_min") is not None or args.get("budget_max") is not None:
            db.set_session_flag(session_id, "mentioned_budget", 1)
        if args.get("flexible_timing"):
            db.set_session_flag(session_id, "flexible_timing", 1)
        if args.get("mentioned_urgency"):
            db.set_session_flag(session_id, "mentioned_urgency", 1)

        if args.get("name"):
            db.get_or_create_user(user_id, display_name=args["name"])

        return {"saved": True, "profile_updates": prefs_update}

    return {"error": f"Unknown tool '{name}'"}
