"""
Lead scoring system.

Rules (as specified by the business):
    Mentioned budget            +10
    Asked >3 questions          +10
    Requested viewing           +30
    Has flexible timings        +5
    Returning user               +10
    Mentioned urgent purchase   +20
    Doesn't specify requirements -5

Score buckets:
    >= 70  -> Hot Lead
    40-69  -> Warm Lead
    < 40   -> Cold Lead

Every time a lead-relevant event happens (budget mentioned, viewing booked,
etc.) we recompute the score and upsert a row into data/leads.csv - this
simulates leads being captured/qualified for a sales team, per the brief.
"""
import csv
import os
import time
from dataclasses import dataclass, field

from backend.config import LEADS_CSV_PATH

CSV_FIELDS = [
    "user_id", "name", "email", "budget_min", "budget_max", "preferred_make",
    "body_type", "viewing_booked", "returning_user", "lead_score",
    "lead_status", "reasons", "updated_at",
]


@dataclass
class LeadSignals:
    mentioned_budget: bool = False
    questions_asked: int = 0
    requested_viewing: bool = False
    flexible_timing: bool = False
    returning_user: bool = False
    mentioned_urgency: bool = False
    has_stated_requirements: bool = False  # make/body-type/anything specific


def compute_score(signals: LeadSignals) -> tuple[int, str, list[str]]:
    score = 0
    reasons = []

    if signals.mentioned_budget:
        score += 10
        reasons.append("Budget specified")
    if signals.questions_asked > 3:
        score += 10
        reasons.append(f"Engaged buyer ({signals.questions_asked} questions asked)")
    if signals.requested_viewing:
        score += 30
        reasons.append("Test drive / viewing booked")
    if signals.flexible_timing:
        score += 5
        reasons.append("Flexible on viewing timing")
    if signals.returning_user:
        score += 10
        reasons.append("Returning user")
    if signals.mentioned_urgency:
        score += 20
        reasons.append("Looking to purchase urgently")
    if not signals.has_stated_requirements:
        score -= 5
        reasons.append("Requirements not yet specified")

    score = max(0, score)
    if score >= 70:
        status = "Hot Lead"
    elif score >= 40:
        status = "Warm Lead"
    else:
        status = "Cold Lead"
    return score, status, reasons


def upsert_lead_csv(
    user_id: str,
    name: str | None,
    email: str | None,
    budget_min: int | None,
    budget_max: int | None,
    preferred_make: str | None,
    body_type: str | None,
    viewing_booked: bool,
    signals: LeadSignals,
):
    score, status, reasons = compute_score(signals)
    row = {
        "user_id": user_id,
        "name": name or "",
        "email": email or "",
        "budget_min": budget_min or "",
        "budget_max": budget_max or "",
        "preferred_make": preferred_make or "",
        "body_type": body_type or "",
        "viewing_booked": viewing_booked,
        "returning_user": signals.returning_user,
        "lead_score": score,
        "lead_status": status,
        "reasons": "; ".join(reasons),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    existing_rows = []
    if os.path.exists(LEADS_CSV_PATH):
        with open(LEADS_CSV_PATH, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    replaced = False
    for i, r in enumerate(existing_rows):
        if r["user_id"] == user_id:
            existing_rows[i] = row
            replaced = True
            break
    if not replaced:
        existing_rows.append(row)

    with open(LEADS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)

    return row
