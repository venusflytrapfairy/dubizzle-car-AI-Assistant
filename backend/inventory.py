"""
Inventory module.

The provided dataset only has: Listing_ID, year, make, model, trim, title,
description, photo_url. There is NO clean `price`, `mileage`, or `body_type`
column - those live inconsistently inside the free-text `title`/`description`
fields (e.g. only 13/100 listings mention an AED price at all).

Design decision: rather than have the LLM guess/hallucinate a price or body
type for listings where it isn't stated, we run a light regex-based
enrichment pass ONCE at load time to pull out whatever structured signal is
actually present in the text (price, mileage, regional spec, body type,
warranty mention). Anything we can't confidently extract is left as None and
the agent is instructed to say "not listed" rather than invent a number.

Retrieval itself is a hybrid:
  1. Structured filtering (make / model / year range / price range / body
     type) on the enriched dataframe - this is the accurate, non-hallucinating
     part for the fields we trust.
  2. A keyword/full-text fallback across title+description (simple
     token-overlap scoring, no external vector DB needed at this scale of
     ~100 rows) to catch feature mentions like "sunroof", "AMG", "panoramic
     roof", "GCC", etc. that structured columns can't express.
This keeps grounding tight (SQL/pandas-style tool calling on real data) while
still supporting fuzzy natural-language queries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backend.config import CARS_XLSX_PATH

_PRICE_CANDIDATE_PATTERN = re.compile(
    r"AED\s*([\d,]{4,})(?:\.\d+)?|([\d,]{5,})(?:\.\d+)?\s*(?:AED|aed)", re.I
)
_MONTHLY_HINT = re.compile(r"\s*/\s*mo(?:nth)?\b|\bmonthly\b|\bper month\b|\bp\.?m\.?\b", re.I)
_CASH_HINT = re.compile(r"\bin cash\b|\bcash price\b", re.I)
_MILEAGE_PATTERN = re.compile(r"([\d,]{2,3},?\d{3})\s*k\.?m", re.I)
_BODY_TYPE_KEYWORDS = {
    "suv": "SUV",
    "crossover": "SUV",
    "coupe": "Coupe",
    "sedan": "Sedan",
    "saloon": "Sedan",
    "hatchback": "Hatchback",
    "convertible": "Convertible",
    "cabriolet": "Convertible",
    "roadster": "Convertible",
    "pickup": "Pickup",
    "truck": "Pickup",
    "wagon": "Wagon",
    "estate": "Wagon",
    "van": "Van",
}
# Fallback: known body type by model name, for listings whose text doesn't
# say the body type explicitly (common on dubizzle listings).
_MODEL_BODY_TYPE = {
    "explorer": "SUV", "range rover velar": "SUV", "cayenne": "SUV",
    "bentayga": "SUV", "cullinan": "SUV", "g-class brabus": "SUV",
    "gls-class": "SUV", "pajero": "SUV", "countryman": "SUV",
    "aviator": "SUV", "velar": "SUV",
    "c-class": "Sedan", "e-class": "Sedan", "s-class": "Sedan",
    "3": "Sedan", "altima": "Sedan", "megane": "Sedan",
    "cooper": "Hatchback",
    "continental": "Coupe", "dawn": "Convertible", "phantom": "Sedan",
}
_SPEC_KEYWORDS = {
    "gcc": "GCC Specs", "japanese specs": "Japanese Specs",
    "american specs": "American Specs", "european specs": "European Specs",
    "us specs": "American Specs", "usa": "American Specs",
}


def _extract_price(text: str) -> Optional[int]:
    """Pick the most likely *total cash* price, ignoring monthly-installment
    figures like 'AED 5,805/mo'. When several candidates remain (e.g. a down
    payment plan followed by the actual cash price), prefer one explicitly
    tagged 'in cash' / 'cash price', otherwise take the largest figure since
    financing breakdowns are always smaller than the full price."""
    candidates = []
    for m in _PRICE_CANDIDATE_PATTERN.finditer(text):
        raw = m.group(1) or m.group(2)
        try:
            val = int(raw.replace(",", ""))
        except ValueError:
            continue
        if not (3000 <= val <= 20_000_000):
            continue
        window = text[m.end(): m.end() + 15]
        if _MONTHLY_HINT.search(window):
            continue  # skip monthly installment figures
        tagged_cash = bool(_CASH_HINT.search(text[m.end(): m.end() + 30]))
        candidates.append((tagged_cash, val))

    if not candidates:
        return None
    cash_tagged = [v for tagged, v in candidates if tagged]
    if cash_tagged:
        return max(cash_tagged)
    return max(v for _, v in candidates)


def _extract_mileage(text: str) -> Optional[int]:
    m = _MILEAGE_PATTERN.search(text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _extract_body_type(text_lower: str, model: str) -> Optional[str]:
    for kw, label in _BODY_TYPE_KEYWORDS.items():
        if kw in text_lower:
            return label
    return _MODEL_BODY_TYPE.get(model.lower())


def _extract_spec(text_lower: str) -> Optional[str]:
    for kw, label in _SPEC_KEYWORDS.items():
        if kw in text_lower:
            return label
    return None


def _has_warranty(text_lower: str) -> bool:
    return "warranty" in text_lower


@dataclass
class Car:
    listing_id: int
    year: int
    make: str
    model: str
    trim: str
    title: str
    description: str
    photo_url: str
    price_aed: Optional[int]
    mileage_km: Optional[int]
    body_type: Optional[str]
    regional_spec: Optional[str]
    has_warranty: bool

    def to_dict(self) -> dict:
        return {
            "listing_id": self.listing_id,
            "year": self.year,
            "make": self.make,
            "model": self.model,
            "trim": self.trim,
            "title": self.title,
            "price_aed": self.price_aed if self.price_aed else None,
            "price_display": f"AED {self.price_aed:,}" if self.price_aed else "Price on request",
            "mileage_km": self.mileage_km,
            "body_type": self.body_type or "Unspecified",
            "regional_spec": self.regional_spec,
            "has_warranty": self.has_warranty,
            "description": self.description,
            "photo_url": self.photo_url,
        }


class InventoryStore:
    def __init__(self, xlsx_path: str = CARS_XLSX_PATH):
        self.df = self._load_and_enrich(xlsx_path)

    @staticmethod
    def _load_and_enrich(xlsx_path: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx_path)
        df.columns = [c.strip() for c in df.columns]

        df["make"] = df["make"].astype(str)
        df["model"] = df["model"].astype(str)
        df["trim"] = df["trim"].astype(str)
        df["title"] = df["title"].astype(str)
        df["description"] = df["description"].astype(str)

        combined = (df["title"].fillna("") + " " + df["description"].fillna(""))
        df["price_aed"] = combined.apply(_extract_price)
        df["mileage_km"] = combined.apply(_extract_mileage)
        combined_lower = combined.str.lower()
        df["body_type"] = [
            _extract_body_type(t, m) for t, m in zip(combined_lower, df["model"])
        ]
        df["regional_spec"] = combined_lower.apply(_extract_spec)
        df["has_warranty"] = combined_lower.apply(_has_warranty)
        df["search_blob"] = (
            combined_lower + " " + df["make"].str.lower() + " " + df["model"].str.lower()
        )
        return df

    def get_car(self, listing_id: int) -> Optional[dict]:
        row = self.df[self.df["Listing_ID"] == listing_id]
        if row.empty:
            return None
        return self._row_to_car(row.iloc[0]).to_dict()

    def _row_to_car(self, row) -> Car:
        return Car(
            listing_id=int(row["Listing_ID"]),
            year=int(row["year"]),
            make=row["make"],
            model=row["model"],
            trim=row["trim"],
            title=row["title"],
            description=row["description"],
            photo_url=row["photo_url"],
            price_aed=None if pd.isna(row["price_aed"]) else int(row["price_aed"]),
            mileage_km=None if pd.isna(row["mileage_km"]) else int(row["mileage_km"]),
            body_type=row["body_type"] if row["body_type"] else None,
            regional_spec=row["regional_spec"] if row["regional_spec"] else None,
            has_warranty=bool(row["has_warranty"]),
        )

    def search(
        self,
        make: Optional[str] = None,
        model: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        body_type: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        d = self.df

        if make:
            d = d[d["make"].str.contains(make, case=False, na=False)]
        if model:
            d = d[d["model"].str.contains(model, case=False, na=False)]
        if min_year:
            d = d[d["year"] >= min_year]
        if max_year:
            d = d[d["year"] <= max_year]
        if body_type:
            d = d[d["body_type"].str.contains(body_type, case=False, na=False)]
        if min_price:
            d = d[d["price_aed"].notna() & (d["price_aed"] >= min_price)]
        if max_price:
            d = d[d["price_aed"].notna() & (d["price_aed"] <= max_price)]

        if keyword:
            tokens = [t for t in re.split(r"\s+", keyword.lower()) if t]
            if tokens:
                mask = d["search_blob"].apply(lambda blob: all(t in blob for t in tokens))
                narrowed = d[mask]
                if narrowed.empty:
                    # relax to "any token matches" if exact all-token match yields nothing
                    mask_any = d["search_blob"].apply(lambda blob: any(t in blob for t in tokens))
                    narrowed = d[mask_any]
                d = narrowed

        d = d.head(limit)
        return [self._row_to_car(row).to_dict() for _, row in d.iterrows()]

    def stats(self) -> dict:
        return {
            "total_listings": len(self.df),
            "makes": sorted(self.df["make"].unique().tolist()),
            "year_range": [int(self.df["year"].min()), int(self.df["year"].max())],
            "listings_with_known_price": int(self.df["price_aed"].notna().sum()),
        }


inventory_store = InventoryStore()
