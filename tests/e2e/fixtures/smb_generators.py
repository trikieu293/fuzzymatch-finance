"""Deterministic synthetic SMB noise and entity generators for record linkage benchmarks."""

from __future__ import annotations

import datetime
import random

import numpy as np

LEGAL_SUFFIXES: list[str] = [
    "Inc",
    "Inc.",
    ", Inc.",
    "LLC",
    "L.L.C.",
    ", LLC",
    "Ltd",
    "Ltd.",
    "Limited",
    "Corp",
    "Corp.",
    "Corporation",
    "GmbH",
    "GmbH & Co. KG",
    "S.A.",
    "Co",
    "Co.",
    "Company",
]

BASE_BUSINESS_NAMES: list[str] = [
    "Acme Industrial",
    "Apex Logistics",
    "Beacon Global",
    "BlueSky Technologies",
    "Cascade Supply",
    "Crestview Partners",
    "Delta Dynamics",
    "Eagle Eye Freight",
    "Falcon Distribution",
    "Frontier Consulting",
    "Global Vertex",
    "Harbor Light Shipping",
    "Horizon Energy",
    "Imperial Fasteners",
    "Ironclad Manufacturing",
    "Keystone Software",
    "Liberty Warehousing",
    "Meridian Health",
    "Nexus Telecom",
    "NorthStar Logistics",
    "Oakwood Packaging",
    "OmniSource Materials",
    "Pacific Coast Media",
    "Peak Performance",
    "Pinnacle Advisory",
    "Precision Machining",
    "Premier Fleet Services",
    "Quantum Solutions",
    "Redwood Timber",
    "Silverline Transit",
    "Summit Chemical",
    "Terraform Geo",
    "Titan Security",
    "Trident Marine",
    "United Standard",
    "Vanguard Electronics",
    "Velocity Express",
    "Vertex Bioanalytics",
    "Waypoint Navigation",
    "Zenith Capital",
]

BASE_CITIES: list[tuple[str, str, str]] = [
    ("New York", "NY", "10001"),
    ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"),
    ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"),
    ("Philadelphia", "PA", "19101"),
    ("San Antonio", "TX", "78201"),
    ("San Diego", "CA", "92101"),
    ("Dallas", "TX", "75201"),
    ("San Jose", "CA", "95101"),
]


def corrupt_typo(text: str, rng: random.Random, num_edits: int = 1) -> str:
    """Inject typographical errors (insert, delete, replace, transpose) into text."""
    if not text:
        return text
    chars = list(text)
    for _ in range(num_edits):
        if len(chars) <= 1:
            break
        op = rng.choice(["del", "ins", "sub", "trans"])
        pos = rng.randint(0, len(chars) - 1)
        if op == "del" and len(chars) > 3:
            del chars[pos]
        elif op == "ins":
            chars.insert(pos, rng.choice("abcdefghijklmnopqrstuvwxyz"))
        elif op == "sub":
            chars[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        elif op == "trans" and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def add_legal_suffix(base_name: str, rng: random.Random) -> str:
    """Append or modify legal entity suffix for a business name."""
    suffix = rng.choice(LEGAL_SUFFIXES)
    sep = " " if not suffix.startswith(",") else ""
    return f"{base_name}{sep}{suffix}"


def corrupt_amount(amount: float, rng: random.Random, max_delta: float = 0.50) -> float:
    """Inject continuous numerical proximity perturbation (rounding error, small fee)."""
    delta = rng.uniform(-max_delta, max_delta)
    return round(max(0.01, amount + delta), 2)


def corrupt_date(base_date: datetime.date, rng: random.Random, max_days: int = 5) -> datetime.date:
    """Inject date lag or lead within a bounded day window."""
    day_shift = rng.randint(-max_days, max_days)
    return base_date + datetime.timedelta(days=day_shift)


def format_date_randomly(dt: datetime.date, rng: random.Random) -> str:
    """Format a date using common SMB format variations."""
    fmt = rng.choice(["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"])
    return dt.strftime(fmt)


class SMBDataGenerator:
    """Deterministic generator for synthetic SMB entity resolution benchmarks."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_canonical_entities(
        self, count: int = 40
    ) -> list[dict[str, str | float | datetime.date]]:
        """Generate canonical ground-truth entities."""
        entities: list[dict[str, str | float | datetime.date]] = []
        base_names = BASE_BUSINESS_NAMES * ((count // len(BASE_BUSINESS_NAMES)) + 2)
        base_names = base_names[:count]

        for idx, base_name in enumerate(base_names):
            city, state, zip_code = self.rng.choice(BASE_CITIES)
            suffix = self.rng.choice(LEGAL_SUFFIXES)
            full_name = f"{base_name} {suffix}"
            tax_id = f"{self.rng.randint(10, 99)}-{self.rng.randint(1000000, 9999999)}"
            base_date = datetime.date(2025, 1, 1) + datetime.timedelta(
                days=self.rng.randint(0, 360)
            )
            base_amount = round(self.rng.uniform(150.0, 15000.0), 2)

            entities.append(
                {
                    "canonical_id": f"ENT-{idx:04d}",
                    "base_name": base_name,
                    "full_name": full_name,
                    "tax_id": tax_id,
                    "city": city,
                    "state": state,
                    "zip_code": zip_code,
                    "base_date": base_date,
                    "base_amount": base_amount,
                }
            )
        return entities
