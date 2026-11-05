"""Financial identifier validation and check-digit verification.

Implements ISO 7064 Mod 97-10 (IBAN, LEI) and Luhn Mod 10 (CUSIP).
"""

from __future__ import annotations

import re

_ALPHANUM_MAP = {chr(i): str(i - ord("A") + 10) for i in range(ord("A"), ord("Z") + 1)}
_IBAN_REGEX = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4,30}$")
_LEI_REGEX = re.compile(r"^[A-Z0-9]{20}$")
_CUSIP_REGEX = re.compile(r"^[A-Z0-9*@#]{8}\d$")


def _iso7064_mod97_10(rearranged_str: str) -> bool:
    """Calculate ISO 7064 Mod 97-10 check on alphanumeric string."""
    digits: list[str] = []
    for ch in rearranged_str.upper():
        if ch in _ALPHANUM_MAP:
            digits.append(_ALPHANUM_MAP[ch])
        elif ch.isdigit():
            digits.append(ch)
        else:
            return False
    number_str = "".join(digits)
    # Piecewise modulo to avoid arbitrary large integer conversion overhead
    remainder = 0
    for ch in number_str:
        remainder = (remainder * 10 + int(ch)) % 97
    return remainder == 1


def validate_iban(iban: str) -> bool:
    """Validate an International Bank Account Number (IBAN) using ISO 7064 Mod 97-10.

    Guarantees 100% detection of single-character substitutions and adjacent transpositions.
    """
    clean = iban.replace(" ", "").replace("-", "").upper()
    if not _IBAN_REGEX.match(clean):
        return False
    # Move first 4 characters (country + check digits) to end
    rearranged = clean[4:] + clean[:4]
    return _iso7064_mod97_10(rearranged)


def validate_lei(lei: str) -> bool:
    """Validate a Legal Entity Identifier (LEI) using ISO 7064 Mod 97-10.

    20-character alphanumeric code; guarantees 100% single-character substitution detection.
    """
    clean = lei.strip().upper()
    if not _LEI_REGEX.match(clean):
        return False
    return _iso7064_mod97_10(clean)


def validate_cusip(cusip: str) -> bool:
    """Validate a 9-character CUSIP identifier using Luhn Mod 10 check digit."""
    clean = cusip.strip().upper()
    if not _CUSIP_REGEX.match(clean):
        return False

    total = 0
    for i, ch in enumerate(clean[:8]):
        if ch.isdigit():
            val = int(ch)
        elif ch in _ALPHANUM_MAP:
            val = int(_ALPHANUM_MAP[ch])
        elif ch == "*":
            val = 36
        elif ch == "@":
            val = 37
        elif ch == "#":
            val = 38
        else:  # pragma: no cover - regex guarantees valid characters
            return False

        if (i + 1) % 2 == 0:
            val *= 2
        total += (val // 10) + (val % 10)

    computed_check = (10 - (total % 10)) % 10
    actual_check = int(clean[8]) if clean[8].isdigit() else -1
    return computed_check == actual_check


# Backward compatibility alias
validate_cusip_check_digit = validate_cusip
