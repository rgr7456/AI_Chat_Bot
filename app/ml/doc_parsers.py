"""Field extraction for Indian government / KYC documents from OCR lines.

This layer is intentionally OCR-first and best-effort: it pulls the structured
fields a caller usually wants (number, name, dob, etc.) using labels + robust
regexes, and always returns `raw_lines` + a confidence so downstream business
validation can refine or reject. It does NOT perform authority verification
(NSDL / UIDAI / RBI) -- that is left to the caller.

Add a new document type by writing a `parse_*` function and registering it in
`PARSERS` + `_SIGNATURES`.

Format references:
  PAN     : 5 letters + 4 digits + 1 letter        e.g. ABCDE1234F
  Aadhaar : 12 digits (grouped 4-4-4), Verhoeff-checksummed
  IFSC    : 4 letters + '0' + 6 alphanumerics       e.g. HDFC0001234
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

# ----------------------------------------------------------------------------
# Shared regexes
# ----------------------------------------------------------------------------
RE_PAN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
RE_AADHAAR = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
RE_IFSC = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
RE_ACCOUNT = re.compile(r"\b(\d{9,18})\b")

# Unanchored variants, used as a fallback on space-stripped text when OCR has
# split a token (e.g. "SBIN 0001234"). No \b, so they survive concatenation.
RE_PAN_LOOSE = re.compile(r"([A-Z]{5}[0-9]{4}[A-Z])")
RE_IFSC_LOOSE = re.compile(r"([A-Z]{4}0[A-Z0-9]{6})")
RE_ACCOUNT_LOOSE = re.compile(r"(\d{9,18})")


def _find(anchored: re.Pattern, loose: re.Pattern, lines: List[str],
          upper: bool = False) -> Optional[str]:
    """Match `anchored` on the raw text first; fall back to `loose` on the
    space-stripped text. `upper` uppercases before matching (PAN/IFSC)."""
    text = "\n".join(lines)
    if upper:
        text = text.upper()
    m = anchored.search(text)
    if m:
        return m.group(1)
    m = loose.search(text.replace(" ", ""))
    return m.group(1) if m else None
RE_DOB = re.compile(r"\b(\d{2}[/\-.]\d{2}[/\-.]\d{4})\b")
RE_YOB = re.compile(r"(?:year of birth|yob)\s*[:\-]?\s*(\d{4})", re.I)
RE_GENDER = re.compile(r"\b(MALE|FEMALE|TRANSGENDER|Male|Female)\b")

# Lines that are document chrome, never a person's name.
_NOISE = re.compile(
    r"government|govt|india|income\s*tax|department|permanent|account|number|"
    r"unique|identification|authority|aadhaar|signature|date\s*of\s*birth|"
    r"father|male|female|dob|year|address|bank|branch|ifsc|account|cheque|"
    r"pay|rupees|www|http|\.com|\.in",
    re.I,
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _looks_like_name(s: str) -> bool:
    s = _clean(s)
    if len(s) < 3 or _NOISE.search(s) or any(c.isdigit() for c in s):
        return False
    letters = re.sub(r"[^A-Za-z]", "", s)
    # mostly alphabetic, 1-4 words
    return len(letters) >= 3 and 1 <= len(s.split()) <= 4 and letters.isascii()


def _join_names(lines: List[str], start: int, max_parts: int = 4) -> Optional[str]:
    """Join consecutive name-like lines from `start` into one uppercased name.

    Stops at the first non-name line (a label, PAN, date, etc.), so a name that
    OCR split across boxes is reassembled without swallowing the next field.
    """
    parts: List[str] = []
    for ln in lines[start:start + max_parts]:
        if _looks_like_name(ln):
            parts.append(_clean(ln))
        else:
            break
    return " ".join(parts).upper() if parts else None


def _first_dob(lines: List[str]) -> Optional[str]:
    for ln in lines:
        m = RE_DOB.search(ln)
        if m:
            return m.group(1).replace("-", "/").replace(".", "/")
    for ln in lines:
        m = RE_YOB.search(ln)
        if m:
            return m.group(1)
    return None


# ----------------------------------------------------------------------------
# Verhoeff checksum (UIDAI uses it for the 12th Aadhaar digit)
# ----------------------------------------------------------------------------
_V_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_V_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    """True if `number` (digits only) passes the Verhoeff checksum."""
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _V_D[c][_V_P[i % 8][int(ch)]]
    return c == 0


def mask_aadhaar(number: str) -> str:
    """Mask all but the last 4 digits (UIDAI requirement). -> 'XXXX XXXX 1234'."""
    digits = re.sub(r"\D", "", number or "")
    if len(digits) != 12:
        return "XXXX XXXX XXXX"
    return f"XXXX XXXX {digits[-4:]}"


# ----------------------------------------------------------------------------
# Per-document parsers.  Each returns a dict; `number` fields may be None.
# ----------------------------------------------------------------------------
def parse_pan(lines: List[str]) -> Dict:
    pan = _find(RE_PAN, RE_PAN_LOOSE, lines, upper=True)
    name, father = None, None
    # PAN layout: Name label, then name; Father's Name label, then father.
    # OCR may split a multi-word name across boxes, so gather consecutive
    # name-like lines after each label until the next label / PAN / date.
    for i, ln in enumerate(lines):
        low = ln.lower()
        joined = _join_names(lines, i + 1)
        if "father" in low and joined:
            father = joined
        elif low.strip() in ("name", "/ name") and joined:
            name = joined
    if not name:
        cands = [l for l in lines if _looks_like_name(l)]
        if cands:
            name = _clean(cands[0]).upper()
        if not father and len(cands) > 1:
            father = _clean(cands[1]).upper()
    return {
        "document_type": "PAN",
        "pan_number": pan,
        "name": name,
        "father_name": father,
        "date_of_birth": _first_dob(lines),
    }


def parse_aadhaar(lines: List[str], mask: bool = True) -> Dict:
    text = "\n".join(lines)
    raw_num, checksum_ok = None, False
    for m in RE_AADHAAR.finditer(text):
        cand = re.sub(r"\s", "", m.group(1))
        if len(cand) == 12:
            raw_num = cand
            checksum_ok = verhoeff_valid(cand)
            if checksum_ok:
                break  # prefer a checksum-valid match
    gender = None
    g = RE_GENDER.search(text)
    if g:
        gender = g.group(1).upper()
    # Name usually sits just above the DOB / gender line.
    name = None
    for i, ln in enumerate(lines):
        if RE_DOB.search(ln) or RE_YOB.search(ln) or RE_GENDER.search(ln):
            for j in range(i - 1, -1, -1):
                if _looks_like_name(lines[j]):
                    name = _clean(lines[j]).upper()
                    break
            break
    if not name:
        cands = [l for l in lines if _looks_like_name(l)]
        name = _clean(cands[0]).upper() if cands else None

    out = {
        "document_type": "AADHAAR",
        "name": name,
        "date_of_birth": _first_dob(lines),
        "gender": gender,
        "checksum_valid": checksum_ok,
    }
    if mask:
        out["aadhaar_number_masked"] = mask_aadhaar(raw_num) if raw_num else None
    else:
        out["aadhaar_number"] = raw_num
    return out


def parse_bank(lines: List[str]) -> Dict:
    # IFSC: prefer the line mentioning "IFSC" (targeted, avoids false matches
    # like 'NUMBER000...'); else an anchored match over the raw text.
    ifsc = None
    for ln in lines:
        if re.search(r"ifsc", ln, re.I):
            after = re.split(r"ifsc\s*(?:code)?\s*[:\-]?", ln, flags=re.I)[-1]
            m = RE_IFSC_LOOSE.search(after.upper().replace(" ", ""))
            if m:
                ifsc = m.group(1)
                break
    if not ifsc:
        m = RE_IFSC.search("\n".join(lines).upper())
        ifsc = m.group(1) if m else None
    # Account number: exclude any IFSC's trailing digits first. Digit-only, so
    # the despaced loose fallback is safe here.
    acct_lines = [RE_IFSC.sub("", l) for l in lines]
    account = _find(RE_ACCOUNT, RE_ACCOUNT_LOOSE, acct_lines)
    bank_name = next((_clean(l) for l in lines if re.search(r"bank", l, re.I)), None)
    holder = None
    for i, ln in enumerate(lines):
        if re.search(r"\bname\b", ln, re.I):
            tail = _clean(RE_IFSC.sub("", ln))
            tail = _clean(re.split(r"name\s*[:\-]?", tail, flags=re.I)[-1])
            if _looks_like_name(tail):
                holder = tail.upper()
                break
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _looks_like_name(nxt):
                holder = _clean(nxt).upper()
                break
    return {
        "document_type": "BANK",
        "account_number": account,
        "ifsc": ifsc,
        "bank_name": bank_name,
        "account_holder_name": holder,
    }


# ----------------------------------------------------------------------------
# Registry + auto-detection
# ----------------------------------------------------------------------------
PARSERS: Dict[str, Callable[..., Dict]] = {
    "pan": parse_pan,
    "aadhaar": parse_aadhaar,
    "bank": parse_bank,
}

# Ordered: most specific signature first.
_SIGNATURES = [
    ("pan", lambda t: bool(RE_PAN.search(t)) or "income tax" in t.lower()),
    ("aadhaar", lambda t: bool(RE_AADHAAR.search(t))
        or "aadhaar" in t.lower() or "unique identification" in t.lower()),
    ("bank", lambda t: bool(RE_IFSC.search(t)) or "ifsc" in t.lower()),
]


def detect_doc_type(text: str) -> Optional[str]:
    up = text.upper().replace(" ", "")
    for name, sig in _SIGNATURES:
        try:
            if sig(up if name != "aadhaar" else text):
                return name
        except Exception:
            continue
    return None


def parse(doc_type: str, lines: List[str], *, mask_aadhaar: bool = True) -> Dict:
    doc_type = (doc_type or "").lower()
    if doc_type not in PARSERS:
        raise ValueError(f"unsupported document type: {doc_type!r}")
    if doc_type == "aadhaar":
        return parse_aadhaar(lines, mask=mask_aadhaar)
    return PARSERS[doc_type](lines)
