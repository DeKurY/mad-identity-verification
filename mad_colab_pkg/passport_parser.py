import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _text(value: Any) -> str:
    return str(value or "").strip()


def digits_only(value: Any) -> str:
    return re.sub(r"\D+", "", _text(value))


LATIN_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У", "I": "И", "J": "Л", "L": "Л",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
})

COMMON_CORRECTIONS = {
    "ПLО": "ПО",
    "ВЛАЛИМИРСКОЙ": "ВЛАДИМИРСКОЙ",
    "БЛАДИМИРСКАЯ": "ВЛАДИМИРСКАЯ",
    "ОВЛАСТИ": "ОБЛАСТИ",
    "ОКЛ": "ОБЛ",
    "УИНД": "УМВД",
    "P H": "Р-Н",
}


def normalize_cyrillic(value: Any) -> str:
    t = _text(value).translate(LATIN_TO_CYR)
    t = t.replace("Ё", "Е").replace("ё", "е")
    t = re.sub(r"[^А-Яа-я0-9.\-/ <]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip().upper()
    for bad, good in COMMON_CORRECTIONS.items():
        t = t.replace(bad, good)
    return t


def normalize_name(value: Any) -> str:
    t = normalize_cyrillic(value)
    t = re.sub(r"[^А-Я -]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_date(value: Any) -> Optional[str]:
    if not value:
        return None
    s = _text(value)
    # Already ISO
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s, "%Y-%m-%d").date().isoformat()
    except Exception:
        pass
    m = re.search(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", s)
    if not m:
        return None
    dd, mm, yy = m.groups()
    if len(yy) == 2:
        yy = "19" + yy
    try:
        return datetime(int(yy), int(mm), int(dd)).date().isoformat()
    except Exception:
        return None


def display_date_to_iso(value: Any) -> Optional[str]:
    return normalize_date(value)


def _score(it: Dict[str, Any]) -> float:
    try:
        return float(it.get("score") or 0.0)
    except Exception:
        return 0.0


def _tag(it: Dict[str, Any]) -> str:
    return _text(it.get("tag")).lower()


def _cy(it: Dict[str, Any]) -> float:
    try:
        return float(it.get("cy") or 0.0)
    except Exception:
        return 0.0


def parse_date_candidates(items: List[Dict[str, Any]], kind: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for it in items:
        txt = _text(it.get("text"))
        tag = _tag(it)
        for m in re.finditer(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", txt):
            dd, mm, yy = m.groups()
            yy = "19" + yy if len(yy) == 2 else yy
            try:
                d, mo, y = int(dd), int(mm), int(yy)
                dt = datetime(y, mo, d).date()
            except Exception:
                continue
            if y < 1900 or y > 2035:
                continue
            weight = _score(it)
            if kind == "issue_date":
                if "top" in tag:
                    weight += 0.90
                if tag == "full":
                    weight += 0.40
                if y < 1997 or y > 2030:
                    weight -= 0.60
            if kind == "birth_date":
                if "bottom" in tag or "birth" in tag:
                    weight += 1.10
                if y > datetime.now().year - 10:
                    weight -= 1.00
            candidates.append({"value": dt.isoformat(), "display": dt.strftime("%d.%m.%Y"), "weight": weight, "tag": tag, "text": txt})
    candidates.sort(key=lambda x: x["weight"], reverse=True)
    return (candidates[0]["value"] if candidates else None), candidates


def parse_department_code(items: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for it in items:
        txt = _text(it.get("text"))
        tag = _tag(it)
        # Strictly prefer top/full/department ROI. Avoid treating vertical passport number as XXX-XXX.
        if "right" in tag and "department" not in tag:
            continue
        for m in re.finditer(r"(?<!\d)(\d{3})[- ]?(\d{3})(?!\d)", txt):
            value = f"{m.group(1)}-{m.group(2)}"
            weight = _score(it)
            if "-" in txt:
                weight += 0.80
            if "top" in tag or "department" in tag:
                weight += 0.80
            candidates.append({"value": value, "weight": weight, "tag": tag, "text": txt})
    candidates.sort(key=lambda x: x["weight"], reverse=True)
    return (candidates[0]["value"] if candidates else None), candidates


MRZ_TO_CYR = {
    "A": "А", "B": "Б", "C": "Ц", "D": "Д", "E": "Е", "F": "Ф", "G": "Г", "H": "Х",
    "I": "И", "J": "Й", "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О", "P": "П",
    "Q": "К", "R": "Р", "S": "С", "T": "Т", "U": "У", "V": "В", "W": "В", "X": "КС",
    "Y": "Ы", "Z": "З",
}
MRZ_DIGIT_FIX = str.maketrans({"0": "O", "1": "I", "3": "C", "5": "S", "8": "B"})


def mrz_word_to_cyr(word: str) -> str:
    w = re.sub(r"[^A-Z0-9]", "", word.upper()).translate(MRZ_DIGIT_FIX)
    replacements = {
        "SHCH": "Щ", "SCH": "Щ", "YO": "Е", "YE": "Е", "YU": "Ю", "YA": "Я",
        "ZH": "Ж", "KH": "Х", "TS": "Ц", "CH": "Ч", "SH": "Ш",
    }
    out = ""
    i = 0
    keys = sorted(replacements, key=len, reverse=True)
    while i < len(w):
        matched = False
        for k in keys:
            if w.startswith(k, i):
                out += replacements[k]
                i += len(k)
                matched = True
                break
        if matched:
            continue
        out += MRZ_TO_CYR.get(w[i], "")
        i += 1
    # Frequent OCR loss in Russian patronymics.
    out = re.sub(r"ЕВИ[СЦК]$", "ЕВИЧ", out)
    out = re.sub(r"И[СЦК]$", "ИЧ", out)
    return out


def parse_mrz_names(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for it in items:
        raw = _text(it.get("text")).upper().replace(" ", "")
        if "<" not in raw or not ("P" in raw or "Р" in raw):
            continue
        latinized = raw.translate(str.maketrans({"А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X"}))
        m = re.search(r"P.?RUS(.+)", latinized)
        if not m:
            continue
        rest = m.group(1).strip("<")
        parts = [p for p in rest.split("<") if p]
        if len(parts) < 2:
            continue
        value = {
            "surname": mrz_word_to_cyr(parts[0]),
            "firstname": mrz_word_to_cyr(parts[1]),
            "patronymic": mrz_word_to_cyr(parts[2]) if len(parts) >= 3 else None,
            "weight": _score(it) + 0.5,
            "raw": raw,
            "parts": parts,
        }
        if best is None or value["weight"] > best["weight"]:
            best = value
    return best


def _is_bad_passport_number(value: str) -> bool:
    if len(value) != 6:
        return True
    if len(set(value)) == 1:
        return True
    # Fragments that often come from codes/MRZ checks; keep it conservative.
    if value.startswith("000"):
        return True
    return False


def parse_passport_id(items: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    number_scores: Dict[str, float] = defaultdict(float)
    series_scores: Dict[str, float] = defaultdict(float)
    events: List[Dict[str, Any]] = []

    for it in items:
        txt = _text(it.get("text"))
        tag = _tag(it)
        digits = digits_only(txt)
        upper = txt.upper().replace(" ", "")
        score = _score(it) or 0.5
        source_weight = 1.0
        if "right" in tag or "vertical" in tag:
            source_weight += 2.0
        if "rot90" in tag or "rot270" in tag:
            source_weight += 0.8

        # MRZ second line: first four digits are the most useful signal for series.
        if "<" in upper and len(digits) >= 10:
            series = digits[:4]
            series_scores[series] += score * 8.0
            events.append({"type": "series_mrz_first4", "value": series, "weight": score * 8.0, "tag": tag, "text": txt})
            # Use number windows as weak evidence only; MRZ OCR can shift digits.
            for i in range(4, min(len(digits) - 5, 12)):
                num = digits[i:i+6]
                if not _is_bad_passport_number(num):
                    number_scores[num] += score * 0.4
                    events.append({"type": "number_mrz_window", "value": num, "weight": score * 0.4, "tag": tag, "text": txt})

        if len(digits) == 10:
            series = digits[:4]
            number = digits[4:]
            series_scores[series] += score * source_weight
            if not _is_bad_passport_number(number):
                number_scores[number] += score * source_weight
            events.append({"type": "id10", "value": digits, "weight": score * source_weight, "tag": tag, "text": txt})

        if len(digits) == 8 and ("right" in tag or "vertical" in tag):
            num = digits[-6:]
            if not _is_bad_passport_number(num):
                number_scores[num] += score * (source_weight + 0.5)
                events.append({"type": "number_from_8", "value": num, "weight": score * (source_weight + 0.5), "tag": tag, "text": txt})

        if len(digits) == 6:
            if "right" in tag or "vertical" in tag:
                if not _is_bad_passport_number(digits):
                    number_scores[digits] += score * source_weight
                    events.append({"type": "number6_right", "value": digits, "weight": score * source_weight, "tag": tag, "text": txt})
            elif score > 0.95 and not _is_bad_passport_number(digits):
                # Weak evidence from full page; vertical number can leak into full OCR.
                number_scores[digits] += score * 0.4
                events.append({"type": "number6_weak", "value": digits, "weight": score * 0.4, "tag": tag, "text": txt})

        if len(digits) == 4 and ("right" in tag or "vertical" in tag):
            # Right-zone four digits are useful, but not stronger than MRZ first4.
            series_scores[digits] += score * source_weight
            events.append({"type": "series4_right", "value": digits, "weight": score * source_weight, "tag": tag, "text": txt})

    series = max(series_scores.items(), key=lambda x: x[1])[0] if series_scores else None
    number = max(number_scores.items(), key=lambda x: x[1])[0] if number_scores else None

    debug = {
        "series_scores": dict(sorted(series_scores.items(), key=lambda x: -x[1])[:10]),
        "number_scores": dict(sorted(number_scores.items(), key=lambda x: -x[1])[:10]),
        "events": sorted(events, key=lambda x: -x["weight"])[:80],
    }
    return series, number, debug


def extract_fio(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
    mrz = parse_mrz_names(items)
    # Bottom/field OCR is usually better for firstname/patronymic if MRZ got glued by OCR.
    candidates: List[Dict[str, Any]] = []
    for it in items:
        tag = _tag(it)
        if "right" in tag:
            continue
        if not (tag == "bottom" or tag in {"surname", "firstname", "name", "patronymic"}):
            continue
        raw = _text(it.get("text"))
        if "<" in raw or re.search(r"\d", raw):
            continue
        norm = normalize_name(raw)
        if len(norm) < 4:
            continue
        bad_markers = ["МУЖ", "ЖЕН", "ОБЛ", "Р-Н", "РОСС", "ФЕДЕРАЦ"]
        if any(marker in norm for marker in bad_markers):
            continue
        candidates.append({"raw": raw, "value": norm, "score": _score(it), "cy": _cy(it), "tag": tag})

    candidates.sort(key=lambda x: (x["cy"], -x["score"]))
    values = []
    for c in candidates:
        if c["value"] not in values:
            values.append(c["value"])

    surname = mrz.get("surname") if mrz else None
    firstname = None
    patronymic = None

    # Prefer direct ROI labels if present.
    by_tag = {c["tag"]: c["value"] for c in candidates}
    surname = by_tag.get("surname") or surname
    firstname = by_tag.get("firstname") or by_tag.get("name") or None
    patronymic = by_tag.get("patronymic") or None

    # Otherwise use ordered bottom lines. If surname came from MRZ, skip the first bottom line as likely surname.
    if values:
        ordered = values[:]
        if surname and ordered and SequenceMatcher(None, surname, ordered[0]).ratio() < 0.55:
            # The first bottom line may be a noisy surname after OCR substitutions.
            ordered_for_names = ordered[1:]
        elif surname:
            ordered_for_names = ordered[1:]
        else:
            surname = ordered[0]
            ordered_for_names = ordered[1:]
        if not firstname and ordered_for_names:
            firstname = ordered_for_names[0]
        if not patronymic and len(ordered_for_names) >= 2:
            patronymic = ordered_for_names[1]

    # MRZ fallback if it parsed cleanly into separate fields.
    if mrz:
        if not surname:
            surname = mrz.get("surname")
        if not firstname and mrz.get("firstname") and len(mrz.get("firstname", "")) <= 18:
            firstname = mrz.get("firstname")
        if not patronymic and mrz.get("patronymic") and len(mrz.get("patronymic", "")) <= 20:
            patronymic = mrz.get("patronymic")

    return {"surname": surname, "firstname": firstname, "patronymic": patronymic}, {"mrz": mrz, "line_candidates": candidates}


def extract_issued_by(items: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for it in items:
        tag = _tag(it)
        raw = _text(it.get("text"))
        norm = normalize_cyrillic(raw)
        if not ("top" in tag or tag == "full" or "issued" in tag):
            continue
        if any(m in norm for m in ["МВД", "УМВД", "УФМС", "ОВД", "ОТДЕЛ"]):
            candidates.append({"value": norm, "weight": _score(it) + 0.8, "tag": tag, "cy": _cy(it), "text": raw})
        elif "ОБЛАСТ" in norm and len(norm) <= 30:
            candidates.append({"value": norm, "weight": _score(it) + 0.2, "tag": tag, "cy": _cy(it), "text": raw})
    # Keep useful unique lines and rebuild in natural order.
    selected: List[Dict[str, Any]] = []
    seen = set()
    for c in sorted(candidates, key=lambda x: (-x["weight"], x["cy"])):
        if c["value"] in seen:
            continue
        seen.add(c["value"])
        selected.append(c)
    selected = sorted(selected[:4], key=lambda x: x["cy"])
    value = " ".join(c["value"] for c in selected) if selected else None
    return value, selected


def extract_gender(items: List[Dict[str, Any]]) -> Optional[str]:
    for it in items:
        norm = normalize_cyrillic(it.get("text"))
        if "МУЖ" in norm or "МУХ" in norm or norm == "М":
            return "МУЖ"
        if "ЖЕН" in norm or norm == "Ж":
            return "ЖЕН"
    return None


def extract_birth_place(items: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    for it in items:
        tag = _tag(it)
        if "right" in tag:
            continue
        if not (tag == "bottom" or "birth_place" in tag or tag == "full"):
            continue
        raw = _text(it.get("text"))
        if "<" in raw or re.search(r"\d", raw):
            continue
        norm = normalize_cyrillic(raw)
        if len(norm) < 3:
            continue
        # Standalone ОБЛАСТИ in the upper page is issuer, not birth place.
        if norm == "ОБЛАСТИ":
            continue
        if any(x in norm for x in ["МВД", "УМВД", "УФМС", "РОССИИ", "ФЕДЕРАЦ"]):
            continue
        markers = ["Г.", "С.", "Д.", "ПОС", "Р-Н", "РАЙОН", "ОБЛ", "КРАЙ", "РЕСП", "МОСК", "ВЛАДИМИР"]
        if any(m in norm for m in markers):
            selected.append({"value": norm, "score": _score(it), "cy": _cy(it), "tag": tag, "text": raw})
    selected.sort(key=lambda x: (x["cy"], -x["score"]))
    values: List[str] = []
    for c in selected:
        v = c["value"]
        if v not in values:
            values.append(v)
    value = ", ".join(values[:4]) if values else None
    return value, selected


def field_confidence(value: Any, candidates: Iterable[Dict[str, Any]] = ()) -> str:
    if value in (None, ""):
        return "missing"
    weights = []
    for c in candidates or []:
        try:
            weights.append(float(c.get("weight", c.get("score", 0))))
        except Exception:
            pass
    if weights and max(weights) >= 3:
        return "high"
    if weights and max(weights) >= 1:
        return "medium"
    return "medium"


def parse_passport_data(raw_items: Any) -> Dict[str, Any]:
    if isinstance(raw_items, dict) and "items" in raw_items:
        items = raw_items.get("items") or []
    elif isinstance(raw_items, list):
        items = raw_items
    else:
        items = []

    issue_date, issue_candidates = parse_date_candidates(items, "issue_date")
    birth_date, birth_candidates = parse_date_candidates(items, "birth_date")
    department_code, department_candidates = parse_department_code(items)
    passport_series, passport_number, id_debug = parse_passport_id(items)
    fio, fio_debug = extract_fio(items)
    issued_by, issued_by_candidates = extract_issued_by(items)
    birth_place, birth_place_candidates = extract_birth_place(items)
    gender = extract_gender(items)

    full_name_parts = [fio.get("surname"), fio.get("firstname"), fio.get("patronymic")]
    full_name = " ".join([p for p in full_name_parts if p]) or None

    passport_data = {
        "issued_by": issued_by,
        "issue_date": issue_date,
        "department_code": department_code,
        "surname": fio.get("surname"),
        "firstname": fio.get("firstname"),
        "patronymic": fio.get("patronymic"),
        "full_name": full_name,
        "gender": gender,
        "birth_date": birth_date,
        "birth_place": birth_place,
        "passport_series": passport_series,
        "passport_number": passport_number,
    }
    confidence = {
        "issue_date": field_confidence(issue_date, issue_candidates),
        "birth_date": field_confidence(birth_date, birth_candidates),
        "department_code": field_confidence(department_code, department_candidates),
        "passport_series": "high" if passport_series and id_debug.get("series_scores", {}).get(passport_series, 0) >= 3 else "medium" if passport_series else "missing",
        "passport_number": "high" if passport_number and id_debug.get("number_scores", {}).get(passport_number, 0) >= 3 else "medium" if passport_number else "missing",
        "full_name": "medium" if full_name else "missing",
    }
    debug = {
        "id": id_debug,
        "fio": fio_debug,
        "issue_date_candidates": issue_candidates[:10],
        "birth_date_candidates": birth_candidates[:10],
        "department_candidates": department_candidates[:10],
        "issued_by_candidates": issued_by_candidates[:10],
        "birth_place_candidates": birth_place_candidates[:10],
        "items_count": len(items),
    }
    return {"passport_data": passport_data, "confidence": confidence, "debug": debug}
