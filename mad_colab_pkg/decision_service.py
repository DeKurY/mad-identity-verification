from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, Optional, Tuple

from .passport_parser import digits_only, normalize_date, normalize_name


def _ratio(a: str, b: str) -> float:
    a = normalize_name(a)
    b = normalize_name(b)
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz.fuzz import token_sort_ratio
        return float(token_sort_ratio(a, b)) / 100.0
    except Exception:
        return SequenceMatcher(None, a, b).ratio()


def profile_full_name(profile: Dict[str, Any]) -> str:
    return normalize_name(profile.get("full_name") or " ".join([
        str(profile.get("last_name") or ""),
        str(profile.get("first_name") or ""),
        str(profile.get("middle_name") or ""),
    ]))


def compute_data_match_score(profile: Dict[str, Any], passport_data: Dict[str, Any]) -> Dict[str, Any]:
    weights = {
        "passport_number": 0.35,
        "passport_series": 0.25,
        "birth_date": 0.25,
        "full_name": 0.15,
    }
    details: Dict[str, Any] = {}
    score = 0.0

    profile_number = digits_only(profile.get("passport_number"))[-6:]
    input_number = digits_only(passport_data.get("passport_number"))[-6:]
    number_match = bool(profile_number and input_number and profile_number == input_number)
    details["passport_number"] = {"match": number_match, "profile": profile_number, "input": input_number, "weight": weights["passport_number"]}
    if number_match:
        score += weights["passport_number"]

    profile_series = digits_only(profile.get("passport_series"))[-4:]
    input_series = digits_only(passport_data.get("passport_series"))[-4:]
    series_match = bool(profile_series and input_series and profile_series == input_series)
    details["passport_series"] = {"match": series_match, "profile": profile_series, "input": input_series, "weight": weights["passport_series"]}
    if series_match:
        score += weights["passport_series"]

    profile_birth = normalize_date(profile.get("birth_date"))
    input_birth = normalize_date(passport_data.get("birth_date"))
    birth_match = bool(profile_birth and input_birth and profile_birth == input_birth)
    details["birth_date"] = {"match": birth_match, "profile": profile_birth, "input": input_birth, "weight": weights["birth_date"]}
    if birth_match:
        score += weights["birth_date"]

    profile_name = profile_full_name(profile)
    input_name = normalize_name(passport_data.get("full_name") or " ".join([
        str(passport_data.get("surname") or ""),
        str(passport_data.get("firstname") or ""),
        str(passport_data.get("patronymic") or ""),
    ]))
    name_ratio = _ratio(profile_name, input_name)
    name_points = weights["full_name"] * name_ratio
    details["full_name"] = {"ratio": name_ratio, "profile": profile_name, "input": input_name, "weight": weights["full_name"], "points": name_points}
    score += name_points

    return {"score": round(float(score), 4), "details": details}


def choose_best_profile(profiles: Iterable[Dict[str, Any]], passport_data: Dict[str, Any]) -> Dict[str, Any]:
    best_profile = None
    best_score = -1.0
    best_details = {}
    for profile in profiles:
        result = compute_data_match_score(profile, passport_data)
        if result["score"] > best_score:
            best_profile = profile
            best_score = result["score"]
            best_details = result["details"]
    return {"profile": best_profile, "score": max(best_score, 0.0), "details": best_details}


def face_status(distance: Optional[float], accept_threshold: float, review_threshold: float) -> str:
    if distance is None:
        return "missing"
    if distance <= accept_threshold:
        return "accept"
    if distance <= review_threshold:
        return "review"
    return "reject"


def final_decision(
    data_score: float,
    passport_reference_distance: Optional[float],
    selfie_passport_distance: Optional[float],
    selfie_reference_distance: Optional[float],
    accept_threshold: float = 0.32,
    review_threshold: float = 0.43,
    data_accept_threshold: float = 0.80,
    data_review_threshold: float = 0.55,
) -> Dict[str, Any]:
    passport_reference_status = face_status(passport_reference_distance, accept_threshold, review_threshold)
    selfie_passport_status = face_status(selfie_passport_distance, accept_threshold, review_threshold)
    selfie_reference_status = face_status(selfie_reference_distance, accept_threshold, review_threshold)

    statuses = [passport_reference_status, selfie_passport_status]
    if passport_reference_status == "missing" and selfie_reference_status != "missing":
        statuses = [selfie_reference_status]

    if data_score >= data_accept_threshold and all(s == "accept" for s in statuses):
        decision = "ACCEPT"
    elif any(s == "reject" for s in statuses):
        decision = "REJECT"
    elif data_score < data_review_threshold:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    return {
        "final_decision": decision,
        "data_verified": data_score >= data_accept_threshold,
        "passport_photo_verified": passport_reference_status == "accept",
        "selfie_verified": selfie_passport_status == "accept" or selfie_reference_status == "accept",
        "face_statuses": {
            "passport_reference": passport_reference_status,
            "selfie_passport": selfie_passport_status,
            "selfie_reference": selfie_reference_status,
        },
    }
