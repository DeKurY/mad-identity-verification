import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from .decision_service import choose_best_profile, final_decision
from .face_service import cosine_distance, safe_embedding
from .identity_store import IdentityStore
from .image_utils import auto_normalize_passport, make_passport_crops
from .passport_parser import digits_only, normalize_date, normalize_name, parse_passport_data
from .text_classifier import classify_ocr_items


@dataclass
class Settings:
    base_dir: Path
    storage_dir: Path
    artifacts_dir: Path
    supabase_url: str = ""
    supabase_key: str = ""
    ocr_device: str = "cpu"
    ocr_worker_timeout_sec: int = 900
    ocr_quality_presets: bool = False
    face_model_name: str = "Facenet512"
    face_detector_backend: str = "retinaface"
    face_accept_threshold: float = 0.32
    face_review_threshold: float = 0.43
    data_accept_threshold: float = 0.80
    data_review_threshold: float = 0.55
    max_upload_mb: int = 20
    store_debug_artifacts: bool = False


ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_settings() -> Settings:
    base_dir = Path(os.getenv("MAD_BASE_DIR", "/content/mad_identity_verification"))
    storage_dir = Path(os.getenv("APP_STORAGE_DIR", str(base_dir / "storage")))
    artifacts_dir = Path(os.getenv("APP_ARTIFACTS_DIR", str(base_dir / "artifacts")))
    storage_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        base_dir=base_dir,
        storage_dir=storage_dir,
        artifacts_dir=artifacts_dir,
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_key=os.getenv("SUPABASE_KEY", ""),
        ocr_device=os.getenv("OCR_DEVICE", "cpu"),
        ocr_worker_timeout_sec=_int_env("OCR_WORKER_TIMEOUT_SEC", 900),
        ocr_quality_presets=_bool_env("OCR_QUALITY_PRESETS", False),
        face_model_name=os.getenv("FACE_MODEL_NAME", "Facenet512"),
        face_detector_backend=os.getenv("FACE_DETECTOR_BACKEND", "retinaface"),
        face_accept_threshold=_float_env("FACE_ACCEPT_THRESHOLD", 0.32),
        face_review_threshold=_float_env("FACE_REVIEW_THRESHOLD", 0.43),
        data_accept_threshold=_float_env("DATA_ACCEPT_THRESHOLD", 0.80),
        data_review_threshold=_float_env("DATA_REVIEW_THRESHOLD", 0.55),
        max_upload_mb=_int_env("MAX_UPLOAD_MB", 20),
        store_debug_artifacts=_bool_env("STORE_DEBUG_ARTIFACTS", False),
    )


def get_store(settings: Optional[Settings] = None) -> IdentityStore:
    settings = settings or get_settings()
    return IdentityStore(settings.base_dir, settings.supabase_url, settings.supabase_key)


def _save_input_file(input_path: str, subdir: str, settings: Settings) -> Path:
    """Validate and copy a Gradio-uploaded image into private runtime storage.

    The app works with passport and selfie images, so uploads are treated as
    sensitive. Only common image extensions are accepted, the size is limited,
    and filenames are replaced with UUIDs before saving. This prevents accidental
    disclosure of original filenames and reduces risk from unexpected file types.
    """
    if not input_path:
        raise ValueError("No input file")

    src = Path(input_path).expanduser().resolve(strict=True)
    if not src.is_file():
        raise ValueError("Input path is not a file")

    suffix = src.suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_SUFFIXES))
        raise ValueError(f"Unsupported image type: {suffix or '<empty>'}. Allowed: {allowed}")

    max_bytes = max(int(settings.max_upload_mb), 1) * 1024 * 1024
    if src.stat().st_size > max_bytes:
        raise ValueError(f"Input file is too large. Limit: {settings.max_upload_mb} MB")

    safe_subdir = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in subdir)[:64] or "uploads"
    out_dir = settings.storage_dir / "uploads" / safe_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex}{suffix}"
    shutil.copyfile(src, out_path)
    try:
        out_path.chmod(0o600)
    except Exception:
        pass
    return out_path


def run_paddle_ocr(passport_image_path: Path, settings: Optional[Settings] = None) -> Dict[str, Any]:
    settings = settings or get_settings()
    work_dir = settings.storage_dir / "ocr_jobs" / uuid.uuid4().hex
    crops_dir = work_dir / "crops"
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = auto_normalize_passport(passport_image_path, work_dir)
    manifest = make_passport_crops(normalized_path, crops_dir, quality_presets=settings.ocr_quality_presets)
    manifest_path = work_dir / "manifest.json"
    output_path = work_dir / "paddle_ocr_output.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    worker_path = Path(__file__).resolve().parent / "paddle_ocr_worker.py"
    env = os.environ.copy()
    env["OCR_DEVICE"] = settings.ocr_device
    cmd = [sys.executable, str(worker_path), "--manifest", str(manifest_path), "--output", str(output_path)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=settings.ocr_worker_timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"PaddleOCR worker timeout after {settings.ocr_worker_timeout_sec}s",
            "stdout": str(exc.stdout)[-4000:],
            "stderr": str(exc.stderr)[-4000:],
            "items": [],
            "passport_data": {},
            "work_dir": str(work_dir),
            "normalized_path": str(normalized_path),
            "passport_face_crop": str(crops_dir / "passport_face.jpg"),
        }

    if not output_path.exists():
        return {
            "ok": False,
            "error": "PaddleOCR worker did not create output JSON",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "items": [],
            "passport_data": {},
            "work_dir": str(work_dir),
            "normalized_path": str(normalized_path),
            "passport_face_crop": str(crops_dir / "passport_face.jpg"),
        }

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    raw_items = raw.get("items", [])

    classifier_path = settings.artifacts_dir / "text_line_classifier.joblib"
    classified_items = classify_ocr_items(raw_items, classifier_path)
    raw_with_classification = dict(raw)
    raw_with_classification["items"] = classified_items

    # Parser works with the same OCR items, enriched with pred_label/pred_confidence for debug.
    parsed = parse_passport_data(raw_with_classification)

    raw_csv_path = work_dir / "ocr_items.csv"
    classified_csv_path = work_dir / "ocr_classified.csv"
    try:
        import pandas as pd
        pd.DataFrame(raw_items).to_csv(raw_csv_path, index=False)
        pd.DataFrame(classified_items).to_csv(classified_csv_path, index=False)
    except Exception:
        raw_csv_path = None
        classified_csv_path = None

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "items": classified_items,
        "worker_errors": raw.get("errors", []),
        "worker_preflight": raw.get("preflight", {}),
        "text_classifier_used": classifier_path.exists(),
        "text_classifier_path": str(classifier_path),
        "passport_data": parsed["passport_data"],
        "confidence": parsed["confidence"],
        "debug": parsed["debug"],
        "work_dir": str(work_dir),
        "normalized_path": str(normalized_path),
        "passport_face_crop": str(crops_dir / "passport_face.jpg"),
        "ocr_items_csv": str(raw_csv_path) if raw_csv_path else None,
        "ocr_classified_csv": str(classified_csv_path) if classified_csv_path else None,
    }


def ocr_passport_image(passport_image_path: str) -> Dict[str, Any]:
    settings = get_settings()
    saved = _save_input_file(passport_image_path, "passport_ocr", settings)
    return run_paddle_ocr(saved, settings)


def register_profile(
    last_name: str,
    first_name: str,
    middle_name: str,
    birth_date: str,
    passport_series: str,
    passport_number: str,
    reference_image_path: str,
) -> Dict[str, Any]:
    settings = get_settings()
    store = get_store(settings)
    ref_path = _save_input_file(reference_image_path, "reference", settings)
    emb = safe_embedding(ref_path, settings.face_model_name, settings.face_detector_backend, enforce_detection=True)
    if not emb.get("ok"):
        return {"ok": False, "error": "cannot_extract_reference_face", "details": emb.get("error")}
    profile = {
        "full_name": normalize_name(f"{last_name} {first_name} {middle_name}"),
        "last_name": normalize_name(last_name),
        "first_name": normalize_name(first_name),
        "middle_name": normalize_name(middle_name),
        "birth_date": normalize_date(birth_date),
        "passport_series": digits_only(passport_series)[-4:],
        "passport_number": digits_only(passport_number)[-6:],
        "reference_face_embedding": emb["embedding"],
        "embedding_model": settings.face_model_name,
        "detector_backend": settings.face_detector_backend,
        "embedding_dim": len(emb["embedding"]),
    }
    if settings.store_debug_artifacts:
        profile["reference_image_path"] = str(ref_path)
    saved = store.create_profile(profile)
    result = {k: v for k, v in saved.items() if k != "reference_face_embedding"}
    return {"ok": True, "profile": result, "stored_in": "supabase" if store.using_supabase else "local_json", "storage_error": store.last_error}


def verify_identity(passport_image_path: str, selfie_image_path: str) -> Dict[str, Any]:
    settings = get_settings()
    store = get_store(settings)
    passport_path = _save_input_file(passport_image_path, "passport_verify", settings)
    selfie_path = _save_input_file(selfie_image_path, "selfie", settings)

    ocr = run_paddle_ocr(passport_path, settings)
    if not ocr.get("items"):
        return {"ok": False, "error": "ocr_failed", "details": ocr}
    passport_data = ocr.get("passport_data", {})

    profiles = store.list_profiles()
    best = choose_best_profile(profiles, passport_data)
    profile = best.get("profile")
    data_score = float(best.get("score") or 0.0)

    passport_face_path = Path(ocr.get("passport_face_crop") or passport_path)
    passport_face = safe_embedding(passport_face_path, settings.face_model_name, settings.face_detector_backend, enforce_detection=True)
    selfie_face = safe_embedding(selfie_path, settings.face_model_name, settings.face_detector_backend, enforce_detection=True)

    reference_embedding = profile.get("reference_face_embedding") if profile else None
    passport_embedding = passport_face.get("embedding")
    selfie_embedding = selfie_face.get("embedding")

    passport_reference_distance = cosine_distance(reference_embedding, passport_embedding)
    selfie_passport_distance = cosine_distance(selfie_embedding, passport_embedding)
    selfie_reference_distance = cosine_distance(selfie_embedding, reference_embedding)

    decision = final_decision(
        data_score=data_score,
        passport_reference_distance=passport_reference_distance,
        selfie_passport_distance=selfie_passport_distance,
        selfie_reference_distance=selfie_reference_distance,
        accept_threshold=settings.face_accept_threshold,
        review_threshold=settings.face_review_threshold,
        data_accept_threshold=settings.data_accept_threshold,
        data_review_threshold=settings.data_review_threshold,
    )

    attempt = {
        "identity_id": profile.get("id") if profile else None,
        "input_passport_data": passport_data,
        "passport_reference_distance": passport_reference_distance,
        "selfie_passport_distance": selfie_passport_distance,
        "selfie_reference_distance": selfie_reference_distance,
        "face_accept_threshold": settings.face_accept_threshold,
        "face_review_threshold": settings.face_review_threshold,
        "data_match_score": data_score,
        "data_verified": decision["data_verified"],
        "passport_photo_verified": decision["passport_photo_verified"],
        "selfie_verified": decision["selfie_verified"],
        "final_decision": decision["final_decision"],
        "error_message": "; ".join([str(passport_face.get("error") or ""), str(selfie_face.get("error") or "")]).strip("; "),
    }
    if settings.store_debug_artifacts:
        # Debug artifacts can contain personal data, OCR text, local file paths and
        # biometric embeddings. They are therefore opt-in.
        attempt.update({
            "parser_debug": ocr.get("debug"),
            "ocr_items": ocr.get("items", [])[:350],
            "passport_photo_embedding": passport_embedding,
            "selfie_embedding": selfie_embedding,
            "passport_photo_path": str(passport_path),
            "selfie_path": str(selfie_path),
        })
    try:
        store.save_attempt(attempt)
    except Exception:
        pass

    safe_profile = None
    if profile:
        safe_profile = {k: v for k, v in profile.items() if k != "reference_face_embedding"}
    return {
        "ok": True,
        "final_decision": decision["final_decision"],
        "matched_profile_id": str(profile.get("id")) if profile else None,
        "stored_in": "supabase" if store.using_supabase else "local_json",
        "data_match_score": data_score,
        "passport_data": passport_data,
        "face": {
            "passport_reference_distance": passport_reference_distance,
            "selfie_passport_distance": selfie_passport_distance,
            "selfie_reference_distance": selfie_reference_distance,
            "statuses": decision["face_statuses"],
            "passport_face_error": passport_face.get("error"),
            "selfie_error": selfie_face.get("error"),
        },
        "debug": {
            "matched_profile": safe_profile,
            "data_match_details": best.get("details"),
            "ocr_confidence": ocr.get("confidence"),
            "worker_errors": ocr.get("worker_errors"),
            "storage_error": store.last_error,
            "work_dir": ocr.get("work_dir"),
            "normalized_path": ocr.get("normalized_path"),
            "passport_face_crop": ocr.get("passport_face_crop"),
            "ocr_items_csv": ocr.get("ocr_items_csv"),
            "ocr_classified_csv": ocr.get("ocr_classified_csv"),
            "text_classifier_used": ocr.get("text_classifier_used"),
        },
    }


def make_runtime_zip() -> str:
    """Create a safe export archive for sharing the project code.

    Runtime folders may contain passports, selfies, local_store.json, face embeddings,
    OCR debug CSVs and .env files. Those artifacts are intentionally excluded from
    the downloadable ZIP so the Gradio export button is safe to use before GitHub
    publication.
    """
    settings = get_settings()
    out_base = settings.base_dir / "mad_identity_runtime_export"
    out_zip = out_base.with_suffix(".zip")

    exclude_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        "storage",
        "uploads",
        "ocr_jobs",
        "debug",
    }
    exclude_files = {".env", "local_store.json"}
    exclude_suffixes = {".pyc", ".pyo", ".log", ".key", ".pem"}

    def should_include(path: Path) -> bool:
        rel = path.relative_to(settings.base_dir)
        if any(part in exclude_dirs for part in rel.parts):
            return False
        if path.name in exclude_files or path.name.startswith(".env."):
            return False
        if path.name.startswith("mad_identity_runtime_export"):
            return False
        if path.suffix.lower() in exclude_suffixes:
            return False
        return True

    if out_zip.exists():
        out_zip.unlink()

    import zipfile
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in settings.base_dir.rglob("*"):
            if path.is_file() and should_include(path):
                zf.write(path, path.relative_to(settings.base_dir))
    return str(out_zip)
