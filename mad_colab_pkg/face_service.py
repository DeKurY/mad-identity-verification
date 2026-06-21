import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# DeepFace on modern Colab/TensorFlow/Keras stacks is usually more stable with legacy tf-keras.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("DEEPFACE_HOME", str(Path.home() / ".deepface"))


_DEEPFACE_IMPORT_ERROR: Optional[str] = None


def cosine_distance(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return None
    return float(1.0 - np.dot(va, vb) / denom)


def _run_pip_install(args: List[str]) -> str:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + args
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"pip install failed: {' '.join(args)}\n{proc.stdout[-4000:]}")
    return proc.stdout[-4000:]


def deepface_status() -> Dict[str, Any]:
    """Return an import diagnostic that is shown in the notebook sanity-check cell."""
    try:
        from deepface import DeepFace  # noqa: F401
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def ensure_deepface_available(auto_install: bool = False) -> Dict[str, Any]:
    """Import DeepFace, optionally trying a small Colab-safe repair install first.

    The previous error message said only "DeepFace is not installed", which was misleading:
    an import can also fail because tf-keras / retina-face / gdown / lightphe is missing.
    """
    global _DEEPFACE_IMPORT_ERROR
    status = deepface_status()
    if status["ok"]:
        return status

    _DEEPFACE_IMPORT_ERROR = status["error"]
    if not auto_install:
        return status

    try:
        _run_pip_install([
            "deepface==0.0.100",
            "retina-face==0.0.17",
            "tf-keras",
            "gdown",
            "fire",
            "flask",
            "flask-cors",
            "mtcnn",
            "gunicorn",
            "lightphe",
            "lightdsa",
        ])
        # In case a broken/partial import was cached before installation.
        for name in list(sys.modules):
            if name == "deepface" or name.startswith("deepface.") or name == "retinaface" or name.startswith("retinaface."):
                sys.modules.pop(name, None)
        status = deepface_status()
        _DEEPFACE_IMPORT_ERROR = status.get("error")
        return status
    except Exception as exc:
        _DEEPFACE_IMPORT_ERROR = f"initial_import={_DEEPFACE_IMPORT_ERROR}; repair_install={repr(exc)}"
        return {"ok": False, "error": _DEEPFACE_IMPORT_ERROR}


def get_face_embedding(
    image_path: Path,
    model_name: str = "Facenet512",
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
) -> List[float]:
    auto_install = os.getenv("MAD_AUTO_INSTALL_DEEPFACE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    status = ensure_deepface_available(auto_install=auto_install)
    if not status.get("ok"):
        raise RuntimeError(
            "DeepFace import failed. Run the notebook install cell again and then Runtime -> Restart runtime if needed. "
            f"Original import error: {status.get('error')}"
        )

    from deepface import DeepFace

    result = DeepFace.represent(
        img_path=str(image_path),
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=True,
    )
    if isinstance(result, list):
        if not result:
            raise RuntimeError("DeepFace did not return any face embedding")
        embedding = result[0].get("embedding") if isinstance(result[0], dict) else result[0]
    elif isinstance(result, dict):
        embedding = result.get("embedding")
    else:
        embedding = None
    if embedding is None:
        raise RuntimeError("DeepFace result has no embedding")
    return [float(x) for x in embedding]


def safe_embedding(
    image_path: Path,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool = True,
) -> Dict[str, Any]:
    attempts = []
    backends = [detector_backend]
    # If RetinaFace is unavailable or too strict, OpenCV is a useful Colab fallback.
    if detector_backend != "opencv":
        backends.append("opencv")

    for backend in backends:
        try:
            emb = get_face_embedding(image_path, model_name, backend, enforce_detection=enforce_detection)
            return {"ok": True, "embedding": emb, "error": None, "detector_backend_used": backend, "enforce_detection": enforce_detection}
        except Exception as exc:
            attempts.append({"backend": backend, "enforce_detection": enforce_detection, "error": repr(exc)})

    if enforce_detection:
        for backend in backends:
            try:
                emb = get_face_embedding(image_path, model_name, backend, enforce_detection=False)
                return {
                    "ok": True,
                    "embedding": emb,
                    "error": "fallback_without_enforce_detection",
                    "detector_backend_used": backend,
                    "enforce_detection": False,
                    "attempts": attempts,
                }
            except Exception as exc:
                attempts.append({"backend": backend, "enforce_detection": False, "error": repr(exc)})

    return {"ok": False, "embedding": None, "error": attempts[-1]["error"] if attempts else "unknown", "attempts": attempts}
