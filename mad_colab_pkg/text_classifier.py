from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np


def _extract_text(item: Dict[str, Any]) -> str:
    for key in ("text", "rec_text", "value"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def classify_ocr_items(items: Iterable[Dict[str, Any]], artifact_path: Path) -> List[Dict[str, Any]]:
    """Add text-line classifier predictions to OCR items.

    The classifier is not the only source of truth. It is used as an ML diagnostic layer,
    while the final parser still relies on ROI, regex, MRZ and consensus rules.
    """
    items = [dict(x) for x in items]
    if not artifact_path.exists():
        for item in items:
            item["pred_label"] = None
            item["pred_confidence"] = None
        return items

    model = joblib.load(artifact_path)
    texts = [_extract_text(item) for item in items]
    if not texts:
        return items

    labels = model.predict(texts)
    confidences = [None] * len(texts)
    try:
        proba = model.predict_proba(texts)
        confidences = np.max(proba, axis=1).astype(float).tolist()
    except Exception:
        try:
            scores = model.decision_function(texts)
            if scores.ndim == 1:
                # Binary fallback, normalize approximately.
                confidences = (1.0 / (1.0 + np.exp(-np.abs(scores)))).astype(float).tolist()
            else:
                exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                softmax = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
                confidences = np.max(softmax, axis=1).astype(float).tolist()
        except Exception:
            pass

    for item, label, conf in zip(items, labels, confidences):
        item["pred_label"] = str(label)
        item["pred_confidence"] = None if conf is None else round(float(conf), 4)
    return items
