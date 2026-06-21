#!/usr/bin/env python3
"""PaddleOCR worker for Colab.

Why this is a separate process:
PaddleOCR 3.x initializes PaddleX/PDX internally. Reinitializing it many times in a
notebook kernel can be unstable, so the Gradio app calls this file through
subprocess.run(...). If PaddleOCR fails, the main Colab process survives and the
worker writes a diagnostic JSON.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Reduce PaddleX/model-host checks. This must be set before importing PaddleOCR.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", os.getenv("PADDLE_GPU_MEMORY_FRACTION", "0.80"))


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _box_to_points(box: Any):
    if box is None:
        return None
    try:
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = box
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            return [[float(p[0]), float(p[1])] for p in box[:4]]
    except Exception:
        return None
    return None


def _box_stats(points):
    if not points:
        return None, None, None, None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys), max(xs) - min(xs), max(ys) - min(ys))


def _extract_result_json(res: Any) -> Dict[str, Any]:
    data = getattr(res, "json", None)
    if callable(data):
        data = data()
    if data is None:
        data = {}
    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        data = data["res"]
    return data if isinstance(data, dict) else {}


def _items_from_result(data: Dict[str, Any], tag: str, image_path: str) -> List[Dict[str, Any]]:
    texts = data.get("rec_texts") or data.get("texts") or data.get("text") or []
    scores = data.get("rec_scores") or data.get("scores") or []
    boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or data.get("boxes") or []

    items: List[Dict[str, Any]] = []
    for i, text in enumerate(texts):
        if str(text).strip() == "":
            continue
        score = _safe_float(scores[i] if i < len(scores) else 0.0)
        box = boxes[i] if i < len(boxes) else None
        points = _box_to_points(box)
        cx, cy, bw, bh = _box_stats(points)
        items.append({
            "tag": tag,
            "image_path": image_path,
            "text": str(text),
            "score": score,
            "box": points,
            "cx": cx,
            "cy": cy,
            "bw": bw,
            "bh": bh,
        })
    return items


def _preflight() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "python": sys.version,
        "ocr_device_env": os.getenv("OCR_DEVICE", "cpu"),
        "paddle_ok": False,
        "torch_ok": False,
    }

    try:
        import torch  # noqa: F401
        status["torch_ok"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        status["torch_cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception as exc:
        status["torch_error"] = repr(exc)

    try:
        import paddle
        status["paddle_ok"] = True
        status["paddle_version"] = getattr(paddle, "__version__", None)
        status["paddle_compiled_with_cuda"] = bool(paddle.is_compiled_with_cuda())
    except Exception as exc:
        status["paddle_error"] = repr(exc)

    return status


def build_ocr():
    from paddleocr import PaddleOCR

    use_doc_unwarping = os.getenv("PADDLE_USE_DOC_UNWARPING", "false").strip().lower() in {"1", "true", "yes", "y"}
    use_textline_orientation = os.getenv("PADDLE_TEXTLINE_ORIENTATION", "false").strip().lower() in {"1", "true", "yes", "y"}

    kwargs = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        text_detection_model_name=os.getenv("PADDLE_DET_MODEL", "PP-OCRv5_mobile_det"),
        text_recognition_model_name=os.getenv("PADDLE_REC_MODEL", "cyrillic_PP-OCRv5_mobile_rec"),
        # IMPORTANT: max prevents narrow vertical crops from becoming 960x4210.
        text_det_limit_type=os.getenv("PADDLE_DET_LIMIT_TYPE", "max"),
        text_det_limit_side_len=int(os.getenv("PADDLE_DET_LIMIT_SIDE_LEN", "1600")),
        text_det_thresh=float(os.getenv("PADDLE_DET_THRESH", "0.25")),
        text_det_box_thresh=float(os.getenv("PADDLE_DET_BOX_THRESH", "0.55")),
        text_det_unclip_ratio=float(os.getenv("PADDLE_DET_UNCLIP_RATIO", "1.35")),
    )

    device = os.getenv("OCR_DEVICE", "cpu")
    if device and device.lower() not in {"", "auto", "none"}:
        kwargs["device"] = device

    print(json.dumps({"stage": "build_ocr", "kwargs": kwargs}, ensure_ascii=False), flush=True)

    # PaddleOCR 3.x changes accepted kwargs across minor versions.
    # If a kwarg is rejected as "Unknown argument: X", I drop it and retry.
    while True:
        try:
            return PaddleOCR(**kwargs)
        except ValueError as exc:
            msg = str(exc)
            prefix = "Unknown argument: "
            if prefix in msg:
                bad_arg = msg.split(prefix, 1)[1].strip().strip("'").strip('"')
                if bad_arg in kwargs:
                    kwargs.pop(bad_arg, None)
                    print(
                        json.dumps(
                            {
                                "stage": "build_ocr_retry_without_unknown_arg",
                                "removed": bad_arg,
                                "kwargs": kwargs,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
            raise


def _write_output(output_path: Path, payload: Dict[str, Any], exit_code: int = 0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": exit_code == 0, "output": str(output_path), "items": len(payload.get("items", [])), "errors": len(payload.get("errors", []))}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON list with {'tag','path'} items")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    preflight = _preflight()
    print(json.dumps({"stage": "preflight", **preflight}, ensure_ascii=False), flush=True)

    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except Exception as exc:
        _write_output(output_path, {"items": [], "errors": [{"stage": "read_manifest", "error": repr(exc), "traceback": traceback.format_exc()}], "preflight": preflight}, exit_code=1)
        sys.exit(1)

    try:
        ocr = build_ocr()
    except Exception as exc:
        _write_output(output_path, {"items": [], "errors": [{"stage": "build_ocr", "error": repr(exc), "traceback": traceback.format_exc()}], "preflight": preflight}, exit_code=1)
        sys.exit(1)

    all_items: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    max_images = int(os.getenv("OCR_MAX_IMAGES", "0") or "0")
    if max_images > 0:
        manifest = manifest[:max_images]

    for entry in manifest:
        tag = str(entry.get("tag") or Path(entry.get("path", "image")).stem)
        image_path = str(entry.get("path"))
        try:
            result = ocr.predict(image_path)
            for res in result:
                data = _extract_result_json(res)
                all_items.extend(_items_from_result(data, tag, image_path))
        except Exception as exc:
            errors.append({"tag": tag, "image_path": image_path, "error": repr(exc), "traceback": traceback.format_exc()})

    payload = {"items": all_items, "errors": errors, "preflight": preflight}
    _write_output(output_path, payload, exit_code=0 if not errors else 2)
    sys.exit(0 if not errors else 2)


if __name__ == "__main__":
    main()
