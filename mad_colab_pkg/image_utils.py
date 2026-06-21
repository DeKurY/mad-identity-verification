from pathlib import Path
from typing import Dict, List

import os
import cv2
import numpy as np
from PIL import Image, ImageOps


def ensure_rgb_image(input_path: Path, output_path: Path) -> Path:
    img = Image.open(input_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95, subsampling=0)
    return output_path


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def auto_normalize_passport(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "input_rgb.jpg"
    ensure_rgb_image(input_path, rgb_path)

    image = cv2.imread(str(rgb_path))
    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")

    max_side = int(os.getenv("PASSPORT_NORMALIZE_MAX_SIDE", "1700"))
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray_blur, 45, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    warped = None
    image_area = image.shape[0] * image.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 0.16 * image_area:
            pts = approx.reshape(4, 2).astype("float32")
            warped = _four_point_transform(image, pts)
            break

    if warped is None:
        warped = image

    # Ensure portrait orientation for Russian passport spread.
    hh, ww = warped.shape[:2]
    if ww > hh * 1.25:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    norm = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    out = output_dir / "passport_normalized.jpg"
    cv2.imwrite(str(out), norm, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _crop_frac(img: Image.Image, box) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, int(w * x1)))
    y1 = max(0, min(h, int(h * y1)))
    x2 = max(0, min(w, int(w * x2)))
    y2 = max(0, min(h, int(h * y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return img.crop((x1, y1, x2, y2))


def _pil_to_cv_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _cv_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr.astype("uint8"), "RGB")


def _gray_clahe_sharp(img: Image.Image) -> Image.Image:
    rgb = _pil_to_cv_rgb(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.45, blur, -0.45, 0)
    return _cv_to_pil_rgb(sharp)


def _light_norm(img: Image.Image) -> Image.Image:
    rgb = _pil_to_cv_rgb(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)
    return _cv_to_pil_rgb(norm)


def _save_image(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95, subsampling=0)


def _save_variant(manifest: List[Dict[str, str]], tag: str, img: Image.Image, output_dir: Path, suffix: str = "") -> None:
    safe_tag = f"{tag}{suffix}"
    path = output_dir / f"{safe_tag}.jpg"
    _save_image(path, img)
    manifest.append({"tag": safe_tag, "path": str(path)})


def _upscale_if_small(img: Image.Image, min_long_side: int = 1500) -> Image.Image:
    w, h = img.size
    if max(w, h) >= min_long_side:
        return img
    factor = min_long_side / max(w, h)
    return img.resize((int(w * factor), int(h * factor)), Image.Resampling.LANCZOS)


def make_passport_crops(normalized_path: Path, output_dir: Path, quality_presets: bool = True) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(normalized_path).convert("RGB")
    img = _upscale_if_small(img, min_long_side=int(os.getenv("PASSPORT_CROP_MIN_LONG_SIDE", "1500")))

    fast_mode = _bool_env("OCR_FAST_MODE", True)
    include_full = _bool_env("OCR_INCLUDE_FULL", False)
    include_right_unrotated = _bool_env("OCR_INCLUDE_RIGHT_UNROTATED", False)

    # Template coordinates for a normalized Russian passport spread.
    # Boxes are deliberately wider than the printed fields; the parser later filters by regex and source tag.
    boxes = {
        "top":          (0.00, 0.00, 1.00, 0.46),
        "bottom":       (0.00, 0.43, 1.00, 0.96),

        # Right vertical passport id. The rotated versions are sent to OCR.
        "right_full":   (0.74, 0.00, 1.00, 0.96),
        "right_bottom": (0.74, 0.42, 1.00, 0.92),

        # Face crop is for DeepFace/debug only, not OCR.
        "passport_face": (0.04, 0.50, 0.36, 0.88),

        # Top page fields.
        "issued_by":    (0.08, 0.055, 0.88, 0.235),
        "issue_date":   (0.16, 0.215, 0.48, 0.325),
        "department":   (0.62, 0.215, 0.93, 0.325),

        # Bottom page fields. These are shifted/widened compared with the previous v3 boxes.
        "surname":      (0.46, 0.500, 0.95, 0.590),
        "firstname":    (0.46, 0.570, 0.95, 0.660),
        "patronymic":   (0.46, 0.625, 0.95, 0.715),
        "gender":       (0.34, 0.635, 0.67, 0.750),
        "birth_date":   (0.55, 0.635, 0.96, 0.750),
        "birth_place":  (0.32, 0.705, 0.96, 0.875),
        "mrz":          (0.00, 0.815, 1.00, 0.970),
    }

    manifest: List[Dict[str, str]] = []
    _save_image(output_dir / "full.jpg", img)
    _save_image(output_dir / "passport_face.jpg", _crop_frac(img, boxes["passport_face"]))

    if include_full:
        _save_variant(manifest, "full", img, output_dir)

    horizontal_tags = [
        "top", "bottom", "issued_by", "issue_date", "department",
        "surname", "firstname", "patronymic", "gender", "birth_date", "birth_place", "mrz",
    ]

    for tag in horizontal_tags:
        crop = _crop_frac(img, boxes[tag])
        _save_variant(manifest, tag, crop, output_dir)
        if quality_presets and not fast_mode:
            _save_variant(manifest, tag, _gray_clahe_sharp(crop), output_dir, suffix="__gray")
            if tag in {"issue_date", "department", "birth_date", "mrz", "right_full", "right_bottom"}:
                _save_variant(manifest, tag, _light_norm(crop), output_dir, suffix="__light")

    for tag in ["right_full", "right_bottom"]:
        crop = _crop_frac(img, boxes[tag])
        _save_image(output_dir / f"{tag}.jpg", crop)
        if include_right_unrotated:
            _save_variant(manifest, tag, crop, output_dir)
        for angle in (90, 270):
            rotated = crop.rotate(angle, expand=True)
            _save_variant(manifest, f"{tag}_rot{angle}", rotated, output_dir)
            if quality_presets and not fast_mode:
                _save_variant(manifest, f"{tag}_rot{angle}", _gray_clahe_sharp(rotated), output_dir, suffix="__gray")

    return manifest
