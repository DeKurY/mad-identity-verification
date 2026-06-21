#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def cosine_distance(a, b) -> float:
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.dot(va, vb) / denom)


def find_identity_file(root: Path) -> Path:
    candidates = [
        root / "identity_CelebA.txt",
        root / "list_identity_celeba.txt",
        root / "Anno" / "identity_CelebA.txt",
        root / "Anno" / "list_identity_celeba.txt",
        root / "Eval" / "identity_CelebA.txt",
        root / "Eval" / "list_identity_celeba.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(root.rglob("*identity*CelebA*.txt")) + list(root.rglob("*identity*celeba*.txt"))
    if hits:
        return hits[0]
    raise FileNotFoundError("Identity labels not found. Need identity_CelebA.txt or list_identity_celeba.txt.")


def find_images_dir(root: Path) -> Path:
    candidates = [
        root / "img_align_celeba",
        root / "Img" / "img_align_celeba",
        root / "img_celeba",
        root / "Img" / "img_celeba",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    # Fast-ish heuristic: find common first image name.
    for p in root.rglob("000001.jpg"):
        return p.parent
    raise FileNotFoundError("CelebA images folder not found. Expected img_align_celeba or Img/img_align_celeba.")


def load_groups(identity_path: Path, images_dir: Path, max_identities: int, images_per_identity: int, seed: int) -> Dict[str, List[Path]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Path]] = defaultdict(list)
    with identity_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_id, person_id = parts[0], parts[1]
            p = images_dir / image_id
            if p.exists():
                groups[person_id].append(p)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    ids = list(groups.keys())
    rng.shuffle(ids)
    ids = ids[:max_identities]
    return {k: rng.sample(groups[k], min(images_per_identity, len(groups[k]))) for k in ids}


def build_pairs(groups: Dict[str, List[Path]], max_positive: int, max_negative: int, seed: int) -> List[Tuple[Path, Path, int]]:
    rng = random.Random(seed)
    positives: List[Tuple[Path, Path, int]] = []
    for _, imgs in groups.items():
        if len(imgs) >= 2:
            positives.append((imgs[0], imgs[1], 1))
    rng.shuffle(positives)
    positives = positives[:max_positive]

    ids = list(groups.keys())
    negatives: List[Tuple[Path, Path, int]] = []
    tries = 0
    while len(negatives) < max_negative and tries < max_negative * 20 and len(ids) >= 2:
        tries += 1
        a, b = rng.sample(ids, 2)
        negatives.append((rng.choice(groups[a]), rng.choice(groups[b]), 0))
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def represent_with_cache(paths: List[Path], cache_path: Path, model_name: str, detector_backend: str) -> Dict[str, List[float]]:
    cache: Dict[str, List[float]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [p for p in paths if str(p) not in cache]
    if missing:
        from deepface import DeepFace
        for i, p in enumerate(missing, 1):
            try:
                result = DeepFace.represent(
                    img_path=str(p),
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    align=True,
                )
            except Exception:
                # aligned CelebA works well with skip, but fallback can save a run if skip is unsupported.
                result = DeepFace.represent(
                    img_path=str(p),
                    model_name=model_name,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                )
            if isinstance(result, list):
                emb = result[0].get("embedding") if isinstance(result[0], dict) else result[0]
            else:
                emb = result.get("embedding")
            cache[str(p)] = [float(x) for x in emb]
            if i % 25 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                print(f"embedded {i}/{len(missing)} new images")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def evaluate(distances: List[float], labels: List[int]) -> Dict:
    distances = np.asarray(distances, dtype="float32")
    labels = np.asarray(labels, dtype="int32")
    grid = np.linspace(float(np.nanmin(distances)), float(np.nanmax(distances)), 350)
    best_f1 = {"threshold": None, "f1": -1, "accuracy": -1, "far": None, "frr": None}
    best_acc = {"threshold": None, "f1": -1, "accuracy": -1, "far": None, "frr": None}
    far_1pct = None
    far_5pct = None
    rows = []
    for thr in grid:
        pred = (distances <= thr).astype("int32")
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        acc = (tp + tn) / max(len(labels), 1)
        far = fp / max(fp + tn, 1)
        frr = fn / max(fn + tp, 1)
        row = {"threshold": float(thr), "f1": float(f1), "accuracy": float(acc), "far": float(far), "frr": float(frr)}
        rows.append(row)
        if f1 > best_f1["f1"]:
            best_f1 = row
        if acc > best_acc["accuracy"]:
            best_acc = row
        if far <= 0.01 and (far_1pct is None or frr < far_1pct["frr"]):
            far_1pct = row
        if far <= 0.05 and (far_5pct is None or frr < far_5pct["frr"]):
            far_5pct = row
    return {
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "far_le_1pct": far_1pct,
        "far_le_5pct": far_5pct,
        "positive_distance_mean": float(distances[labels == 1].mean()) if (labels == 1).any() else None,
        "negative_distance_mean": float(distances[labels == 0].mean()) if (labels == 0).any() else None,
        "positive_distance_p95": float(np.quantile(distances[labels == 1], 0.95)) if (labels == 1).any() else None,
        "negative_distance_p05": float(np.quantile(distances[labels == 0], 0.05)) if (labels == 0).any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeba-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="Facenet512")
    parser.add_argument("--detector-backend", default="skip")
    parser.add_argument("--max-identities", type=int, default=80)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument("--positive-pairs", type=int, default=300)
    parser.add_argument("--negative-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.celeba_root)
    images_dir = find_images_dir(root)
    identity_path = find_identity_file(root)
    groups = load_groups(identity_path, images_dir, args.max_identities, args.images_per_identity, args.seed)
    if not groups:
        raise RuntimeError("No identities with at least two existing images found.")
    pairs = build_pairs(groups, args.positive_pairs, args.negative_pairs, args.seed)
    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)}, key=lambda p: str(p))

    out = Path(args.out)
    cache_path = out.with_name(f"celeba_embeddings_{args.model_name}_{args.detector_backend}.json")
    embeddings = represent_with_cache(unique_paths, cache_path, args.model_name, args.detector_backend)

    distances = []
    labels = []
    for a, b, label in pairs:
        da = embeddings.get(str(a))
        db = embeddings.get(str(b))
        if da is None or db is None:
            continue
        d = cosine_distance(da, db)
        if math.isfinite(d):
            distances.append(d)
            labels.append(label)
    metrics = evaluate(distances, labels)
    result = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "celeba_root": str(root),
        "images_dir": str(images_dir),
        "identity_path": str(identity_path),
        "model_name": args.model_name,
        "detector_backend": args.detector_backend,
        "identity_count_used": len(groups),
        "pair_count": len(labels),
        "positive_pairs": int(sum(labels)),
        "negative_pairs": int(len(labels) - sum(labels)),
        "metrics": metrics,
        "important_note": "Do not automatically set production ACCEPT threshold to best_f1 if it is too soft. Keep a conservative accept/review split.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
