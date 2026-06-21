#!/usr/bin/env python3
import argparse
import json
import random
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NOISE_LINES = [
    "РОССИЙСКАЯ ФЕДЕРАЦИЯ", "ПАСПОРТ", "ЛИЧНЫЙ КОД", "ПОДПИСЬ ВЛАДЕЛЬЦА", "КОД ПОДРАЗДЕЛЕНИЯ",
    "МЕСТО ЖИТЕЛЬСТВА", "PNRUS<<<<<<<<<<<", "<<<<<<<<<<<<<<<<", "ОТДЕЛЕНИЕ", "ДАТА ВЫДАЧИ",
]

OCR_REPL = {
    "В": ["B", "8"], "Н": ["H"], "К": ["K"], "С": ["C"], "О": ["O", "0"], "Р": ["P"],
    "А": ["A"], "Е": ["E"], "Т": ["T"], "М": ["M"], "Л": ["J", "L"], "И": ["N", "I"],
    "З": ["3"], "Ч": ["4"], "Д": ["D"], "Б": ["6"],
}


def noise_text(s: str, p: float = 0.08) -> str:
    out = []
    for ch in str(s).upper():
        if ch in OCR_REPL and random.random() < p:
            out.append(random.choice(OCR_REPL[ch]))
        elif ch.isdigit() and random.random() < p / 2:
            out.append(random.choice("0123456789"))
        else:
            out.append(ch)
    text = "".join(out)
    if random.random() < 0.15:
        text = text.replace(".", random.choice([".", " ", "-", "/"]))
    if random.random() < 0.10:
        text = " ".join(text.split())
    return text


def add(rows: List[Dict], person_id: int, label: str, text: str, copies: int = 1):
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return
    for _ in range(copies):
        rows.append({"person_id": person_id, "label": label, "text": noise_text(text, p=random.uniform(0.02, 0.14))})


def build_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows: List[Dict] = []
    for idx, r in df.iterrows():
        person_id = int(idx)
        series = re.sub(r"\D+", "", str(r.get("series", ""))).zfill(4)[-4:]
        number = re.sub(r"\D+", "", str(r.get("number", ""))).zfill(6)[-6:]
        add(rows, person_id, "surname", r.get("lastName"), 2)
        add(rows, person_id, "firstname", r.get("firstName"), 2)
        add(rows, person_id, "patronymic", r.get("middleName"), 2)
        add(rows, person_id, "birth_date", r.get("birthDate"), 3)
        add(rows, person_id, "issue_date", r.get("issuedDate"), 2)
        add(rows, person_id, "birth_place", r.get("birthPlace"), 2)
        add(rows, person_id, "department_code", r.get("subdivisionCode"), 3)
        add(rows, person_id, "issued_by", r.get("issuingAuthority"), 2)
        add(rows, person_id, "passport_series", series, 3)
        add(rows, person_id, "passport_number", number, 3)
        add(rows, person_id, "passport_id", f"{series} {number}", 2)
        add(rows, person_id, "passport_id", f"{series}{number}", 2)
        add(rows, person_id, "gender", random.choice(["МУЖ.", "ЖЕН.", "МУЖ", "ЖЕН"]), 1)
        for _ in range(4):
            add(rows, person_id, "noise", random.choice(NOISE_LINES), 1)
        # Hard negatives: date-like and id-like wrong values.
        add(rows, person_id, "noise", f"{random.randint(1,31):02d}.{random.randint(1,12):02d}.{random.randint(1900,2035)}", 1)
        add(rows, person_id, "noise", f"{random.randint(100,999)}-{random.randint(100,999)}", 1)
        add(rows, person_id, "noise", "".join(random.choice("0123456789") for _ in range(6)), 1)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global RANDOM_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/passport-2000.csv")
    parser.add_argument("--out", default="artifacts/text_line_classifier.joblib")
    parser.add_argument("--metrics", default="artifacts/text_line_classifier_metrics.json")
    parser.add_argument("--dataset-out", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    RANDOM_SEED = int(args.seed)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    csv_path = Path(args.csv)
    dataset = build_dataset(csv_path)
    person_ids = dataset["person_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(person_ids)
    split = int(len(person_ids) * 0.8)
    train_ids = set(person_ids[:split])
    train = dataset[dataset["person_id"].isin(train_ids)]
    test = dataset[~dataset["person_id"].isin(train_ids)]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=60000)),
        ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=80, tol=1e-3, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    model.fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    report = classification_report(test["label"], pred, output_dict=True, zero_division=0)
    labels = sorted(dataset["label"].unique())
    cm = confusion_matrix(test["label"], pred, labels=labels).tolist()
    dataset_out = Path(args.dataset_out) if args.dataset_out else Path(args.metrics).with_name("text_line_classifier_dataset.csv")
    metrics = {
        "ok": True,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(csv_path),
        "input_csv_sha256": sha256_file(csv_path),
        "seed": int(RANDOM_SEED),
        "dataset_rows": int(len(dataset)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_person_count": int(len(train_ids)),
        "test_person_count": int(len(set(person_ids) - set(train_ids))),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm,
        "artifact": str(args.out),
        "dataset_out": str(dataset_out),
        "note": "This file is generated by training/train_text_classifier.py inside the Colab notebook, not manually copied.",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    Path(args.metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset.to_csv(dataset_out, index=False)
    print(json.dumps({
        "ok": True,
        "artifact": args.out,
        "metrics": args.metrics,
        "dataset_out": str(dataset_out),
        "trained_at_utc": metrics["trained_at_utc"],
        "input_csv_sha256": metrics["input_csv_sha256"],
        "dataset_rows": metrics["dataset_rows"],
        "macro_f1": report["macro avg"]["f1-score"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
