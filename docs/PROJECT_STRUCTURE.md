# Project structure

## Runtime package

- `mad_colab_pkg/colab_app.py` — Gradio UI.
- `mad_colab_pkg/colab_pipeline.py` — orchestration layer: settings, OCR worker, parser, embeddings, store, decision.
- `mad_colab_pkg/image_utils.py` — document normalization, perspective transform, CLAHE, ROI crops.
- `mad_colab_pkg/paddle_ocr_worker.py` — isolated PaddleOCR CLI worker.
- `mad_colab_pkg/passport_parser.py` — structured passport data extraction from OCR items.
- `mad_colab_pkg/text_classifier.py` — loads joblib classifier and adds predicted labels to OCR items.
- `mad_colab_pkg/face_service.py` — DeepFace availability, embeddings, cosine distance.
- `mad_colab_pkg/decision_service.py` — data matching score and final decision.
- `mad_colab_pkg/identity_store.py` — Supabase-first storage with local JSON fallback.

## Training

- `training/train_text_classifier.py` — trains OCR text-line classifier.
- `training/calibrate_face_threshold_celeba.py` — optional threshold calibration experiment on CelebA identity pairs.

## Data and artifacts

- `data/passport-2000.csv` — source table for synthetic/noisy OCR-line dataset generation.
- `artifacts/text_line_classifier.joblib` — trained classifier.
- `artifacts/text_line_classifier_metrics.json` — metrics report.
- `artifacts/text_line_classifier_dataset.csv` — generated training dataset snapshot.

## Database

- `sql/supabase_schema.sql` — secure-by-default schema for `identity_profiles` and `verification_attempts` with RLS enabled.
- `sql/supabase_schema_demo_insecure.sql` — demo-only schema variant with RLS disabled for synthetic classroom runs.

## Notebook

- `notebooks/MAD_Identity_Verification_Final_Defense_Clean.ipynb` — sanitized Colab notebook that reads Supabase credentials from Secrets.
