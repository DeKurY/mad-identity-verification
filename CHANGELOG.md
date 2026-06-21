# Changelog

## v4.1 — PII and development security cleanup

- Re-scanned the repository for owner-provided real personal data and removed/avoided target matches.
- Added upload type/size validation and safer runtime file permissions.
- Changed Gradio debug and DeepFace auto-repair installs to opt-in.
- Made Supabase schema secure-by-default with RLS enabled; moved insecure RLS-disabled mode to a separate demo-only SQL file.
- Made storage of OCR debug artifacts, image paths and extra attempt embeddings opt-in via `STORE_DEBUG_ARTIFACTS=1`.
- Added dedicated PII cleanup and development security audit reports.

## v4 — GitHub cleanup

- Added safe Colab Secrets flow for `SUPABASE_URL` and `SUPABASE_KEY`.
- Added `.env.example`, `.gitignore`, security notes and GitHub-ready README.
- Kept runtime package structure intact to avoid breaking imports.

## v4 — Runtime baseline

- PaddleOCR returned as the main OCR engine.
- PaddleOCR runs in a separate subprocess worker.
- OCR resize defaults changed to `text_det_limit_type=max`, `text_det_limit_side_len=1600`.
- Text-line classifier artifact and metrics included.
