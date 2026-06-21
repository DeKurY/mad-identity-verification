# Changelog

## v0.4.2 — Public README cleanup

- Reworked the README as a public-facing project document.
- Renamed the Colab notebook to `notebooks/mad_identity_verification_colab.ipynb`.
- Removed owner-only reports and release notes from the public repository package.
- Kept public documentation focused on architecture, setup, limitations and future improvements.

## v0.4.1 — Secure public release cleanup

- Added upload type/size validation and safer runtime file permissions.
- Changed Gradio public sharing and debug defaults to opt-in.
- Made Supabase schema secure-by-default with RLS enabled.
- Moved the RLS-disabled SQL schema to a separate demo-only file.
- Made storage of OCR debug artifacts, image paths and extra attempt embeddings opt-in via `STORE_DEBUG_ARTIFACTS=1`.

## v0.4.0 — GitHub-ready baseline

- Added safe Colab Secrets flow for `SUPABASE_URL` and `SUPABASE_KEY`.
- Added `.env.example`, `.gitignore`, security notes and a public README.
- Kept runtime package structure intact to avoid breaking imports.
- Included PaddleOCR worker, passport parser, face service, decision service and OCR text-line classifier artifacts.
