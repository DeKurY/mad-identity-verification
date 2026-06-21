# Security notes

This repository is a research and educational prototype, not a production KYC system.

## Do not commit

- `.env` files.
- Supabase service-role keys.
- Real passport photos, selfies, user uploads or OCR outputs.
- `storage/`, `local_store.json`, OCR jobs, Gradio exports and verification logs.
- Real biometric embeddings.

## Credentials

The Colab notebook reads credentials from Colab Secrets or environment variables:

```python
from google.colab import userdata

SUPABASE_URL = userdata.get("SUPABASE_URL")
SUPABASE_KEY = userdata.get("SUPABASE_KEY")
```

Use only the anon public key for the demo runtime. Never put a service-role key into Colab UI code, Gradio callbacks or a public repository.

If a key was previously hardcoded in a notebook or committed to Git, treat it as exposed and rotate it.

## Supabase RLS

`sql/supabase_schema.sql` enables row-level security by default. Add restrictive policies before connecting the app to non-synthetic data.

`sql/supabase_schema_demo_insecure.sql` disables RLS and is intended only for isolated demos with synthetic data.

## Runtime data

The runtime may generate uploads, OCR jobs, cropped images, local JSON stores and logs. These files are intentionally ignored by Git and excluded from safe export. Do not publish them if they contain real documents, faces or embeddings.

## Model artifacts

The included classifier is a `joblib` artifact. Do not load user-supplied pickle/joblib files in a real service because pickle-based formats can execute code during deserialization.
