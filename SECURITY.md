# Security notes

This repository is a coursework demo, not a production KYC system.

## Do not commit

- `.env` files.
- Supabase service-role keys.
- Real passport photos, selfies, user uploads or OCR outputs.
- `storage/`, `local_store.json`, OCR jobs, Gradio exports and verification logs.
- Real biometric embeddings.

## Supabase keys

The notebook must read credentials from Colab Secrets or environment variables:

```python
from google.colab import userdata
SUPABASE_URL = userdata.get("SUPABASE_URL")
SUPABASE_KEY = userdata.get("SUPABASE_KEY")
```

Use only the anon public key for the demo. Never put a service-role key into Colab UI code or GitHub.

If a key was previously hardcoded in a notebook or committed to Git, treat it as exposed. Create a new Supabase project/key or rotate credentials, and do not store real data in the exposed project.

## RLS warning

`sql/supabase_schema.sql` disables RLS for demo convenience. This is acceptable only for an isolated coursework demo with synthetic data. For anything public or real, enable RLS and create restrictive policies before connecting the app.
