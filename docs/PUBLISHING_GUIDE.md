# Publishing guide

Инструкция для аккуратной публикации GitHub-ready проекта.

## 1. Перед публикацией

Публиковать нужно только очищенную папку `mad-identity-verification/` из GitHub-ready архива.

Не добавлять:

- старые notebook и runtime zip до очистки;
- DOCX/PDF/PPTX защиты с титульными листами и персональными данными;
- `.env`;
- `storage/`, `local_store.json`, OCR jobs, uploads;
- реальные паспорта, selfie, embeddings, debug CSV/JSON;
- Supabase dumps или старые ключи.

## 2. Создать репозиторий

Название:

```text
mad-identity-verification
```

Description:

```text
Coursework KYC-like demo: passport OCR, face verification and Supabase/local JSON identity checks in Google Colab.
```

Topics:

```text
computer-vision, ocr, face-recognition, deepface, paddleocr, gradio, supabase, google-colab, machine-learning, coursework, kyc
```

## 3. Локальная проверка

```bash
cd mad-identity-verification

python -m compileall -q mad_colab_pkg training

rg -n --hidden --no-ignore \
  'https://[a-zA-Z0-9-]+\.supabase\.co|eyJ[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN' \
  mad_colab_pkg training notebooks sql || true

find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.env' -o -name 'local_store.json' -o -path './storage/*' \) -print
```

Если последняя команда что-то вывела — это удалить перед коммитом.

## 4. Первый push

```bash
git init
git branch -M main

git add .
git status

git commit -m "Initial KYC coursework release"
git remote add origin https://github.com/<your-login>/mad-identity-verification.git
git push -u origin main
```

## 5. Красивый release

```bash
git tag -a v0.4.0 -m "KYC coursework demo release"
git push origin v0.4.0
```

На GitHub открыть **Releases -> Draft a new release** и написать:

```text
Initial cleaned coursework release.

Includes:
- Google Colab notebook;
- Gradio demo UI;
- PaddleOCR worker;
- DeepFace/Facenet512 face verification;
- OCR text-line classifier;
- Supabase/local JSON storage;
- ACCEPT / REVIEW / REJECT decision layer;
- security cleanup and publishing notes.
```

## 6. Настройки репозитория

Рекомендуется включить:

- branch protection для `main`;
- GitHub secret scanning;
- Dependabot alerts;
- GitHub Actions.

## 7. После публикации

Проверь, что на GitHub нет:

- notebook outputs;
- `.env`;
- реальных ФИО/паспортов/selfie;
- runtime zip после запуска;
- `storage/`;
- `local_store.json`;
- Supabase keys.

Если что-то попало — не просто удалить новым коммитом, а чистить историю через `git filter-repo`/BFG и перевыпускать ключи.
