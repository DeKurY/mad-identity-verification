# Предложение по переименованию файлов

Ниже — безопасный план переименований. Я не переименовывал Python-пакет автоматически, чтобы не сломать импорты в notebook. Лучше делать это отдельным коммитом после проверки.

## Репозиторий

| Сейчас | Лучше |
|---|---|
| `mad_colab_v4_project` | `mad-identity-verification` |
| `MAD_Identity_Verification_Final_Defense_Clean.ipynb` | `notebooks/identity_verification_colab_demo.ipynb` |
| `README_COLAB_ONLY.md` | `docs/colab_runtime_notes.md` |

## Python package

| Сейчас | Лучше | Зачем |
|---|---|---|
| `mad_colab_pkg` | `identity_verification` | понятное имя пакета без привязки к курсовой |
| `colab_app.py` | `gradio_app.py` | UI-слой |
| `colab_pipeline.py` | `verification_pipeline.py` | основной pipeline |
| `image_utils.py` | `passport_image_preprocessing.py` | обработка изображения паспорта |
| `paddle_ocr_worker.py` | `ocr_worker.py` | worker OCR |
| `passport_parser.py` | `passport_parser.py` | уже нормальное имя |
| `text_classifier.py` | `ocr_line_classifier.py` | inference classifier-а |
| `face_service.py` | `face_verification.py` | DeepFace embeddings |
| `decision_service.py` | `decision_engine.py` | слой решения |
| `identity_store.py` | `profile_store.py` | Supabase/local JSON store |

## Training

| Сейчас | Лучше |
|---|---|
| `training/train_text_classifier.py` | `training/train_ocr_line_classifier.py` |
| `training/calibrate_face_threshold_celeba.py` | `training/calibrate_face_thresholds.py` |

## Artifacts

| Сейчас | Лучше |
|---|---|
| `text_line_classifier.joblib` | `ocr_line_classifier.joblib` |
| `text_line_classifier_metrics.json` | `ocr_line_classifier_metrics.json` |
| `text_line_classifier_dataset.csv` | `ocr_line_classifier_dataset.csv` |

## Что нужно поменять после переименования пакета

1. Все импорты `from mad_colab_pkg...`.
2. Notebook cell с запуском Gradio.
3. Пути в `colab_pipeline.py`, если меняются artifact names.
4. README и docs.
5. Прогнать `python -m compileall identity_verification training`.
