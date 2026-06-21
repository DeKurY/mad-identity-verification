# MAD Identity Verification

Учебный, но рабочий **KYC-like pipeline** для проверки личности по связке **паспорт + selfie + профиль в базе**. Проект объединяет OCR паспортных данных, сравнение лица по embeddings, сверку с хранилищем профилей и итоговый статус `ACCEPT / REVIEW / REJECT`.

Проект подготовлен для запуска в **Google Colab**: интерфейс Gradio, PaddleOCR worker в отдельном процессе, DeepFace/Facenet512 для face embeddings, Supabase или локальный JSON fallback для хранения профилей.

> Это не production KYC и не юридически значимая проверка личности. Это учебная демонстрация архитектуры, методов анализа данных и безопасного decision layer. Не используйте реальные паспорта, selfie и персональные данные в публичной demo-среде.

## Итоговый вывод по проекту

В результате получился работающий прототип KYC-проверки: пользователь регистрирует эталонный профиль, система распознаёт данные паспорта, сравнивает их с профилем, извлекает embeddings лица из эталонного фото, паспорта и selfie, а затем принимает решение. Важная особенность проекта — он не делает опасный бинарный вывод только по одному признаку. OCR, паспортные данные и лицо рассматриваются как независимые сигналы, а спорные случаи отправляются в `REVIEW`.

Сильные стороны реализации:

- полноценный end-to-end pipeline: регистрация → OCR → parser → data matching → face verification → decision;
- отдельный PaddleOCR worker, изолирующий нестабильные OCR-зависимости от основного Gradio-процесса;
- собственный обученный text-line classifier OCR-строк;
- fallback на local JSON, чтобы демонстрация не ломалась при недоступности Supabase;
- безопасная зона `REVIEW` вместо грубого `yes/no`;
- документация по запуску, секретам, безопасности и публикации на GitHub.

Главный практический вывод: даже в учебном KYC нельзя полагаться только на OCR или только на face recognition. Надёжнее строить многоступенчатую проверку, где OCR даёт кандидатов, parser и validators очищают данные, база подтверждает паспортный профиль, а биометрия проверяет согласованность лица.

## Что умеет проект

- Регистрирует профиль: ФИО, дата рождения, серия/номер паспорта, эталонное фото лица.
- Извлекает 512-мерный face embedding через DeepFace/Facenet512.
- Нормализует фото паспорта: EXIF correction, perspective transform, CLAHE, ROI crop generation.
- Запускает OCR паспорта через `paddle_ocr_worker.py` как subprocess.
- Классифицирует OCR-строки собственным `text_line_classifier.joblib`.
- Парсит паспортные поля: ФИО, дату рождения, серию, номер, дату выдачи, код подразделения, MRZ/debug candidates.
- Ищет лучший профиль в Supabase или local JSON.
- Считает cosine distance между тремя источниками лица: reference, passport face, selfie.
- Выдаёт итог `ACCEPT`, `REVIEW` или `REJECT`.

## Архитектура

```text
Регистрация профиля
    ФИО + паспорт + reference photo
        -> DeepFace / Facenet512 embedding
        -> Supabase identity_profiles или local JSON

OCR паспорта
    passport image
        -> auto_normalize_passport
        -> ROI crops: поля, MRZ, vertical ID, face crop
        -> PaddleOCR worker subprocess
        -> raw OCR items
        -> text-line classifier
        -> passport parser
        -> passport_data + confidence + debug

Полная проверка
    passport image + selfie
        -> OCR passport_data
        -> choose_best_profile по паспортным данным
        -> embeddings: passport face, selfie, reference
        -> cosine distances
        -> decision_service
        -> ACCEPT / REVIEW / REJECT
```

## Структура репозитория

```text
mad-identity-verification/
├── mad_colab_pkg/              # основной Python-пакет runtime
│   ├── colab_app.py            # Gradio UI
│   ├── colab_pipeline.py       # orchestration: OCR, parser, face, store, decision
│   ├── image_utils.py          # нормализация паспорта и ROI crop generation
│   ├── paddle_ocr_worker.py    # отдельный OCR subprocess
│   ├── passport_parser.py      # parser паспортных OCR-строк
│   ├── text_classifier.py      # inference обученного OCR-line classifier
│   ├── face_service.py         # DeepFace/Facenet512 embeddings и cosine distance
│   ├── decision_service.py     # data score + face statuses -> final decision
│   └── identity_store.py       # Supabase-first + local JSON fallback
├── training/
│   ├── train_text_classifier.py
│   └── calibrate_face_threshold_celeba.py
├── artifacts/
│   ├── text_line_classifier.joblib
│   ├── text_line_classifier_metrics.json
│   └── text_line_classifier_dataset.csv
├── data/
│   └── passport-2000.csv
├── examples/
├── sql/
│   ├── supabase_schema.sql                 # secure-by-default: RLS enabled
│   └── supabase_schema_demo_insecure.sql   # demo-only: RLS disabled
├── notebooks/
│   └── MAD_Identity_Verification_Final_Defense_Clean.ipynb
├── docs/
├── .github/workflows/security-check.yml
├── .env.example
├── .gitignore
└── README.md
```

## Быстрый запуск в Google Colab

1. Откройте `notebooks/MAD_Identity_Verification_Final_Defense_Clean.ipynb`.
2. Выберите `Runtime -> Change runtime type -> GPU`.
3. Выполните ячейку установки зависимостей.
4. Если Colab попросит restart runtime, перезапустите runtime и продолжите со следующей ячейки.
5. Для Supabase добавьте `SUPABASE_URL` и `SUPABASE_KEY` через Colab Secrets. Без них проект работает через local JSON fallback.
6. Для безопасной Supabase-схемы используйте `sql/supabase_schema.sql`. Для быстрой classroom demo только с синтетикой можно использовать `sql/supabase_schema_demo_insecure.sql`.
7. Запустите Gradio:

```python
from mad_colab_pkg.colab_app import launch_gradio

demo = launch_gradio(share=False)
```

`share=True` включайте только для защиты/демонстрации и только на синтетических данных.

## Supabase и секреты

В notebook нет захардкоженных API keys. Значения читаются через Colab Secrets:

```python
from google.colab import userdata

SUPABASE_URL = userdata.get("SUPABASE_URL")
SUPABASE_KEY = userdata.get("SUPABASE_KEY")
```

Для demo используйте отдельный Supabase-проект и anon key. Не используйте service-role key в notebook, Gradio UI или публичной среде. Если ключ когда-либо попадал в notebook output, Git history или чат, считайте его скомпрометированным и перевыпустите.

## Метрики text-line classifier

Собственное обучение в проекте относится к OCR text-line classifier, а не к DeepFace и не к PaddleOCR.

| Метрика | Значение |
|---|---:|
| Accuracy | 0.8855 |
| Macro F1 | 0.9082 |
| Weighted F1 | 0.8815 |
| Dataset rows | 72000 |
| Train / Test | 57600 / 14400 |
| Split | по `person_id` |

Класс `passport_number` сложнее остальных, потому что отдельная шестизначная строка похожа на коды, даты, шум и MRZ-фрагменты. Поэтому classifier не является единственным источником истины: итоговые серия/номер выбираются parser-ом через ROI + MRZ + weighted consensus.

## Decision logic

```text
FACE_ACCEPT_THRESHOLD = 0.32
FACE_REVIEW_THRESHOLD = 0.43
DATA_ACCEPT_THRESHOLD = 0.80
DATA_REVIEW_THRESHOLD = 0.55
```

```text
если data_score >= 0.80 и обязательные face-сравнения accept:
    ACCEPT
если data_score < 0.55 или обязательное face-сравнение reject:
    REJECT
иначе:
    REVIEW
```

`REVIEW` — штатная безопасная зона для спорного OCR, пограничного лица или неполного совпадения данных.

## Что было усилено перед публикацией

- Удалён хардкод Supabase URL/key из notebook.
- Реальные ФИО и демонстрационные персональные значения заменены синтетикой.
- Gradio `share=False` по умолчанию.
- Gradio `debug=False` по умолчанию.
- Upload-файлы проверяются по расширению и размеру.
- Runtime export исключает `.env`, `storage/`, `local_store.json`, uploads, OCR jobs, debug и кэши.
- Local JSON пишется с правами `0600`, где это поддерживается ОС.
- Сохранение OCR debug, путей к изображениям и дополнительных embeddings в попытках проверки сделано opt-in через `STORE_DEBUG_ARTIFACTS=1`.
- Основная SQL-схема теперь secure-by-default: RLS включён.
- Добавлен GitHub Actions security check.

## Что можно улучшить дальше

- Добавить нормальную авторизацию пользователей и строгие Supabase RLS policies.
- Разделить роли: оператор, пользователь, администратор, аудит.
- Шифровать биометрические embeddings и чувствительные поля на уровне приложения или БД.
- Не хранить selfie/passport embeddings в попытках проверки без необходимости.
- Добавить liveness detection для selfie.
- Добавить детектор подделки/скрина документа.
- Улучшить OCR через дообучение/тонкую настройку на реальных обезличенных примерах документов.
- Добавить unit/integration tests для parser-а, decision_service и identity_store.
- Подключить ruff, mypy, bandit/pip-audit в CI.
- Сделать отдельный backend API вместо прямого Gradio runtime.
- Перевести артефакты модели в безопасный формат, если появится загрузка пользовательских моделей.
- Добавить Docker/Dev Container для воспроизводимого локального запуска.

## Что не коммитить

- `.env`, Colab secrets, Supabase service-role keys.
- `storage/`, `local_store.json`, загруженные паспорта, selfie, debug jobs.
- Реальные документы, реальные biometric embeddings, реальные OCR логи.
- Сырые материалы курсовой, внутренние промпты и черновики, если они не нужны в публичном репозитории.

Перед публикацией см.:

- `docs/PII_CLEANUP_REPORT.md`
- `docs/DEVELOPMENT_SECURITY_AUDIT.md`
- `docs/PUBLISHING_GUIDE.md`

## Как объяснять проект на защите

- DeepFace/Facenet512 используется как предобученный feature extractor, не дообучается.
- PaddleOCR используется как предобученный OCR, не дообучается.
- Собственное обучение — `training/train_text_classifier.py`, OCR text-line classifier.
- CelebA используется только для экспериментальной калибровки face threshold.
- OCR не является источником истины: parser и validators выбирают поля по ROI, MRZ, confidence и domain rules.
- `ACCEPT / REVIEW / REJECT` безопаснее бинарного `yes/no`.

## License

Лицензия не выбрана. Перед публикацией добавьте `LICENSE`, если хотите разрешить другим использовать код.
