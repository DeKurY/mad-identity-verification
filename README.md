# MAD Identity Verification

Рабочий **KYC-like pipeline для верификации личности** по фотографии паспорта, selfie и сохраненному профилю пользователя. Проект объединяет OCR паспортных данных, извлечение face embeddings, сверку с базой профилей и консервативный слой принятия решения с тремя исходами: `ACCEPT`, `REVIEW` и `REJECT`.

Runtime рассчитан на запуск в **Google Colab** и включает Gradio-интерфейс, изолированный PaddleOCR worker, DeepFace/Facenet512 для биометрического сравнения и хранилище на Supabase либо локальный JSON fallback.

> Репозиторий является исследовательским и учебным прототипом. Это не сертифицированная промышленная KYC-система, не юридически значимая проверка личности и не production-решение для биометрического контроля доступа.

## Что умеет проект

Система реализует полный сценарий проверки личности:

1. Создает эталонный профиль по паспортным полям и reference photo.
2. Нормализует изображение паспорта и нарезает его на ROI-области.
3. Запускает OCR через отдельный PaddleOCR worker-процесс.
4. Классифицирует OCR-строки и собирает структурированные паспортные поля.
5. Сравнивает извлеченные паспортные данные с сохраненными профилями.
6. Извлекает embeddings лица из reference photo, паспортного фото и selfie.
7. Считает cosine distances и data score.
8. Формирует итоговое решение: `ACCEPT`, `REVIEW` или `REJECT`.

Главная идея проекта: система не доверяет одному признаку. OCR, parser, база данных, face verification и decision layer работают как отдельные этапы. Если данные спорные, результат уходит в `REVIEW`, а не принимается автоматически.

## Возможности

- Предобработка паспорта: учет EXIF-поворота, нормализация перспективы, CLAHE и ROI crop generation.
- OCR через `paddle_ocr_worker.py`, запущенный отдельным subprocess.
- Классификация OCR-фрагментов через обученную модель TF-IDF + SGDClassifier.
- Parser паспортных данных с validators, MRZ-логикой и weighted consensus для серии/номера паспорта.
- Face verification через DeepFace/Facenet512 и cosine distance.
- Хранение профилей и попыток проверки в Supabase или local JSON fallback.
- Gradio UI для регистрации профиля, OCR паспорта, полной проверки и диагностики.
- Безопасная публичная структура репозитория: без ключей, runtime uploads, local stores и реальных документов.

## Архитектура

```text
Регистрация профиля
    ФИО + дата рождения + паспорт + reference photo
        -> DeepFace / Facenet512 embedding
        -> Supabase identity_profiles или local JSON fallback

OCR паспорта
    изображение паспорта
        -> нормализация изображения
        -> ROI crops: поля, MRZ, вертикальный номер, зона фото
        -> PaddleOCR worker subprocess
        -> raw OCR items
        -> text-line classifier
        -> passport parser
        -> structured passport_data + confidence/debug

Полная проверка
    изображение паспорта + selfie
        -> OCR passport_data
        -> выбор лучшего профиля по data score
        -> face embeddings: reference, passport, selfie
        -> cosine distances
        -> decision service
        -> ACCEPT / REVIEW / REJECT
```

## Структура репозитория

```text
mad-identity-verification/
├── mad_colab_pkg/              # основной runtime-пакет
│   ├── colab_app.py            # Gradio UI
│   ├── colab_pipeline.py       # orchestration layer
│   ├── image_utils.py          # предобработка паспорта и crop-ы
│   ├── paddle_ocr_worker.py    # изолированный OCR subprocess
│   ├── passport_parser.py      # парсинг паспортных OCR-строк
│   ├── text_classifier.py      # инференс OCR-line classifier
│   ├── face_service.py         # DeepFace embeddings и cosine distance
│   ├── decision_service.py     # data score + face status -> final decision
│   └── identity_store.py       # Supabase/local JSON storage adapter
├── training/
│   ├── train_text_classifier.py
│   └── calibrate_face_threshold_celeba.py
├── artifacts/
│   ├── text_line_classifier.joblib
│   └── text_line_classifier_metrics.json
├── data/
│   └── passport-2000.csv       # небольшой публичный синтетический sample
├── examples/
├── sql/
│   ├── supabase_schema.sql                 # схема с включенным RLS
│   └── supabase_schema_demo_insecure.sql   # demo-only вариант без RLS
├── notebooks/
│   └── mad_identity_verification_colab.ipynb
├── docs/
│   ├── API_SECRETS.md
│   ├── COLAB_TROUBLESHOOTING.md
│   └── PROJECT_STRUCTURE.md
├── .github/workflows/security-check.yml
├── .env.example
├── SECURITY.md
└── README.md
```

## Быстрый запуск в Google Colab

1. Откройте `notebooks/mad_identity_verification_colab.ipynb`.
2. Выберите `Runtime -> Change runtime type -> GPU`.
3. Запустите ячейки установки зависимостей.
4. Если Colab попросит перезапустить runtime, перезапустите его и продолжите выполнение со следующей ячейки.
5. Добавьте Supabase credentials через Colab Secrets или используйте local JSON fallback без Supabase.
6. Запустите интерфейс:

```python
from mad_colab_pkg.colab_app import launch_gradio

demo = launch_gradio(share=False)
```

`share=True` стоит включать только тогда, когда действительно нужна публичная Gradio-ссылка, и только на синтетических данных.

## Настройка Supabase

Notebook читает ключи из Colab Secrets или environment variables. Ключи не хранятся в репозитории.

```python
from google.colab import userdata

SUPABASE_URL = userdata.get("SUPABASE_URL")
SUPABASE_KEY = userdata.get("SUPABASE_KEY")
```

Рекомендации:

- Используйте отдельный Supabase-проект для экспериментов.
- Для demo runtime используйте anon public key.
- Никогда не добавляйте service-role key в notebook, frontend UI или публичный репозиторий.
- Предпочитайте `sql/supabase_schema.sql`, где включен RLS.
- `sql/supabase_schema_demo_insecure.sql` используйте только для изолированных синтетических демонстраций.

Подробнее: `docs/API_SECRETS.md`.

## OCR text-line classifier

В проекте есть собственная обученная модель, которая классифицирует OCR-фрагменты паспорта по типам строк: фамилия, имя, дата рождения, код подразделения, серия паспорта, номер паспорта, MRZ, noise и т.д.

Pipeline модели:

```text
TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    -> SGDClassifier(loss="modified_huber", class_weight="balanced")
```

Метрики bundled classifier artifact:

| Метрика | Значение |
|---|---:|
| Accuracy | 0.8855 |
| Macro F1 | 0.9082 |
| Weighted F1 | 0.8815 |
| Dataset rows | 72000 |
| Train / Test | 57600 / 14400 |
| Split | по `person_id` |

Classifier не является единственным источником истины. Для чувствительных полей, например серии и номера паспорта, parser дополнительно учитывает confidence OCR, источник crop-а, MRZ-подсказки, вертикальные номера и validators.

## Face verification

Face embeddings извлекаются через DeepFace/Facenet512 и сравниваются по cosine distance.

```text
FACE_ACCEPT_THRESHOLD = 0.32
FACE_REVIEW_THRESHOLD = 0.43
```

Система считает расстояния между:

- лицом из паспорта и reference photo;
- selfie и лицом из паспорта;
- selfie и reference photo.

Чем меньше расстояние, тем выше сходство лиц. Пограничные значения не принимаются автоматически, а переводятся в `REVIEW`.

## Сравнение паспортных данных и итоговое решение

Паспортные поля сравниваются через взвешенный score:

```text
S_data = 0.35 * passport_number_match
       + 0.25 * passport_series_match
       + 0.25 * birth_date_match
       + 0.15 * full_name_similarity
```

```text
DATA_ACCEPT_THRESHOLD = 0.80
DATA_REVIEW_THRESHOLD = 0.55
```

Правила принятия решения:

```text
if data_score >= 0.80 and required face checks are accept:
    ACCEPT
elif data_score < 0.55 or a required face check is reject:
    REJECT
else:
    REVIEW
```

`REVIEW` является нормальным результатом для спорного OCR, неполных данных, плохого качества изображения или пограничного сходства лиц.

## Выводы по проекту

Проект показывает, что рабочий KYC-like сценарий нельзя строить как один вызов OCR или одну проверку лица. Надежнее использовать pipeline, где каждый этап решает отдельную задачу и передает дальше не только результат, но и confidence/debug-информацию.

Основные выводы:

- OCR хорошо извлекает кандидаты, но сам по себе не гарантирует корректность паспортных полей.
- Parser с validators и source-aware scoring снижает риск выбрать случайные цифры, дату или код подразделения как номер паспорта.
- Face verification полезна для проверки биометрической согласованности, но не должна игнорировать конфликт паспортных данных.
- Слой `ACCEPT / REVIEW / REJECT` безопаснее бинарного решения, потому что спорные случаи не принимаются автоматически.
- Изоляция PaddleOCR в отдельный worker повышает стабильность запуска в Colab и упрощает диагностику ошибок.
- Supabase/local JSON fallback делает демонстрацию устойчивой: проект может работать даже без внешней базы.

Итог: это рабочий демонстрационный KYC-like pipeline, который связывает паспортный OCR, ML-классификацию OCR-строк, биометрическое сравнение и сверку с хранилищем профилей в единую систему.

## Ограничения

- Проект не является сертифицированной системой идентификации личности.
- Качество OCR зависит от освещения, размытия, угла съемки, бликов и качества документа.
- Пороги face verification являются консервативными и должны калиброваться под конкретный набор данных.
- Local JSON fallback предназначен только для demo-сценариев.
- В проекте нет liveness detection и защиты от presentation attacks.
- Gradio UI не заменяет production backend и не реализует полноценную пользовательскую аутентификацию.
- Публичные sample-данные и model artifacts нужно перепроверить или заменить перед любым не-demo использованием.

## Что можно улучшить

- Добавить liveness detection для selfie.
- Добавить проверку подлинности документа и screen-recapture detection.
- Заменить фиксированные ROI crop-ы на обученный layout/document detector.
- Расширить validators для паспортных полей и MRZ.
- Добавить более строгие Supabase RLS policies для разных ролей пользователей.
- Шифровать biometric embeddings и чувствительные поля при хранении.
- Добавить unit/integration tests для parser, decision service и storage layer.
- Добавить `ruff`, `mypy`, `bandit` и `pip-audit` в CI.
- Вынести runtime в backend API вместо запуска полной логики прямо из Gradio.
- Заменить pickle/joblib artifacts на более безопасный формат, если появится поддержка пользовательских моделей.
- Упаковать проект в Docker/Dev Container для воспроизводимого локального запуска.

## Безопасность

Не добавляйте в репозиторий:

- `.env` файлы и API-ключи;
- Supabase service-role keys;
- реальные фото паспортов, selfie, OCR outputs и verification logs;
- `storage/`, `local_store.json`, uploads и OCR jobs;
- реальные biometric embeddings.

Подробнее: `SECURITY.md` и `docs/API_SECRETS.md`.

## Лицензия

This project is under the MIT License.