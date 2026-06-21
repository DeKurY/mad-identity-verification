# История разработки и принятые решения

Проект начался как простая проверка лица по двум фотографиям, но для курсовой по методам анализа данных был расширен до KYC-like pipeline: паспорт, selfie, база профилей, OCR, parser, classifier и decision layer.

## Почему не один бинарный classifier

Для паспортной верификации мало правила `same_person = distance <= threshold`. OCR может ошибаться в цифрах, лицо может попасть в пограничную зону, а паспортные данные могут частично совпадать. Поэтому итоговая система использует три статуса:

- `ACCEPT` — данные и лицо уверенно совпали.
- `REVIEW` — часть сигналов спорная, нужна ручная проверка.
- `REJECT` — есть явный конфликт данных или лица.

## Почему DeepFace не дообучается

DeepFace/Facenet512 используется как предобученный feature extractor. Дообучение face recognition требует большого размеченного датасета лиц, корректных pair/triplet loss, контроля утечек и ресурсов. Для учебного проекта честнее использовать готовый extractor и обучать отдельный модуль OCR text-line classifier.

## Почему отказ от YOLO для полей паспорта

YOLO-подход требовал бы разметки каждого паспортного поля и отдельного обучения detector-а. Вместо этого выбран воспроизводимый pipeline: normalization, ROI crop generation, PaddleOCR, text-line classifier, parser и validators.

## Почему PaddleOCR вынесен в worker

PaddleOCR/PaddleX в Colab может нестабильно вести себя при повторной инициализации. Worker-подход изолирует OCR:

```text
Gradio callback -> manifest crop-ов -> paddle_ocr_worker.py -> JSON output -> parser
```

Если worker падает, основной notebook и интерфейс продолжают жить, а ошибка возвращается в debug JSON.

## Что именно обучается

Обучается `training/train_text_classifier.py`: классификатор OCR-строк на synthetic/noisy данных из `passport-2000.csv`. DeepFace и PaddleOCR не обучаются.
