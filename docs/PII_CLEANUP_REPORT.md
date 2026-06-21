# PII cleanup report

Назначение: личная проверка перед публикацией проекта на GitHub.

## Что проверялось

Проверялись все файлы GitHub-ready репозитория, включая:

- Python-код;
- notebook;
- Markdown-документацию;
- SQL-файлы;
- JSON-примеры;
- CSV/metrics artifacts;
- бинарный `joblib` через `strings`;
- ZIP-сборку после пересборки.

Отдельно проверялись реальные ФИО, указанные владельцем проекта, их части и возможные латинские варианты. В сам отчёт эти ФИО не внесены, чтобы не вернуть персональные данные обратно в публичный репозиторий.

## Результат

В GitHub-ready репозитории совпадений по целевым реальным ФИО не найдено.

Найденные в `data/passport-2000.csv` имена и отчества являются частью синтетического/учебного датасета и не совпадают с целевыми ФИО. Если происхождение CSV вызывает сомнения, файл лучше не публиковать и оставить только скрипт генерации/обучения и итоговые метрики.

## Что было дополнительно обезличено ранее

- Удалён hardcoded Supabase URL/key из notebook.
- Реальные значения в Gradio demo заменены синтетическими.
- OCR example заменён синтетическим JSON.
- Python cache files удалены.
- Runtime export исключает `storage/`, uploads, OCR jobs и локальные хранилища.

## Что нельзя добавлять обратно

- старые DOCX/PDF/PPTX защиты с титульными листами;
- старые notebook/zip-архивы до очистки;
- реальные фото паспорта и selfie;
- `storage/local_store.json`;
- OCR debug CSV/JSON с реальных документов;
- Supabase dumps с профилями или verification attempts.

## Команды для повторной проверки

```bash
# Поиск типовых секретов в исполняемой части проекта
rg -n --hidden --no-ignore \
  'https://[a-zA-Z0-9-]+\.supabase\.co|eyJ[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN' \
  mad_colab_pkg training notebooks sql || true

# Поиск runtime-мусора
find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.env' -o -name 'local_store.json' -o -path './storage/*' \) -print
```
