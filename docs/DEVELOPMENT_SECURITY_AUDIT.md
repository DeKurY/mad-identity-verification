# Development security audit

Назначение: личный отчёт по уязвимостям разработки перед публикацией проекта.

## Итог

Проект можно публиковать в текущей GitHub-ready сборке, если не добавлять старые исходные материалы и реальные runtime-артефакты. Основные проблемы были связаны не с алгоритмом KYC, а с безопасностью разработки: hardcoded key, debug-данные, небезопасные defaults для demo, возможное попадание персональных данных в архив и отсутствие ограничений на uploads.

## Что было найдено и исправлено

| Риск | Где был риск | Что исправлено |
|---|---|---|
| Hardcoded Supabase credentials | старый notebook | В GitHub-ready notebook ключи читаются только из Colab Secrets / env. |
| Реальные/демонстрационные персональные значения в UI | Gradio registration defaults | Заменены на синтетику. |
| Runtime export мог захватить персональные данные | `make_runtime_zip()` | ZIP теперь исключает `.env`, `storage/`, `local_store.json`, uploads, OCR jobs, debug, кэши, ключи и логи. |
| Public Gradio link по умолчанию | `launch_gradio(share=True)` в запуске | Default `share=False`; публичная ссылка включается только вручную. |
| Gradio debug по умолчанию | `debug=True` | Default `debug=False`; включение только через `GRADIO_DEBUG=1`. |
| Неограниченный upload | `_save_input_file()` | Добавлена проверка существования файла, типа изображения, размера, безопасного имени и прав `0600` на сохранённый файл. |
| Local JSON мог сохраняться обычными правами | `identity_store.py` | Запись через временный файл, попытка выставить `0600`, обработка битого JSON. |
| Попытки проверки сохраняли debug/embeddings/пути | `verify_identity()` | Сохранение OCR debug, image paths и дополнительных embeddings сделано opt-in через `STORE_DEBUG_ARTIFACTS=1`. |
| Автоустановка зависимостей | `face_service.py`, diagnostics | Auto repair install выключен по умолчанию; включение только через `MAD_AUTO_INSTALL_DEEPFACE=1`. |
| Supabase RLS отключён в основной схеме | `sql/supabase_schema.sql` | Основная схема теперь secure-by-default: RLS включён. Для быстрой синтетической demo вынесен отдельный `supabase_schema_demo_insecure.sql`. |
| Случайная публикация cache-файлов | `__pycache__`, `.pyc` | Кэши удалены; `.gitignore` и GitHub Actions проверяют отсутствие cache-файлов. |

## Что остаётся архитектурным риском

1. **Биометрические embeddings — чувствительные данные.** Reference embedding нужен для работы, но его нельзя публиковать в дампах БД или `local_store.json`.
2. **`joblib` использует pickle-механику.** Публиковать свой artifact допустимо, но нельзя давать пользователю загружать произвольный `.joblib` и потом делать `joblib.load()`.
3. **OCR/debug outputs могут содержать паспортные данные.** Включать `STORE_DEBUG_ARTIFACTS=1` только для локальной отладки на синтетике.
4. **Colab/Gradio не является production backend.** Для настоящего KYC нужен отдельный backend, auth, audit logging, RLS/policies, encryption и контроль доступа.
5. **Датасет выглядит как реалистичные паспортные записи.** Если есть сомнения, что `data/passport-2000.csv` полностью синтетический, не публиковать его.

## Рекомендации после публикации

- Перевыпустить старый Supabase key/проект, если он когда-либо был в notebook или чате.
- Для GitHub demo использовать только synthetic data.
- Для защиты лучше запускать без Supabase или на отдельном demo-проекте.
- Не включать `share=True` при работе с реальными изображениями.
- Не коммитить `storage/`, `local_store.json`, outputs notebook, OCR CSV/JSON и zip runtime после запуска.
- Добавить `LICENSE`, если проект должен быть открытым для повторного использования.

## Минимальный pre-push security check

```bash
python -m compileall -q mad_colab_pkg training

rg -n --hidden --no-ignore \
  'https://[a-zA-Z0-9-]+\.supabase\.co|eyJ[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN' \
  mad_colab_pkg training notebooks sql || true

find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.env' -o -name 'local_store.json' -o -path './storage/*' \) -print
```
