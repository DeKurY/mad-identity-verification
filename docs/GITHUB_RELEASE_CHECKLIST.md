# GitHub release checklist

Перед публикацией:

- [ ] Открыть notebook и убедиться, что в нем нет реального `SUPABASE_KEY`.
- [ ] Проверить поиск по проекту: `SUPABASE_KEY=`, `eyJ`, `service_role`, `sk-`.
- [ ] Не добавлять `storage/`, реальные паспорта, selfie, OCR jobs и local store.
- [ ] Решить, публиковать ли `data/passport-2000.csv` и `artifacts/text_line_classifier_dataset.csv`.
- [ ] Добавить license, если проект должен быть переиспользуемым.
- [ ] Создать чистый репозиторий и сделать первый commit.

Команды:

```bash
git init
git add .
git status
git commit -m "Initial coursework project cleanup"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

Проверка секретов:

```bash
grep -RInE "SUPABASE_KEY\s*=|eyJ|service_role|sk-" . --exclude-dir=.git
```
