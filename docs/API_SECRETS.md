# Настройка Supabase без хардкода ключей

## Colab Secrets

1. Откройте notebook в Google Colab.
2. Нажмите иконку ключа слева — **Secrets**.
3. Добавьте два значения:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
4. Включите доступ текущего notebook к этим secrets.
5. Запустите ячейку конфигурации.

Код в notebook:

```python
def get_secret(name: str, default: str = "") -> str:
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
```

## Локально через `.env`

```bash
cp .env.example .env
```

Заполните:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key
```

`.env` уже закрыт в `.gitignore`.

## Что делать с уже засвеченным ключом

- Не публиковать старый notebook.
- Удалить ключ из Git history, если он туда попал.
- Пересоздать demo-проект или ключи Supabase.
- Для реальных данных использовать схему с включенным RLS и строгими policies. Demo-схему с отключенным RLS применять только на синтетике.
