# Colab troubleshooting

## PaddleOCR / PaddleX долго стартует

Это нормально для первого запуска: подтягиваются модели и инициализируется PaddleOCR. В проекте OCR вынесен в `paddle_ocr_worker.py`, поэтому падение OCR не должно убивать основной Gradio UI.

## Ошибка Torch / ModelScope / NCCL

Notebook ставит CPU Torch как compatibility layer для PaddleX/ModelScope. OCR при этом продолжает использовать Paddle GPU, если `paddle.is_compiled_with_cuda()` возвращает `True`.

## OCR работает слишком долго

Проверьте переменные:

```env
OCR_FAST_MODE=1
OCR_QUALITY_PRESETS=0
PADDLE_DET_LIMIT_TYPE=max
PADDLE_DET_LIMIT_SIDE_LEN=1600
```

## Supabase не работает

Проект должен перейти в local JSON fallback. Проверьте вкладку diagnostics в Gradio: там видно `supabase_configured`, текущий `store` и `storage_error`.

## Первая регистрация падает на DeepFace weights

Убедитесь, что выполнена install/warmup часть notebook. Для Colab выставляется:

```env
TF_USE_LEGACY_KERAS=1
DEEPFACE_HOME=/root
MAD_AUTO_INSTALL_DEEPFACE=1
```
