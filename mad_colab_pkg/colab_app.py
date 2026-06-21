from pathlib import Path
from typing import Any, Dict


PROJECT_VERSION = "v4_paddle_gpu_torch_repair"


def _compact(obj: Any) -> Any:
    """Avoid rendering gigantic embeddings in Gradio JSON output."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if "embedding" in k.lower():
                if isinstance(v, list):
                    out[k] = f"<embedding:{len(v)}>"
                else:
                    out[k] = "<embedding>"
            elif k in {"ocr_items", "items"} and isinstance(v, list):
                out[k] = v[:20]
            else:
                out[k] = _compact(v)
        return out
    if isinstance(obj, list):
        return [_compact(x) for x in obj[:50]]
    return obj


def launch_gradio(share: bool = False):
    import importlib.util
    import json
    import os
    import sys
    import gradio as gr

    from .colab_pipeline import make_runtime_zip, ocr_passport_image, register_profile, verify_identity, get_settings, get_store
    from .face_service import deepface_status, ensure_deepface_available

    def diagnostics_ui():
        settings = get_settings()
        result: Dict[str, Any] = {
            "ok": True,
            "project_version": PROJECT_VERSION,
            "python": sys.version,
            "base_dir": str(settings.base_dir),
            "ocr_device": settings.ocr_device,
            "supabase_configured": bool(settings.supabase_url and settings.supabase_key),
            "store_debug_artifacts": settings.store_debug_artifacts,
            "max_upload_mb": settings.max_upload_mb,
            "store": "unknown",
            "imports": {},
            "artifacts": {},
        }
        store = get_store(settings)
        result["store"] = "supabase" if store.using_supabase else "local_json"
        result["storage_error"] = store.last_error
        for module in ["cv2", "paddle", "paddleocr", "torch", "modelscope", "deepface", "tensorflow", "tf_keras", "supabase", "sklearn", "gradio"]:
            try:
                spec = importlib.util.find_spec(module)
                result["imports"][module] = "FOUND" if spec else "MISSING"
            except Exception as exc:
                result["imports"][module] = repr(exc)
        try:
            import paddle
            result["paddle_runtime"] = {
                "version": getattr(paddle, "__version__", None),
                "compiled_with_cuda": bool(paddle.is_compiled_with_cuda()),
            }
        except Exception as exc:
            result["paddle_runtime"] = {"error": repr(exc)}

        try:
            import torch
            result["torch_runtime"] = {
                "version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False,
            }
        except Exception as exc:
            result["torch_runtime"] = {"error": repr(exc)}

        result["deepface_status_before_repair"] = deepface_status()
        auto_repair = os.getenv("MAD_AUTO_INSTALL_DEEPFACE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        if not result["deepface_status_before_repair"].get("ok") and auto_repair:
            result["deepface_status_after_repair"] = ensure_deepface_available(auto_install=True)
        else:
            result["deepface_status_after_repair"] = result["deepface_status_before_repair"]
            result["deepface_auto_repair_enabled"] = auto_repair
        for rel in [
            "artifacts/text_line_classifier.joblib",
            "artifacts/text_line_classifier_metrics.json",
            "artifacts/text_line_classifier_dataset.csv",
            "data/passport-2000.csv",
        ]:
            p = settings.base_dir / rel
            result["artifacts"][rel] = {"exists": p.exists(), "size": p.stat().st_size if p.exists() else None}
        return result

    def register_ui(last_name, first_name, middle_name, birth_date, series, number, reference_image):
        if not reference_image:
            return {"ok": False, "error": "Загрузите эталонное фото лица"}
        result = register_profile(last_name, first_name, middle_name, birth_date, series, number, reference_image)
        return _compact(result)

    def ocr_ui(passport_image):
        if not passport_image:
            return {"ok": False, "error": "Загрузите фото паспорта"}, None, None, None, None
        result = ocr_passport_image(passport_image)
        result_compact = _compact(result)
        normalized = result.get("normalized_path")
        face_crop = result.get("passport_face_crop")
        csv_path = result.get("ocr_items_csv")
        classified_csv_path = result.get("ocr_classified_csv")
        return (
            result_compact,
            normalized if normalized and Path(normalized).exists() else None,
            face_crop if face_crop and Path(face_crop).exists() else None,
            csv_path if csv_path and Path(csv_path).exists() else None,
            classified_csv_path if classified_csv_path and Path(classified_csv_path).exists() else None,
        )

    def verify_ui(passport_image, selfie_image):
        if not passport_image or not selfie_image:
            return {"ok": False, "error": "Загрузите фото паспорта и selfie"}
        result = verify_identity(passport_image, selfie_image)
        return _compact(result)

    def export_ui():
        return make_runtime_zip()

    with gr.Blocks(title="MAD Identity Verification Colab") as demo:
        gr.Markdown("""
        # MAD Identity Verification: Colab-only demo
        Вся тяжёлая работа выполняется в Google Colab: PaddleOCR worker, OCR parser, DeepFace, Supabase/local JSON и финальный вердикт.
        Локальный Windows-ПК нужен только как браузер. Вкладка OCR только распознаёт паспорт, а вкладка полной проверки дополнительно сверяет данные с Supabase/local JSON.
        """)
        with gr.Tab("0. Диагностика"):
            gr.Markdown("Сначала нажми эту кнопку. Она показывает, видит ли Gradio-backend DeepFace, PaddleOCR, Supabase и classifier artifacts.")
            diag_btn = gr.Button("Проверить runtime")
            diag_out = gr.JSON(label="Runtime diagnostics")
            diag_btn.click(diagnostics_ui, inputs=None, outputs=diag_out)

        with gr.Tab("1. Регистрация"):
            gr.Markdown("Создаёт эталонный профиль: паспортные данные + embedding эталонного фото лица. Если Supabase не настроен, сохранение идёт в local JSON.")
            with gr.Row():
                last = gr.Textbox(label="Фамилия", value="ИВАНОВ")
                first = gr.Textbox(label="Имя", value="ИВАН")
                middle = gr.Textbox(label="Отчество", value="ИВАНОВИЧ")
            with gr.Row():
                birth = gr.Textbox(label="Дата рождения", value="01.01.1990")
                series = gr.Textbox(label="Серия", value="1234")
                number = gr.Textbox(label="Номер", value="567890")
            ref_img = gr.Image(label="Эталонное фото лица", type="filepath")
            reg_btn = gr.Button("Зарегистрировать профиль")
            reg_out = gr.JSON(label="Результат регистрации")
            reg_btn.click(register_ui, inputs=[last, first, middle, birth, series, number, ref_img], outputs=reg_out)

        with gr.Tab("2. OCR паспорта"):
            gr.Markdown("Проверяет PaddleOCR worker + parser без face verification. Здесь Supabase не используется.")
            passport = gr.Image(label="Фото паспорта", type="filepath")
            ocr_btn = gr.Button("Распознать паспорт")
            ocr_out = gr.JSON(label="OCR + parser result")
            normalized_out = gr.Image(label="Нормализованный паспорт")
            face_crop_out = gr.Image(label="Crop лица из паспорта")
            csv_out = gr.File(label="ocr_items.csv")
            classified_csv_out = gr.File(label="ocr_classified.csv")
            ocr_btn.click(ocr_ui, inputs=passport, outputs=[ocr_out, normalized_out, face_crop_out, csv_out, classified_csv_out])

        with gr.Tab("3. Полная проверка"):
            gr.Markdown("OCR паспорта → поиск профиля в Supabase/local JSON → лицо паспорта/selfie/reference → ACCEPT / REVIEW / REJECT.")
            with gr.Row():
                pass_img = gr.Image(label="Фото паспорта", type="filepath")
                selfie_img = gr.Image(label="Selfie", type="filepath")
            verify_btn = gr.Button("Запустить верификацию")
            verify_out = gr.JSON(label="Final result")
            verify_btn.click(verify_ui, inputs=[pass_img, selfie_img], outputs=verify_out)

        with gr.Tab("4. Экспорт"):
            gr.Markdown("Собирает безопасный ZIP runtime-проекта без .env, storage, local_store.json, загруженных документов, selfie, debug jobs и кэшей.")
            export_btn = gr.Button("Собрать ZIP результатов")
            export_file = gr.File(label="runtime export zip")
            export_btn.click(export_ui, inputs=None, outputs=export_file)

    debug = os.getenv("GRADIO_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    return demo.launch(share=share, debug=debug)
