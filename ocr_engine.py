"""
ocr_engine.py - محرك OCR لاستخراج النصوص من صور صفحات الكتب المدرسية التونسية.

يدعم Tesseract و EasyOCR مع كشف تلقائي للغة (عربية/فرنسية).
يُرجع النص المستخرج مع مصفوفة الثقة لتقييم الحاجة لمراجعة بشرية.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import pytesseract

logger = logging.getLogger(__name__)

# ── الثوابت ──────────────────────────────────────────────────────────────────

SUPPORTED_OCR_ENGINES: List[str] = ["tesseract", "easyocr"]

TESSERACT_LANG_ARABIC: str = "ara"
TESSERACT_LANG_FRENCH: str = "fra"
TESSERACT_LANG_BOTH: str = "ara+fra"

EASYOCR_LANGUAGE_CODES: List[str] = ["ar", "fr"]

LOW_CONFIDENCE_THRESHOLD: float = 60.0

# ── متغير المستوى module لموديل EasyOCR (تحميل كسول) ─────────────────────────

_easyocr_reader: Optional[Any] = None


def _get_easyocr_reader() -> Any:
    """
    يحمّل موديل EasyOCR مرة واحدة فقط ويُعيده في الاستدعاءات اللاحقة.
    يستخدم اللغتين العربية والفرنسية معاً.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr

            logger.info("جارٍ تحميل موديل EasyOCR (قد يستغرق وقتاً في أول مرة)...")
            _easyocr_reader = easyocr.Reader(
                EASYOCR_LANGUAGE_CODES, gpu=False, verbose=False
            )
            logger.info("تم تحميل موديل EasyOCR بنجاح.")
        except Exception as exc:
            logger.error("فشل تحميل EasyOCR: %s", exc)
            raise RuntimeError(
                "تعذّر تحميل موديل EasyOCR. تأكّد من تثبيته وتوفر مكتبات CUDA."
            ) from exc
    return _easyocr_reader


# ── دوال مساعدة داخلية ──────────────────────────────────────────────────────

def _compute_average_confidence(scores: List[float]) -> float:
    """يحسب المعدل الحسابي لمصفوفة الدرجات."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _classify_confidence(score: float) -> str:
    """يصنّف درجة الثقة إلى فئات."""
    if score >= 80.0:
        return "high"
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


# ── كشف اللغة ────────────────────────────────────────────────────────────────

def detect_language(image_path: str) -> Dict[str, Any]:
    """
    يكشف تلقائياً لغة النص في الصورة (عربية، فرنسية، أو مزيج).

    المعاملات:
        image_path: مسار ملف الصورة.

    يُرجع:
        قاموس يحتوي على:
            - "language": اللغة المكتشفة ("arabic", "french", "mixed", "unknown")
            - "confidence": درجة الثقة في الكشف (0-100)
            - "details": قاموس بدرجات كل لغة
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"تعذّر فتح الصورة {image_path}: {exc}") from exc

    arabic_scores: List[float] = []
    french_scores: List[float] = []

    # ── تقييم Tesseract لكل لغة ────────────────────────────────────────────
    for lang_code, bucket in [
        (TESSERACT_LANG_ARABIC, arabic_scores),
        (TESSERACT_LANG_FRENCH, french_scores),
    ]:
        try:
            data = pytesseract.image_to_data(
                img, lang=lang_code, output_type=pytesseract.Output.DICT
            )
            confs = [
                int(c) for c in data["conf"] if c != "-1" and int(c) > 0
            ]
            if confs:
                bucket.append(_compute_average_confidence(confs))
        except Exception as exc:
            logger.warning(
                "فشل تقييم Tesseract للغة %s: %s", lang_code, exc
            )

    # ── تقييم EasyOCR لكل لغة ──────────────────────────────────────────────
    try:
        reader = _get_easyocr_reader()
        results = reader.readtext(img, detail=1, paragraph=False)
        for (_, text, conf) in results:
            text_lower = text.lower()
            has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in text)
            has_french = any(ch.isalpha() and ch.isascii() for ch in text)
            if has_arabic:
                arabic_scores.append(conf * 100)
            if has_french:
                french_scores.append(conf * 100)
    except Exception as exc:
        logger.warning("فشل تقييم EasyOCR لكشف اللغة: %s", exc)

    avg_arabic: float = _compute_average_confidence(arabic_scores)
    avg_french: float = _compute_average_confidence(french_scores)

    details: Dict[str, float] = {
        "arabic": round(avg_arabic, 2),
        "french": round(avg_french, 2),
    }

    if avg_arabic == 0.0 and avg_french == 0.0:
        detected_language: str = "unknown"
        confidence: float = 0.0
    elif avg_arabic > avg_french + 10.0:
        detected_language = "arabic"
        confidence = avg_arabic
    elif avg_french > avg_arabic + 10.0:
        detected_language = "french"
        confidence = avg_french
    else:
        detected_language = "mixed"
        confidence = max(avg_arabic, avg_french)

    logger.info(
        "كشف اللغة: %s (عربية: %.1f%%، فرنسية: %.1f%%، ثقة: %.1f%%)",
        detected_language,
        avg_arabic,
        avg_french,
        confidence,
    )

    return {
        "language": detected_language,
        "confidence": round(confidence, 2),
        "details": details,
    }


# ── استخراج النص بـ Tesseract ────────────────────────────────────────────────

def _extract_with_tesseract(
    image_path: str,
    lang: str = TESSERACT_LANG_BOTH,
) -> Dict[str, Any]:
    """
    يستخرج النص من الصورة باستخدام Tesseract مع درجات الثقة لكل كلمة.

    المعاملات:
        image_path: مسار ملف الصورة.
        lang: كود اللغة لـ Tesseract (افتراضي: عربية+فرنسية).

    يُرجع:
        قاموس يحتوي على:
            - "text": النص المستخرج كنص كامل.
            - "confidence_scores": قائمة قواميس بكل كلمة ودرجة ثقتها.
            - "average_confidence": المعدل العام للثقة.
            - "engine": اسم المحرك المستخدم.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(
            f"تعذّر فتح الصورة {image_path}: {exc}"
        ) from exc

    try:
        data: Dict[str, Any] = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )
    except Exception as exc:
        raise RuntimeError(
            f"فشل Tesseract في معالجة {image_path}: {exc}"
        ) from exc

    confidence_scores: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    valid_confs: List[float] = []

    num_lines: int = len(data["text"])
    for i in range(num_lines):
        word: str = data["text"][i]
        conf: Any = data["conf"][i]

        if word.strip() == "":
            continue

        conf_value: float = float(conf) if conf != "-1" else 0.0

        if conf_value > 0:
            valid_confs.append(conf_value)

        confidence_scores.append(
            {
                "word": word,
                "confidence": round(conf_value, 2),
                "classification": _classify_confidence(conf_value),
                "line_number": data["line_num"][i] if "line_num" in data else i,
            }
        )

        if word.strip():
            text_parts.append(word)

    full_text: str = " ".join(text_parts)
    average_confidence: float = _compute_average_confidence(valid_confs)

    logger.info(
        "Tesseract: استُخرج %d كلمة، المعدل: %.1f%%، النص: %s",
        len(text_parts),
        average_confidence,
        full_text[:200] + ("..." if len(full_text) > 200 else ""),
    )

    return {
        "text": full_text,
        "confidence_scores": confidence_scores,
        "average_confidence": round(average_confidence, 2),
        "engine": "tesseract",
    }


# ── استخراج النص بـ EasyOCR ─────────────────────────────────────────────────

def _extract_with_easyocr(image_path: str) -> Dict[str, Any]:
    """
    يستخرج النص من الصورة باستخدام EasyOCR مع درجات الثقة لكل كتلة نصية.

    المعاملات:
        image_path: مسار ملف الصورة.

    يُرجع:
        قاموس يحتوي على:
            - "text": النص المستخرج كنص كامل.
            - "confidence_scores": قائمة قواميس بكل كتلة نصية ودرجة ثقتها.
            - "average_confidence": المعدل العام للثقة.
            - "engine": اسم المحرك المستخدم.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(
            f"تعذّر فتح الصورة {image_path}: {exc}"
        ) from exc

    try:
        reader = _get_easyocr_reader()
        results: List[Tuple[Any, str, float]] = reader.readtext(
            img, detail=1, paragraph=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"فشل EasyOCR في معالجة {image_path}: {exc}"
        ) from exc

    confidence_scores: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    valid_confs: List[float] = []

    for idx, (bbox, text, conf) in enumerate(results):
        conf_percent: float = conf * 100.0
        valid_confs.append(conf_percent)

        confidence_scores.append(
            {
                "text": text,
                "confidence": round(conf_percent, 2),
                "classification": _classify_confidence(conf_percent),
                "block_index": idx,
            }
        )

        if text.strip():
            text_parts.append(text)

    full_text: str = "\n".join(text_parts)
    average_confidence: float = _compute_average_confidence(valid_confs)

    logger.info(
        "EasyOCR: استُخرج %d كتلة نصية، المعدل: %.1f%%، النص: %s",
        len(text_parts),
        average_confidence,
        full_text[:200] + ("..." if len(full_text) > 200 else ""),
    )

    return {
        "text": full_text,
        "confidence_scores": confidence_scores,
        "average_confidence": round(average_confidence, 2),
        "engine": "easyocr",
    }


# ── الدالة الرئيسية ──────────────────────────────────────────────────────────

def extract_text_from_image(
    image_path: str,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    يستخرج النص من صورة صفحة كتاب مدرسي تونسي باستخدام المحرك المحدد
    أو المحرك الافتراضي من الإعدادات.

    المعاملات:
        image_path: مسار ملف الصورة.
        engine: اسم المحرك ("tesseract" أو "easyocr"). إذا كان None يُستخدم
                المحرك المحدد في الإعدادات (config.py).

    يُرجع:
        قاموس يحتوي على:
            - "text": النص المستخرج.
            - "engine": المحرك المستخدم.
            - "language": اللغة المكتشفة.
            - "confidence_scores": مصفوفة درجات الثقة لكل كلمة/كتلة نصية.
            - "average_confidence": المعدل العام للثقة.
            - "needs_review": True إذا كان النص يحتاج مراجعة بشرية.
            - "language_details": تفاصيل كشف اللغة.
    """
    # ── التحقق من وجود الملف ───────────────────────────────────────────────
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    # ── تحديد المحرك ───────────────────────────────────────────────────────
    selected_engine: str = engine
    if selected_engine is None:
        try:
            from config import get_settings

            selected_engine = get_settings().ocr_engine.lower()
        except Exception:
            selected_engine = "tesseract"

    if selected_engine not in SUPPORTED_OCR_ENGINES:
        logger.warning(
            "محرك OCR غير مدعوم: %s. يتم التراجع إلى tesseract.", selected_engine
        )
        selected_engine = "tesseract"

    logger.info("استخدام محرك OCR: %s للصورة: %s", selected_engine, image_path)

    # ── كشف اللغة ──────────────────────────────────────────────────────────
    language_info: Dict[str, Any] = detect_language(image_path)
    detected_lang: str = language_info["language"]

    # ── اختيار كود اللغة لـ Tesseract ──────────────────────────────────────
    tesseract_lang: str = TESSERACT_LANG_BOTH
    if detected_lang == "arabic":
        tesseract_lang = TESSERACT_LANG_ARABIC
    elif detected_lang == "french":
        tesseract_lang = TESSERACT_LANG_FRENCH

    # ── استخراج النص ───────────────────────────────────────────────────────
    if selected_engine == "easyocr":
        result: Dict[str, Any] = _extract_with_easyocr(image_path)
    else:
        result = _extract_with_tesseract(image_path, lang=tesseract_lang)

    # ── إضافة معلومات إضافية ───────────────────────────────────────────────
    result["language"] = detected_lang
    result["language_details"] = language_info
    result["needs_review"] = result["average_confidence"] < LOW_CONFIDENCE_THRESHOLD

    if result["needs_review"]:
        logger.info(
            "النص المستخرج يحتاج مراجعة بشرية (المعدل: %.1f%% < %.1f%%)",
            result["average_confidence"],
            LOW_CONFIDENCE_THRESHOLD,
        )

    return result
