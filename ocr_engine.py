"""
ocr_engine.py - محرك OCR لاستخراج النصوص من صور صفحات الكتب المدرسية التونسية.

يعتمد فقط على Tesseract (خفيف، بدون PyTorch) مع كشف تلقائي للغة (عربية/فرنسية).
تمت إزالة EasyOCR بالكامل: كانت تُحمّل نموذج PyTorch ثقيل تلقائياً في كل مرة
(حتى أثناء كشف اللغة، بدون علاقة بالمحرك المُختار)، وهذا كان السبب الأساسي
في تجاوز حد الذاكرة (Out of Memory) على استضافة Render المجانية (512MB).

يُرجع النص المستخرج مع مصفوفة الثقة لتقييم الحاجة لمراجعة بشرية.
"""

import logging
import os
from typing import Any, Dict, List

from PIL import Image

import pytesseract

logger = logging.getLogger(__name__)

TESSERACT_LANG_ARABIC: str = "ara"
TESSERACT_LANG_FRENCH: str = "fra"
TESSERACT_LANG_BOTH: str = "ara+fra"

LOW_CONFIDENCE_THRESHOLD: float = 60.0


def _compute_average_confidence(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _classify_confidence(score: float) -> str:
    if score >= 80.0:
        return "high"
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"


def detect_language(image_path: str) -> Dict[str, Any]:
    """يكشف تلقائياً لغة النص في الصورة بالاستناد فقط إلى Tesseract (بدون PyTorch)."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"تعذّر فتح الصورة {image_path}: {exc}") from exc

    arabic_scores: List[float] = []
    french_scores: List[float] = []

    for lang_code, bucket in [
        (TESSERACT_LANG_ARABIC, arabic_scores),
        (TESSERACT_LANG_FRENCH, french_scores),
    ]:
        try:
            data = pytesseract.image_to_data(
                img, lang=lang_code, output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for c in data["conf"] if c != "-1" and int(c) > 0]
            if confs:
                bucket.append(_compute_average_confidence(confs))
        except Exception as exc:
            logger.warning("فشل تقييم Tesseract للغة %s: %s", lang_code, exc)

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
        detected_language, avg_arabic, avg_french, confidence,
    )

    return {
        "language": detected_language,
        "confidence": round(confidence, 2),
        "details": details,
    }


def _extract_with_tesseract(image_path: str, lang: str = TESSERACT_LANG_BOTH) -> Dict[str, Any]:
    """يستخرج النص من الصورة باستخدام Tesseract مع درجات الثقة لكل كلمة."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    try:
        img = Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"تعذّر فتح الصورة {image_path}: {exc}") from exc

    try:
        data: Dict[str, Any] = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT
        )
    except Exception as exc:
        raise RuntimeError(f"فشل Tesseract في معالجة {image_path}: {exc}") from exc

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

        confidence_scores.append({
            "word": word,
            "confidence": round(conf_value, 2),
            "classification": _classify_confidence(conf_value),
            "line_number": data["line_num"][i] if "line_num" in data else i,
        })

        if word.strip():
            text_parts.append(word)

    full_text: str = " ".join(text_parts)
    average_confidence: float = _compute_average_confidence(valid_confs)

    logger.info(
        "Tesseract: استُخرج %d كلمة، المعدل: %.1f%%، النص: %s",
        len(text_parts), average_confidence,
        full_text[:200] + ("..." if len(full_text) > 200 else ""),
    )

    return {
        "text": full_text,
        "confidence_scores": confidence_scores,
        "average_confidence": round(average_confidence, 2),
        "engine": "tesseract",
    }


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """يستخرج النص من صورة صفحة كتاب مدرسي تونسي باستخدام Tesseract فقط."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    language_info: Dict[str, Any] = detect_language(image_path)
    detected_lang: str = language_info["language"]

    tesseract_lang: str = TESSERACT_LANG_BOTH
    if detected_lang == "arabic":
        tesseract_lang = TESSERACT_LANG_ARABIC
    elif detected_lang == "french":
        tesseract_lang = TESSERACT_LANG_FRENCH

    result: Dict[str, Any] = _extract_with_tesseract(image_path, lang=tesseract_lang)

    result["language"] = detected_lang
    result["language_details"] = language_info
    result["needs_review"] = result["average_confidence"] < LOW_CONFIDENCE_THRESHOLD

    if result["needs_review"]:
        logger.info(
            "النص المستخرج يحتاج مراجعة بشرية (المعدل: %.1f%% < %.1f%%)",
            result["average_confidence"], LOW_CONFIDENCE_THRESHOLD,
        )

    return result
