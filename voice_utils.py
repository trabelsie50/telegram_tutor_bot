"""voice_utils.py
أدوات الصوت: تحويل الصوت إلى نص باستخدام Whisper المحلي (يعمل بالكامل بدون إنترنت)،
وتحويل النص إلى صوت باستخدام gTTS. يتضمن معالجة مسبقة للملفات الصوتية الواردة من تيليغرام.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
import tempfile
import time
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import whisper
from gtts import gTTS
from telegram import Voice, Audio as TelegramAudio
from telegram.ext import ContextTypes, Update

from config import get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# ثوابت عامة
# ──────────────────────────────────────────────

WHISPER_LANGUAGES = {
    "ar": "العربية",
    "fr": "الفرنسية",
    "en": "الإنجليزية",
}

DEFAULT_WHISPER_MODEL_SIZE = "small"
DEFAULT_LANGUAGE = "ar"
AUDIO_PREFIXES = ("voice_", "tts_", "converted_", "msg_")


# ──────────────────────────────────────────────
# متغير عام لتخزين نموذج Whisper (تحميل واحد فقط)
# ──────────────────────────────────────────────

_whisper_model: Optional[whisper.Whisper] = None


# ──────────────────────────────────────────────
# دوال مساعدة داخلية
# ──────────────────────────────────────────────

def _get_whisper_model_size() -> str:
    """الحصول على حجم نموذج Whisper من الإعدادات أو القيمة الافتراضية."""
    try:
        settings = get_settings()
        if hasattr(settings, "whisper_model_size") and settings.whisper_model_size:
            return settings.whisper_model_size
    except Exception:
        pass
    return DEFAULT_WHISPER_MODEL_SIZE


def _get_whisper_model() -> whisper.Whisper:
    """تحميل نموذج Whisper المحلي وتخزينه مؤقتاً (يُحمَّل مرة واحدة فقط).

    Returns:
        نموذج Whisper محمَّل ومُجهَّز للاستخدام.
    """
    global _whisper_model
    if _whisper_model is None:
        model_size = _get_whisper_model_size()
        settings = get_settings()
        model_path = None
        try:
            if hasattr(settings, "whisper_model_path") and settings.whisper_model_path:
                model_path = settings.whisper_model_path
        except Exception:
            model_path = None

        logger.info(
            f"تحميل نموذج Whisper: الحجم='{model_size}'"
            f"{'، المسار=' + model_path if model_path else ''}"
        )

        if model_path and os.path.exists(model_path):
            logger.info(f"تحميل نموذج Whisper من مسار محلي: {model_path}")
            _whisper_model = whisper.load_model(model_path)
        else:
            if model_path:
                logger.warning(
                    f"المسار المُحدد لنموذج Whisper غير موجود: {model_path}. "
                    f"سيتم تحميل النموذج '{model_size}' تلقائياً."
                )
            _whisper_model = whisper.load_model(model_size)

        logger.info("تم تحميل نموذج Whisper بنجاح.")

    return _whisper_model


def _ensure_ffmpeg_available() -> bool:
    """التحقق من توفر ffmpeg على الجهاز لتحويل الملفات الصوتية.

    Returns:
        True إذا كان ffmpeg متوفراً، False وإلا.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def _ensure_ffprobe_available() -> bool:
    """التحقق من توفر ffprobe على الجهاز.

    Returns:
        bool: True إذا كان ffprobe متاحاً، False وإلا.
    """
    return shutil.which("ffprobe") is not None


def _convert_audio_format(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """تحويل ملف صوتي إلى صيغة WAV متوافقة مع Whisper باستخدام ffmpeg.

    يحوّل الصيغة إلى WAV بمعدل عينة 16000 هرتز وقناة واحدة (أحادي).

    Args:
        input_path: مسار الملف الصوتي الأصلي (OGG, MP3, M4A, إلخ).
        output_path: مسار الملف الناتج (اختياري). إذا لم يُحدَّد، يُنشأ في المجلد المؤقت.

    Returns:
        مسار الملف الصوتي بعد التحويل بصيغة WAV.

    Raises:
        RuntimeError: إذا لم يكن ffmpeg متوفراً أو فشل التحويل.
        FileNotFoundError: إذا لم يكن الملف الأصلي موجوداً.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {input_path}")

    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"converted_{os.getpid()}_{int(time.time())}.wav",
        )

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لتحويل الملفات الصوتية.\n"
            "في Ubuntu/Debian: sudo apt install ffmpeg\n"
            "في macOS: brew install ffmpeg\n"
            "في Windows: حمّله من https://ffmpeg.org/download.html"
        )

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]

    logger.info(f"تحويل الملف الصوتي: {input_path} → {output_path}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
        raise RuntimeError(
            f"فشل تحويل الملف الصوتي باستخدام ffmpeg:\n{error_msg}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"تم تحويل الملف لكن الملف الناتج فارغ أو غير موجود: {output_path}"
        )

    logger.info(f"تم التحويل بنجاح: {output_path}")

    return output_path


def read_audio_file(
    file_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """قراءة ملف صوتي وإرجاع البيانات الصوتية كمصفوفة numpy.

    Args:
        file_path: مسار الملف الصوتي.
        sample_rate: معدل العيّنات المطلوب (الافتراضي: 16000).
        mono: تحويل إلى قناة واحدة إذا كان True (الافتراضي: True).

    Returns:
        tuple: (مصفوفة البيانات الصوتية, معدل العيّنات الفعلي).

    Raises:
        FileNotFoundError: إذا لم يكن الملف موجوداً.
        RuntimeError: إذا فشل قراءة الملف.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {file_path}")

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لقراءة الملفات الصوتية."
        )

    cmd = [
        "ffmpeg",
        "-i", file_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
        "-ac", "1" if mono else "2",
        "-vn",
        "pipe:1",
    ]

    logger.info(f"قراءة الملف الصوتي: {file_path}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
        raise RuntimeError(
            f"فشل قراءة الملف الصوتي باستخدام ffmpeg:\n{error_msg}"
        )

    audio_data = np.frombuffer(result.stdout, dtype=np.float32)

    if audio_data.size == 0:
        raise RuntimeError(f"الملف الصوتي فارغ أو لا يحتوي على بيانات صوتية: {file_path}")

    return audio_data, sample_rate


def get_audio_duration(file_path: str) -> float:
    """الحصول على مدة الملف الصوتي بالثواني.

    Args:
        file_path: مسار الملف الصوتي.

    Returns:
        float: مدة الملف الصوتي بالثواني.

    Raises:
        FileNotFoundError: إذا لم يكن الملف موجوداً.
        RuntimeError: إذا فشل الحصول على المدة.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {file_path}")

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته للحصول على معلومات الملف الصوتي."
        )

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
        raise RuntimeError(
            f"فشل الحصول على مدة الملف الصوتي باستخدام ffprobe:\n{error_msg}"
        )

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            f"تعذر تحليل مدة الملف الصوتي: {result.stdout.strip()}"
        )

    return duration


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """الحصول على معلومات تفصيلية عن الملف الصوتي.

    Args:
        file_path: مسار الملف الصوتي.

    Returns:
        dict: يحتوي على المدة، معدل العيّنات، عدد القنوات، وحجم الملف.

    Raises:
        FileNotFoundError: إذا لم يكن الملف موجوداً.
        RuntimeError: إذا فشل الحصول على المعلومات.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {file_path}")

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته للحصول على معلومات الملف الصوتي."
        )

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,size:stream=sample_rate,channels,codec_name",
        "-of", "json",
        file_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
        raise RuntimeError(
            f"فشل الحصول على معلومات الملف الصوتي:\n{error_msg}"
        )

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("تعذر تحليل معلومات الملف الصوتي كـ JSON.")

    streams = info.get("streams", [])
    format_info = info.get("format", {})

    audio_stream = None
    for stream in streams:
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None and streams:
        audio_stream = streams[0]

    sample_rate = int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
    channels = int(audio_stream.get("channels", 0)) if audio_stream else 0
    duration = float(format_info.get("duration", 0.0))
    size = int(format_info.get("size", 0))

    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "size_bytes": size,
        "codec": audio_stream.get("codec_name", "unknown") if audio_stream else "unknown",
    }


def validate_audio_file(file_path: str) -> bool:
    """التحقق من أن الملف الصوتي صالح وقابل للقراءة.

    Args:
        file_path: مسار الملف الصوتي.

    Returns:
        bool: True إذا كان الملف الصوتي صالحاً، False وإلا.
    """
    if not os.path.exists(file_path):
        return False

    if os.path.getsize(file_path) == 0:
        return False

    try:
        info = get_audio_info(file_path)
        if info["duration"] <= 0 or info["sample_rate"] <= 0:
            return False
    except (RuntimeError, FileNotFoundError, ValueError):
        return False

    return True


def normalize_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_peak_db: float = -3.0,
) -> str:
    """تطبيع مستوى الصوت في الملف الصوتي.

    Args:
        input_path: مسار الملف الصوتي الأصلي.
        output_path: مسار ملف الإخراج (اختياري).
        target_peak_db: مستوى الذروة المستهدف بالديسيبل (الافتراضي: -3.0).

    Returns:
        str: مسار الملف الصوتي بعد التطبيع.

    Raises:
        FileNotFoundError: إذا لم يكن الملف الأصلي موجوداً.
        RuntimeError: إذا فشل التطبيع.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {input_path}")

    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"normalized_{os.getpid()}_{int(time.time())}.wav",
        )

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لتطبيع الملفات الصوتية."
        )

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-af", f"loudnorm=I=-16:TP={target_peak_db}:LRA=11",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]

    logger.info(f"تطبيع مستوى الصوت: {input_path} → {output_path}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
        raise RuntimeError(
            f"فشل تطبيع الملف الصوتي باستخدام ffmpeg:\n{error_msg}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"تم تطبيع الملف لكن الناتج فارغ أو غير موجود: {output_path}"
        )

    logger.info(f"تم تطبيع مستوى الصوت بنجاح: {output_path}")

    return output_path


def trim_silence(
    input_path: str,
    output_path: Optional[str] = None,
    silence_threshold_db: float = -40.0,
    silence_duration_ms: int = 500,
) -> str:
    """إزالة الفترات الصامتة من الملف الصوتي.

    Args:
        input_path: مسار الملف الصوتي الأصلي.
        output_path: مسار ملف الإخراج (اختياري).
        silence_threshold_db: عتبة الصمت بالديسيبل (الافتراضي: -40.0).
        silence_duration_ms: مدة الصمت المطلوبة للإزالة بالميلي ثانية (الافتراضي: 500).

    Returns:
        str: مسار الملف الصوتي بعد إزالة الصمت.

    Raises:
        FileNotFoundError: إذا لم يكن الملف الأصلي موجوداً.
        RuntimeError: إذا فشلت إزالة الصمت.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {input_path}")

    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"trimmed_{os.getpid()}_{int(time.time())}.wav",
        )

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لإزالة الصمت من الملفات الصوتية."
        )

    silence_duration_sec = silence_duration_ms / 1000.0

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-af",
        f"silenceremove=start_periods=1:start_duration={silence_duration_sec}:start_threshold={silence_threshold_db}:detection=peak:stop_periods=-1:stop_duration={silence_duration_sec}:stop_threshold={silence_threshold_db}",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path,
    ]

    logger.info(f"إزالة الصمت: {input_path} → {output_path}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
        raise RuntimeError(
            f"فشل إزالة الصمت باستخدام ffmpeg:\n{error_msg}"
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(
            f"تمت إزالة الصمت لكن الناتج فارغ أو غير موجود: {output_path}"
        )

    logger.info(f"تمت إزالة الصمت بنجاح: {output_path}")

    return output_path


def split_audio_into_segments(
    input_path: str,
    segment_duration_sec: float,
    output_dir: Optional[str] = None,
    overlap_sec: float = 0.0,
) -> List[str]:
    """تقسيم ملف صوتي إلى مقاطع متساوية المدة.

    Args:
        input_path: مسار الملف الصوتي الأصلي.
        segment_duration_sec: مدة كل مقطع بالثواني.
        output_dir: دليل الإخراج (اختياري: مؤقت).
        overlap_sec: التداخل بين المقاطع بالثواني (الافتراضي: 0.0).

    Returns:
        List[str]: قائمة بمسارات المقاطع الناتجة.

    Raises:
        FileNotFoundError: إذا لم يكن الملف الأصلي موجوداً.
        RuntimeError: إذا فشل التقسيم.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {input_path}")

    if segment_duration_sec <= 0:
        raise ValueError("مدة المقطع يجب أن تكون أكبر من صفر.")

    duration = get_audio_duration(input_path)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_segments_")

    os.makedirs(output_dir, exist_ok=True)

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لتقسيم الملفات الصوتية."
        )

    segment_paths: List[str] = []
    segment_index = 0
    current_start = 0.0

    while current_start < duration:
        segment_path = os.path.join(
            output_dir,
            f"segment_{segment_index:04d}.wav",
        )

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ss", str(current_start),
            "-t", str(segment_duration_sec),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            segment_path,
        ]

        logger.info(
            f"تقسيم المقطع {segment_index}: "
            f"من {current_start:.2f} ثانية لمدة {segment_duration_sec} ثانية"
        )

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "خطأ غير معروف"
            raise RuntimeError(
                f"فشل تقسيم الملف الصوتي باستخدام ffmpeg:\n{error_msg}"
            )

        if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
            segment_paths.append(segment_path)
            segment_index += 1
        else:
            logger.warning(f"المقطع {segment_index} فارغ أو غير موجود، تم تخطيه.")

        current_start += segment_duration_sec - overlap_sec

    logger.info(f"تم تقسيم الملف إلى {len(segment_paths)} مقطعاً في: {output_dir}")

    return segment_paths


def concatenate_audio_files(
    input_paths: List[str],
    output_path: str,
) -> str:
    """دمج عدة ملفات صوتية في ملف واحد.

    Args:
        input_paths: قائمة بمسارات الملفات الصوتية المراد دمجها.
        output_path: مسار ملف الإخراج المدمج.

    Returns:
        str: مسار الملف الصوتي المدمج.

    Raises:
        ValueError: إذا كانت القائمة فارغة.
        FileNotFoundError: إذا لم يكن أي ملف موجوداً.
        RuntimeError: إذا فشل الدمج.
    """
    if not input_paths:
        raise ValueError("قائمة الملفات الصوتية فارغة.")

    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"الملف الصوتي غير موجود: {path}")

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لدمج الملفات الصوتية."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    list_file = os.path.join(
        tempfile.gettempdir(),
        f"ffmpeg_concat_{os.getpid()}_{int(time.time())}.txt",
    )

    with open(list_file, "w", encoding="utf-8") as f:
        for path in input_paths:
            abs_path = os.path.abspath(path)
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-b:a", "192k",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"فشل دمج الملفات الصوتية: {e.stderr.decode('utf-8', errors='replace')}"
        ) from e
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)

    if not os.path.exists(output_path):
        raise RuntimeError("لم يتم إنشاء الملف الصوتي المدمج.")

    logger.info(f"تم دمج {len(input_paths)} ملفات في: {output_path}")
    return output_path


def convert_audio_format(
    input_path: str,
    output_path: str,
    format: str = "wav",
) -> str:
    """تحويل ملف صوتي إلى صيغة مختلفة.

    Args:
        input_path: مسار الملف الصوتي الأصلي.
        output_path: مسار ملف الإخراج المحوّل.
        format: الصيغة المطلوبة (wav, mp3, flac, ogg).

    Returns:
        str: مسار الملف الصوتي المحوّل.

    Raises:
        FileNotFoundError: إذا لم يكن الملف الأصلي موجوداً.
        RuntimeError: إذا فشل التحويل.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {input_path}")

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لتحويل الملفات."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "2",
        "-b:a", "192k",
        "-f", format,
        "-y",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"فشل تحويل الملف الصوتي: {e.stderr.decode('utf-8', errors='replace')}"
        ) from e

    if not os.path.exists(output_path):
        raise RuntimeError("لم يتم إنشاء الملف الصوتي المحوّل.")

    logger.info(f"تم تحويل {input_path} إلى {output_path}")
    return output_path


def silence_detection(
    input_path: str,
    silence_threshold: float = -40.0,
    min_silence_duration: float = 0.5,
) -> List[Dict[str, float]]:
    """اكتشاف فترات الصمت في ملف صوتي.

    Args:
        input_path: مسار الملف الصوتي.
        silence_threshold: عتبة الصمت بالديسيبل.
min_silence_duration: أقل مدة للصمت بالثواني.

    Returns:
        قائمة فترات الصمت، كل فترة فيها start و end و duration.

    Raises:
        FileNotFoundError: إذا لم يكن الملف موجوداً.
        RuntimeError: إذا فشل اكتشاف الصمت.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {input_path}")

    if not _ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg غير مثبت على الجهاز. يُرجى تثبيته لاكتشاف الصمت."
        )

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-af", f"silencedetect=noise={silence_threshold}dB:d={min_silence_duration}",
        "-f", "null",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise RuntimeError(f"فشل اكتشاف الصمت: {error_output}") from e

    stderr_text = result.stderr.decode("utf-8", errors="replace")
    silences: List[Dict[str, float]] = []
    current_start = None

    for line in stderr_text.splitlines():
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:", 1)[1].strip())
            except (IndexError, ValueError):
                current_start = None
        elif "silence_end:" in line:
            if current_start is None:
                continue
            try:
                end_time = float(
                    line.split("silence_end:", 1)[1].split("|", 1)[0].strip()
                )
            except (IndexError, ValueError):
                continue

            duration = end_time - current_start
            if "silence_duration:" in line:
                try:
                    duration = float(line.split("silence_duration:", 1)[1].strip())
                except (IndexError, ValueError):
                    duration = end_time - current_start
            elif "duration:" in line:
                try:
                    duration = float(line.split("duration:", 1)[1].strip())
                except (IndexError, ValueError):
                    duration = end_time - current_start

            silences.append(
                {
                    "start": current_start,
                    "end": end_time,
                    "duration": max(0.0, duration),
                }
            )
            current_start = None

    if current_start is not None:
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_path,
            ]
            probe_result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            end_time = float(
                probe_result.stdout.decode("utf-8", errors="replace").strip()
            )
            silences.append(
                {
                    "start": current_start,
                    "end": end_time,
                    "duration": max(0.0, end_time - current_start),
                }
            )
        except Exception:
            pass

    logger.info(f"تم اكتشاف {len(silences)} فترة صمت في {input_path}")
    return silences
