# استخدام صورة بايثون الرسمية
FROM python:3.10-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# تثبيت أدوات النظام المطلوبة (Tesseract OCR، اللغات العربية والفرنسية، و FFmpeg)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-fra \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف متطلبات بايثون أولاً للاستفادة من التخزين المؤقت (Cache)
COPY requirements.txt .

# تثبيت مكتبات بايثون
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع إلى الحاوية
COPY . .

# أمر تشغيل البوت عند بدء الحاوية
CMD ["python", "main.py"]
