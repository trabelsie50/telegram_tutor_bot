FROM python:3.10-slim

WORKDIR /app

# تثبيت الأدوات الأساسية والنظامية
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات وتثبيتها بشكل نظيف
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# أمر تشغيل البوت مع الويب سيرفر
CMD ["python", "main.py"]
