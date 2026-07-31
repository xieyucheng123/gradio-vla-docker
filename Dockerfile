FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    gradio==4.44.1 \
    gradio-client==1.3.0 \
    huggingface_hub==0.25.2 \
    requests==2.32.3 \
    numpy==1.26.4 \
    Pillow

WORKDIR /app
COPY gradio_app.py /app/
COPY config.py /app/

EXPOSE 7860
CMD ["python", "gradio_app.py"]
