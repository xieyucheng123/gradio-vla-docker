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

# Patch gradio_client/utils.py: handle boolean schema
RUN python -c "\
import re, pathlib; \
p = pathlib.Path('/usr/local/lib/python3.10/site-packages/gradio_client/utils.py'); \
s = p.read_text(); \
s = s.replace('def _json_schema_to_python_type(schema: Any, defs) -> str:', 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    if isinstance(schema, bool):\n        return \"Any\" if schema else \"None\"'); \
p.write_text(s)"

# Patch gradio/networking.py: skip localhost check
RUN python -c "\
import pathlib; \
p = pathlib.Path('/usr/local/lib/python3.10/site-packages/gradio/networking.py'); \
s = p.read_text(); \
s = s.replace('def url_ok(url: str) -> bool:', 'def url_ok(url: str) -> bool:\n    return True  # Patched'); \
p.write_text(s)"

WORKDIR /app
COPY gradio_app.py /app/
COPY config.py /app/

EXPOSE 7860
CMD ["python", "gradio_app.py"]
