FROM python:3.10-slim
WORKDIR /app

# Install dependencies first so code edits don't reinstall everything (layer caching).
# CPU-only PyTorch keeps the image small.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
