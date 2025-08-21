# Python slim base to reduce image size
FROM python:3.10-slim

# Prevents Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application code & assets to keep image lean
COPY app.py ./
COPY models/ models/
COPY templates/ templates/
COPY static/ static/
COPY embed/ embed/

# Required CSV data files (omit large raw files & PDFs)
RUN mkdir -p data
COPY data/Symptom-severity.csv data/
COPY data/symptom_Description.csv data/
COPY data/symptom_precaution.csv data/
COPY data/dataset.csv data/
COPY data/drugsComTrain.csv data/

# (Optional) If embed/ or some data files are absent, container still works (chatbot without retrieval)

# Expose port
ENV PORT=8080
EXPOSE 8080

# Use gunicorn with eventlet for SocketIO
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
