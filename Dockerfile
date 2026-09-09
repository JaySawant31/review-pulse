FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Needed by numerical Python/PyTorch packages
RUN apt-get update && \
    apt-get install -y \
        --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install \
    --no-cache-dir \
    --upgrade pip


RUN pip install \
    --no-cache-dir \
    -r requirements.txt


COPY app ./app

COPY frontend ./frontend

COPY model ./model

COPY data/latest_168h.csv ./data/latest_168h.csv


EXPOSE 8000


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]