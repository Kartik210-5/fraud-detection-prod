FROM python:3.11-slim

WORKDIR /app

# 1. Copy over just the requirements manifest first
COPY requirements.txt /app/

# 2. Install dependencies (Docker will cache this layer safely now)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the lightweight application files and models
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]