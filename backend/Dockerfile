FROM python:3.12-slim

WORKDIR /app

# Abhängigkeiten zuerst (besseres Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
