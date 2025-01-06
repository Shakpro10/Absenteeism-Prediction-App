FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY ./runtime.txt /app/

COPY . /app/

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:${PORT:-8000}", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
