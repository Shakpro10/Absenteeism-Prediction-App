# Use Python 3.11 slim version as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY ./requirements.txt /app/

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app/

# Expose the default port that the application will run on
EXPOSE 8080

# Set the default value for PORT if it's not set (for local development)
ENV PORT=8080

# This allows Gunicorn to bind to the PORT provided by the cloud service or fallback to 8080 for local dev
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-8080} -k uvicorn.workers.UvicornWorker main:app"]
