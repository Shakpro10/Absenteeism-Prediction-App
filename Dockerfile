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

# Copy the runtime.txt file into the container (optional, but can be useful for Heroku)
COPY ./runtime.txt /app/

# Copy the rest of the application files into the container
COPY . /app/

# Expose the default port that the application will run on
EXPOSE 8000

# Set the default value for PORT if it's not set (for local development)
ENV PORT=8000

# This allows Gunicorn to bind to the PORT provided by the cloud service or fallback to 8000 for local dev
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:${PORT:-8000} -k uvicorn.workers.UvicornWorker main:app"]
