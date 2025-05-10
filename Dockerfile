# Use a Python image with Rust pre-installed
FROM python:3.10-slim

# Install system dependencies, including Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    cargo \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to cache them and avoid reinstalling dependencies unnecessarily
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application files
COPY . /app/

# Collect static files (optional if you're using Django's staticfiles)
RUN python manage.py collectstatic --noinput || true

# Expose the port for Render
EXPOSE 8000

# Run the application with Gunicorn (for production)
CMD ["gunicorn", "bookrecommender.wsgi:application", "--bind", "0.0.0.0:8000"]
