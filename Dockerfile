# Dockerfile

# Use Python 3.11-slim as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Create directories for persistent data (vector_db and logs)
RUN mkdir -p vector_db logs

# Copy the requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and static directories (including the knowledge-base)
COPY . /app

# Expose the port used by Gradio
EXPOSE 7860

# Command to run the application
CMD ["python", "v5.py"]

