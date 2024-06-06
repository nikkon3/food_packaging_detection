# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Install the required libraries
RUN pip install --no-cache-dir torch transformers pillow matplotlib

# Define environment variable
ENV MODEL_DIR=/app/models/detr-finetuned-food-packaging

# Run the Python script when the container launches
CMD ["python3", "object_detection.py"]

