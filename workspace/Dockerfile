# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and data
COPY ./scripts /workspace/scripts
COPY ./data /workspace/data

# Set entrypoint for training script
CMD ["python", "/workspace/scripts/initialTraining.py"]

