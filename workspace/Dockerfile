# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and data
COPY ./src /workspace/src
COPY ./data /workspace/data

COPY ./src/menu.py /workspace/src/menu.py
CMD ["python", "/workspace/src/menu.py"]

