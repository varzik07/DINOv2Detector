# PyTorch official runtime image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies required for OpenCV (often used with Albumentations)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install required python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set default command
CMD ["python", "train.py", "--epochs", "50"]
