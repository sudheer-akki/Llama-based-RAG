FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Install essential utilities and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
 
# Install PyTorch and dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Set working directory
WORKDIR /app

CMD ["python","main.py"]

