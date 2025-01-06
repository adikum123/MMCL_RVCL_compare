# Use an official NVIDIA base image with CUDA support
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.7 1

# Upgrade pip
RUN pip3 install --upgrade pip

# Install TensorFlow compatible with Python 3.7
RUN pip3 install tensorflow==2.4.0

# Verify installations
RUN python3 --version && \
    pip3 --version && \
    python3 -c "import tensorflow as tf; print(tf.__version__)"

# Set the working directory
WORKDIR /workspace

# Set environment variables for NVIDIA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Entry point
CMD ["bash"]
