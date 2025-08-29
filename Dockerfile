# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r aegis/requirements.txt

# Install the package
RUN pip3 install .

# Set the entrypoint
ENTRYPOINT ["aegis"]

