# Use an official NVIDIA CUDA runtime as a parent image
FROM nvcr.io/nvidia/pytorch:25.10-py3
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

# To mount your E: drive when running the container, use:
# 
# For WSL/Linux:
#   docker run -it --gpus all -v /mnt/e:/data aegis /bin/bash
#
# For Windows (native Docker Desktop):
#   docker run -it --gpus all -v E:/:/data aegis /bin/bash
#
# Or use docker-compose.yml (see docker-compose.yml file)

