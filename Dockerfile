# Use an official NVIDIA CUDA runtime as a parent image
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenslide-dev \
    sed \
    grep \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install --no-cache-dir uv

# Copy the project files into the container
COPY . /app

# Create a filtered requirements file excluding torch/torchvision/torchaudio
# (these are already installed in the base image)
# Also add dependencies from setup.py/pyproject.toml that might be missing
RUN sed -e '/^torch/d' -e '/^torchvision/d' -e '/^torchaudio/d' \
    aegis/requirements.txt | \
    grep -v "^#" | \
    grep -v "^$" | \
    sed 's/^[[:space:]]*//;s/[[:space:]]*$//' > /tmp/requirements_no_torch.txt && \
    echo "termcolor" >> /tmp/requirements_no_torch.txt && \
    echo "timm" >> /tmp/requirements_no_torch.txt && \
    echo "scikit-image" >> /tmp/requirements_no_torch.txt && \
    echo "pytorch-lightning" >> /tmp/requirements_no_torch.txt

# Install Python dependencies using uv (system-wide, no venv needed in Docker)
RUN uv pip install --system --no-cache -r /tmp/requirements_no_torch.txt

# Install the package using uv
RUN uv pip install --system --no-cache .

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

