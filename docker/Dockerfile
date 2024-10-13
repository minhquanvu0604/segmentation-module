# Use an official PyTorch image as a base
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    git \
    && apt-get clean

# Install Python dependencies directly
RUN pip install --upgrade pip && \
    pip install \
    matplotlib \
    pandas \
    scikit-learn \
    tqdm \
    opencv-python-headless \
    openpyxl \
    pyyaml \
    albumentations \
    torchsummary \
    segmentation-models-pytorch

# Clean up unnecessary files to reduce image size
RUN ldconfig && \
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Set the entry point to bash for easier debugging and usage
ENTRYPOINT ["/bin/bash"]
