# syntax=docker/dockerfile:1
FROM vggfiit/vgg-torch:latest

# Update apt-get
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    openslide-tools \
    python3-openslide \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for OpenSlide
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /workspace/faster-midog-plus

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps SlideRunner==2.0
USER root