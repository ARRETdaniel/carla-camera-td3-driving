# TD3 Autonomous Vehicle System with CARLA 0.9.16
# SOLUTION: Install Python 3.10 on top of official CARLA 0.9.16 image
# Based on: https://carla.readthedocs.io/en/latest/build_docker/
#
# PROBLEM IDENTIFIED: Official carlasim/carla:0.9.16 has Python 3.8.10
# but CARLA 0.9.16 wheels are built for Python 3.10, 3.11, 3.12 only
# SOLUTION: Install Python 3.10 from deadsnakes PPA

FROM carlasim/carla:0.9.16

# Switch to root for package installations
USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CARLA_ROOT=/workspace

# Install Miniforge for Python 3.10 (uses conda-forge by default, no TOS required)
# Miniforge is a community-driven conda distribution
# Also install xvfb for virtual display in headless mode
# Install Docker CLI for docker-exec mode ROS Bridge integration
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    ca-certificates \
    xvfb \
    mesa-utils \
    apt-transport-https \
    gnupg \
    lsb-release && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniforge (conda-forge by default, no TOS)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create Python 3.10 environment and make it default
RUN conda create -y -n py310 python=3.10 pip && \
    conda clean -afy

# Activate the py310 environment by default
ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/opt/conda/envs/py310
ENV PATH="/opt/conda/envs/py310/bin:$PATH"

# Verify Python version and install pip if needed
RUN python --version && python -m pip --version

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools==69.2.0 wheel

# Install CARLA Python API from the included wheel (Python 3.10 wheel)
RUN python -m pip install /workspace/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# Install Gymnasium (modern OpenAI Gym replacement)
# Required for: src/environment/carla_env.py
RUN python -m pip install gymnasium==0.29.1

# Install PyTorch with CUDA 12.1 support (for RTX 2060 compatibility)
RUN python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# NOTE: We do NOT install rclpy here due to Python version incompatibility
# - CARLA 0.9.16 requires Python 3.10 (for wheels)
# - ROS 2 Foxy (Ubuntu 20.04) requires Python 3.8
# - ROS 2 Humble (Ubuntu 22.04) requires Python 3.10 BUT requires Ubuntu 22.04 base
#
# SOLUTION: Use docker-exec mode for ROS communication (see ros_bridge_interface.py)
# This keeps containers properly separated and avoids dependency conflicts!
#
# Future improvement: Upgrade to Ubuntu 22.04 base + ROS 2 Humble + native rclpy

# Install other Python dependencies
RUN python -m pip install \
    numpy==1.24.3 \
    opencv-python==4.8.1.78 \
    pyyaml==6.0.1 \
    tensorboard==2.15.0 \
    matplotlib==3.7.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.2

# Verify all critical dependencies
RUN python -c "import carla; print(f'CARLA API: Successfully imported')" && \
    python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python -c "import yaml; print(f'PyYAML: {yaml.__version__}')"

# Create project directory
WORKDIR /workspace/av_td3_system

# Copy project files (will be mounted as volume in practice)
# COPY . /workspace/av_td3_system/

# Ensure CARLA server script is executable
RUN chmod +x /workspace/CarlaUE4.sh

# Expose CARLA default ports
EXPOSE 2000 2001

# Set default command (don't auto-start CARLA server - let user control it)
CMD ["/bin/bash"]
