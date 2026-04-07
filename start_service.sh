#!/bin/bash

# Parakeet Large ASR Service Startup Script
# This script starts the FastAPI server for Parakeet Large ASR model

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Resolve the directory this script lives in — all paths are relative to it
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo -e "${GREEN}Starting Parakeet Large ASR Service...${NC}"

# Set CUDA environment — uses system symlink, always points to installed version
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Pin to GPU 4 (GPUs 0-3 are reserved for MinMax)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

# Redirect HuggingFace and NeMo model downloads to the dedicated data disk
export HF_HOME=/mnt/hf-cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/hf-cache/huggingface/hub
export NEMO_CACHE_DIR=/mnt/hf-cache/nemo

# Load environment variables from config.env if it exists
if [ -f "${SCRIPT_DIR}/config.env" ]; then
    export $(cat "${SCRIPT_DIR}/config.env" | grep -v '^#' | sed 's/#.*//' | xargs)
fi

# -----------------------------------------------------------------------
# Virtual environment — auto-create if it doesn't exist
# -----------------------------------------------------------------------
VENV_PATH="${HOME}/venvs/parakeet-service"

if [ ! -d "${VENV_PATH}" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating it...${NC}"
    mkdir -p "${HOME}/venvs"
    python3 -m venv "${VENV_PATH}"
    echo -e "${GREEN}Virtual environment created at: ${VENV_PATH}${NC}"

    source "${VENV_PATH}/bin/activate"
    echo -e "${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
    pip install --upgrade pip
    cd "${SCRIPT_DIR}"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
else
    source "${VENV_PATH}/bin/activate"
    echo -e "${YELLOW}Virtual environment activated${NC}"
fi

# -----------------------------------------------------------------------
# Service configuration — load from config.env or use defaults
# -----------------------------------------------------------------------
PORT=${PORT:-8006}
HOST=${HOST:-0.0.0.0}

# Create logs directory relative to the script location
mkdir -p "${SCRIPT_DIR}/logs"

echo ""
echo -e "${GREEN}Starting FastAPI server with the following configuration:${NC}"
echo "  Port:  $PORT"
echo "  Host:  $HOST"
echo ""
echo -e "${YELLOW}Service will be accessible at:   http://localhost:$PORT${NC}"
echo -e "${YELLOW}API documentation at:            http://localhost:$PORT/docs${NC}"
echo -e "${YELLOW}Logs will be saved to:           ${SCRIPT_DIR}/logs/service.log${NC}"
echo -e "${YELLOW}Model cache:                     ${NEMO_CACHE_DIR}${NC}"
echo ""

# Start uvicorn server — log to both console and file
uvicorn app:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    2>&1 | tee "${SCRIPT_DIR}/logs/service.log"
