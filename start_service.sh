#!/bin/bash

# Parakeet Large ASR Service Startup Script
# This script starts the FastAPI server for Parakeet Large ASR model

# Exit on any error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Parakeet Large ASR Service...${NC}"

# Load environment variables from config.env if it exists
if [ -f config.env ]; then
    export $(cat config.env | grep -v '^#' | xargs)
fi

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check and create virtual environment if it doesn't exist
VENV_PATH="/home/naresh/venvs/parakeet-service"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating it...${NC}"
    # Ensure venvs directory exists
    mkdir -p /home/naresh/venvs
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}Virtual environment created${NC}"
    
    # Activate and install dependencies
    source "$VENV_PATH/bin/activate"
    echo -e "${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
    pip install --upgrade pip
    cd /home/naresh/parakeet-service
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
else
    # Activate existing virtual environment
    source "$VENV_PATH/bin/activate"
    echo -e "${YELLOW}Virtual environment activated${NC}"
fi

# Datasets server configuration
DATASETS_DIR="/home/naresh/datasets"
DATASETS_PORT=8080

# Check if port 8080 is already in use
if lsof -Pi :$DATASETS_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}HTTP server already running on port $DATASETS_PORT - reusing existing server${NC}"
    echo "  Media files accessible at: http://localhost:$DATASETS_PORT/"
else
    echo -e "${GREEN}Starting HTTP server for datasets folder...${NC}"
    echo "  Directory: $DATASETS_DIR"
    echo "  Port: $DATASETS_PORT"
    
    # Check if datasets directory exists
    if [ ! -d "$DATASETS_DIR" ]; then
        echo -e "${YELLOW}Warning: Directory $DATASETS_DIR does not exist. Creating it...${NC}"
        mkdir -p "$DATASETS_DIR"
    fi
    
    # Start Python HTTP server in background with nohup for proper detachment
    cd "$DATASETS_DIR"
    nohup python3 -m http.server $DATASETS_PORT > /tmp/datasets_server.log 2>&1 &
    DATASETS_SERVER_PID=$!
    echo $DATASETS_SERVER_PID > /tmp/datasets_server.pid
    echo -e "${GREEN}HTTP server started with PID: $DATASETS_SERVER_PID${NC}"
    echo "  Media URLs: http://localhost:$DATASETS_PORT/<filename>"
    
    # Return to service directory
    cd /home/naresh/parakeet-service
    
    # Give the HTTP server a moment to start and verify it's running
    sleep 3
    
    # Verify the server is actually running
    if lsof -Pi :$DATASETS_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}HTTP server verified running on port $DATASETS_PORT${NC}"
    else
        echo -e "${YELLOW}Warning: HTTP server may not have started properly. Check /tmp/datasets_server.log${NC}"
        if [ -f /tmp/datasets_server.log ]; then
            echo "  Last few lines of server log:"
            tail -5 /tmp/datasets_server.log | sed 's/^/    /'
        fi
    fi
fi

echo ""

# Create logs directory if it doesn't exist
mkdir -p /home/naresh/parakeet-service/logs

# Service configuration
PORT=${PORT:-8006}
HOST=${HOST:-0.0.0.0}

echo -e "${GREEN}Starting FastAPI server with following configuration:${NC}"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo ""
echo -e "${YELLOW}Service will be accessible at: http://localhost:$PORT${NC}"
echo -e "${YELLOW}API documentation at: http://localhost:$PORT/docs${NC}"
echo -e "${YELLOW}Logs will be saved to: /home/naresh/parakeet-service/logs/service.log${NC}"
echo ""

# Start uvicorn server with logging (output to both console and log file)
uvicorn app:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    2>&1 | tee /home/naresh/parakeet-service/logs/service.log

