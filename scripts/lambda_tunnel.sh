#!/usr/bin/env bash
# SSH tunnel script for connecting to Lambda Labs Triton server

set -euo pipefail

# Configuration
LAMBDA_IP="${LAMBDA_IP:-}"
LAMBDA_USER="${LAMBDA_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"

# Ports
GRPC_PORT=8001
HTTP_PORT=8000
METRICS_PORT=8002

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if LAMBDA_IP is set
if [ -z "$LAMBDA_IP" ]; then
    echo -e "${RED}Error: LAMBDA_IP environment variable not set${NC}"
    echo "Usage: LAMBDA_IP=<your-lambda-ip> $0"
    echo "   or: export LAMBDA_IP=<your-lambda-ip> && $0"
    exit 1
fi

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${YELLOW}Warning: SSH key not found at $SSH_KEY${NC}"
    echo "Using default SSH authentication"
    SSH_KEY=""
fi

# Function to check if port is already forwarded
check_port() {
    local port=$1
    if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check if tunnels already exist
if check_port $GRPC_PORT || check_port $HTTP_PORT || check_port $METRICS_PORT; then
    echo -e "${YELLOW}Existing SSH tunnels detected. Checking...${NC}"
    
    existing_pid=$(ps aux | grep "ssh.*${LAMBDA_IP}.*${GRPC_PORT}" | grep -v grep | awk '{print $2}' | head -1)
    
    if [ -n "$existing_pid" ]; then
        echo -e "${GREEN}Active tunnel found (PID: $existing_pid)${NC}"
        echo "To close existing tunnel: kill $existing_pid"
        
        read -p "Kill existing tunnel and create new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill $existing_pid
            echo "Existing tunnel closed"
            sleep 1
        else
            echo "Keeping existing tunnel"
            exit 0
        fi
    fi
fi

# Build SSH command
SSH_CMD="ssh -N"

if [ -n "$SSH_KEY" ]; then
    SSH_CMD="$SSH_CMD -i $SSH_KEY"
fi

SSH_CMD="$SSH_CMD \
    -L ${GRPC_PORT}:localhost:${GRPC_PORT} \
    -L ${HTTP_PORT}:localhost:${HTTP_PORT} \
    -L ${METRICS_PORT}:localhost:${METRICS_PORT} \
    ${LAMBDA_USER}@${LAMBDA_IP}"

echo -e "${GREEN}Creating SSH tunnels to Lambda Labs Triton Server...${NC}"
echo ""
echo "  Lambda Instance: ${LAMBDA_USER}@${LAMBDA_IP}"
echo ""
echo "  Port Forwarding:"
echo "    gRPC:    localhost:${GRPC_PORT} -> ${LAMBDA_IP}:${GRPC_PORT}"
echo "    HTTP:    localhost:${HTTP_PORT} -> ${LAMBDA_IP}:${HTTP_PORT}"
echo "    Metrics: localhost:${METRICS_PORT} -> ${LAMBDA_IP}:${METRICS_PORT}"
echo ""

# Test connection first
echo "Testing SSH connection..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes ${LAMBDA_USER}@${LAMBDA_IP} echo "Connection OK" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH connection successful${NC}"
else
    echo -e "${RED}✗ SSH connection failed${NC}"
    echo "Please check:"
    echo "  1. LAMBDA_IP is correct: $LAMBDA_IP"
    echo "  2. SSH key is correct: $SSH_KEY"
    echo "  3. Lambda Labs instance is running"
    exit 1
fi

# Create tunnel in background
echo ""
echo "Starting SSH tunnel in background..."
$SSH_CMD &
TUNNEL_PID=$!

# Wait a moment for tunnel to establish
sleep 2

# Verify tunnel is working
if ps -p $TUNNEL_PID > /dev/null; then
    echo -e "${GREEN}✓ SSH tunnel established (PID: $TUNNEL_PID)${NC}"
    echo ""
    echo "Tunnel is running in background. To close:"
    echo "  kill $TUNNEL_PID"
    echo ""
    
    # Save PID to file for easy cleanup
    echo $TUNNEL_PID > /tmp/lambda_tunnel.pid
    echo "PID saved to: /tmp/lambda_tunnel.pid"
    
    # Test Triton server connectivity
    echo ""
    echo "Testing Triton server connectivity..."
    sleep 1
    
    if curl -s http://localhost:${HTTP_PORT}/v2/health/ready > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Triton server is reachable and ready${NC}"
    else
        echo -e "${YELLOW}⚠ Triton server not responding (may still be starting up)${NC}"
        echo "  Check server status: curl localhost:${HTTP_PORT}/v2/health/ready"
    fi
    
    echo ""
    echo -e "${GREEN}Tunnel ready! You can now run tests.${NC}"
    echo ""
    echo "Example commands:"
    echo "  # Health check"
    echo "  curl localhost:${HTTP_PORT}/v2/health/ready"
    echo ""
    echo "  # Run tests"
    echo "  python scripts/run_lambda_tests.py"
    echo ""
    echo "  # Benchmark"
    echo "  python scripts/benchmark_lambda.py --iterations 20"
    echo ""
    
else
    echo -e "${RED}✗ Failed to establish SSH tunnel${NC}"
    exit 1
fi
