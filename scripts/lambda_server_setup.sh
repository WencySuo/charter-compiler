#!/usr/bin/env bash
# Automated setup script for Lambda Labs Triton + TensorRT-LLM deployment
# Run this on your Lambda Labs GPU instance

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORK_DIR="${HOME}/triton-llama3"
HF_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
TENSORRT_LLM_VERSION="v0.9.0"
TRITON_VERSION="24.05"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lambda Labs Triton Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on Lambda Labs
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}Error: NVIDIA GPU not found. Are you on a Lambda Labs GPU instance?${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GPU detected${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check for HuggingFace token
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo "You'll need a HuggingFace token to download Llama 3"
    read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN
    export HF_TOKEN
fi

# Step 1: Install Docker and NVIDIA Container Toolkit
echo -e "\n${BLUE}[1/6] Installing Docker and NVIDIA Container Toolkit...${NC}"

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}✓ Docker installed${NC}"
else
    echo -e "${GREEN}✓ Docker already installed${NC}"
fi

# Install NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
else
    echo -e "${GREEN}✓ NVIDIA Container Toolkit already installed${NC}"
fi

# Verify GPU access in Docker
echo "Verifying GPU access in Docker..."
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && \
    echo -e "${GREEN}✓ GPU access verified in Docker${NC}" || \
    echo -e "${RED}✗ GPU access failed in Docker${NC}"

# Step 2: Pull Triton Container
echo -e "\n${BLUE}[2/6] Pulling Triton + TensorRT-LLM container...${NC}"
echo "This may take 5-10 minutes..."

docker pull nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3

echo -e "${GREEN}✓ Container downloaded${NC}"

# Step 3: Setup Working Directory
echo -e "\n${BLUE}[3/6] Setting up working directory...${NC}"

mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo -e "${GREEN}✓ Working directory: ${WORK_DIR}${NC}"

# Step 4: Download Llama 3 Model
echo -e "\n${BLUE}[4/6] Downloading Llama 3 model...${NC}"

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Skipping model download (no HF_TOKEN)${NC}"
    echo "To download manually later:"
    echo "  export HF_TOKEN='your-token'"
    echo "  git clone https://\$HF_TOKEN@huggingface.co/${HF_MODEL} Llama3/Meta-Llama-3-8B-Instruct"
else
    if [ ! -d "Llama3/Meta-Llama-3-8B-Instruct" ]; then
        echo "Downloading Llama 3 from HuggingFace..."
        sudo apt-get update && sudo apt-get install -y git-lfs
        git lfs install
        
        mkdir -p Llama3
        git clone https://${HF_TOKEN}@huggingface.co/${HF_MODEL} Llama3/Meta-Llama-3-8B-Instruct
        
        echo -e "${GREEN}✓ Model downloaded${NC}"
    else
        echo -e "${GREEN}✓ Model already exists${NC}"
    fi
fi

# Step 5: Build TensorRT Engine
echo -e "\n${BLUE}[5/6] Building TensorRT engine...${NC}"
echo "This will take 10-15 minutes..."

# Clone TensorRT-LLM
if [ ! -d "TensorRT-LLM" ]; then
    git clone -b ${TENSORRT_LLM_VERSION} https://github.com/NVIDIA/TensorRT-LLM.git
fi

# Clone tensorrtllm_backend
if [ ! -d "tensorrtllm_backend" ]; then
    git clone -b ${TENSORRT_LLM_VERSION} https://github.com/triton-inference-server/tensorrtllm_backend.git
fi

# Docker alias for convenience
alias triton-exec="docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3"

# Convert checkpoint
if [ ! -d "Llama3/tllm_checkpoint_1gpu_bf16" ]; then
    echo "Converting checkpoint to TensorRT-LLM format..."
    docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace \
        nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
        python3 TensorRT-LLM/examples/llama/convert_checkpoint.py \
        --model_dir Llama3/Meta-Llama-3-8B-Instruct \
        --output_dir Llama3/tllm_checkpoint_1gpu_bf16 \
        --dtype bfloat16
    
    echo -e "${GREEN}✓ Checkpoint converted${NC}"
else
    echo -e "${GREEN}✓ Checkpoint already converted${NC}"
fi

# Build TensorRT engine
if [ ! -d "Llama3/trt_engines/bf16/1-gpu" ]; then
    echo "Building TensorRT engine with cache optimization..."
    docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace \
        nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
        trtllm-build \
        --checkpoint_dir Llama3/tllm_checkpoint_1gpu_bf16 \
        --output_dir Llama3/trt_engines/bf16/1-gpu \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --max_batch_size 64 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --use_paged_context_fmha enable \
        --enable_context_fmha_fp32_acc \
        --use_prompt_tuning \
        --max_prompt_embedding_table_size 5000
    
    echo -e "${GREEN}✓ TensorRT engine built${NC}"
else
    echo -e "${GREEN}✓ TensorRT engine already exists${NC}"
fi

# Step 6: Configure Triton
echo -e "\n${BLUE}[6/6] Configuring Triton Inference Server...${NC}"

# Copy engines
mkdir -p tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/
cp Llama3/trt_engines/bf16/1-gpu/* \
   tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/

# Set environment variables
export HF_LLAMA_MODEL=${PWD}/Llama3/Meta-Llama-3-8B-Instruct
export ENGINE_PATH=${PWD}/tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1

# Configure model templates
echo "Configuring model templates..."

docker run --rm -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
    python3 tensorrtllm_backend/tools/fill_template.py \
    -i tensorrtllm_backend/all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1

docker run --rm -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
    python3 tensorrtllm_backend/tools/fill_template.py \
    -i tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1

docker run --rm -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
    python3 tensorrtllm_backend/tools/fill_template.py \
    -i tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt \
    triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False

docker run --rm -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
    python3 tensorrtllm_backend/tools/fill_template.py \
    -i tensorrtllm_backend/all_models/inflight_batcher_llm/ensemble/config.pbtxt \
    triton_max_batch_size:64

docker run --rm -v ${PWD}:/workspace -w /workspace \
    nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-trtllm-python-py3 \
    python3 tensorrtllm_backend/tools/fill_template.py \
    -i tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
    triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},\
max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.95,\
exclude_input_in_output:True,enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,\
max_queue_delay_microseconds:0,tokens_per_block:32

# Enable decoupled mode for remote access
cat >> tensorrtllm_backend/all_models/inflight_batcher_llm/ensemble/config.pbtxt << 'EOF'

model_transaction_policy {
  decoupled: True
}
EOF

echo -e "${GREEN}✓ Triton configured${NC}"

# Create startup script
cat > start_triton.sh << 'EOF'
#!/bin/bash
docker run -d \
  --name triton-llama3 \
  --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/tensorrtllm_backend/all_models/inflight_batcher_llm:/models \
  nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 \
  tritonserver \
    --model-repository=/models \
    --exit-on-error=false \
    --strict-readiness=false \
    --log-verbose=1

echo "Triton server starting..."
echo "Check logs: docker logs -f triton-llama3"
echo "Check status: curl localhost:8000/v2/health/ready"
EOF

chmod +x start_triton.sh

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start Triton server:"
echo "  cd ${WORK_DIR}"
echo "  ./start_triton.sh"
echo ""
echo "To check if server is ready:"
echo "  curl localhost:8000/v2/health/ready"
echo ""
echo "To test inference:"
echo "  curl -X POST localhost:8000/v2/models/ensemble/generate -d '{\"text_input\": \"Hello\", \"parameters\": {\"max_tokens\": 20}}'"
echo ""
echo "On your local machine, create SSH tunnel:"
echo "  export LAMBDA_IP=\"$(curl -s ifconfig.me)\""
echo "  ssh -N -L 8001:localhost:8001 -L 8000:localhost:8000 -L 8002:localhost:8002 ubuntu@\$LAMBDA_IP"
echo ""
