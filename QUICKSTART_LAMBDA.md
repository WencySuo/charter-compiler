# Quick Start: Running Tests on Lambda Labs

This guide gets you from zero to running charter-compiler tests on Lambda Labs in ~30 minutes.

## Prerequisites

- Lambda Labs account with GPU instance access
- Local machine with Python 3.10+
- SSH access configured

## Step 1: Launch Lambda Instance (2 min)

```bash
# Via Lambda Labs dashboard (https://cloud.lambdalabs.com)
# 1. Click "Launch Instance"
# 2. Select: 1x A100 (40GB) or H100
# 3. Region: Choose closest
# 4. Click "Launch"
# 5. Note the IP address

export LAMBDA_IP="<your-instance-ip>"
```

## Step 2: Setup Server on Lambda (15 min)

SSH into your Lambda instance and run the setup script:

```bash
# SSH into Lambda
ssh ubuntu@$LAMBDA_IP

# Download setup script
curl -O https://raw.githubusercontent.com/your-repo/charter-compiler/main/scripts/lambda_server_setup.sh

# Run setup (installs Docker, Triton, downloads Llama 3, builds TensorRT engine)
chmod +x lambda_server_setup.sh
./lambda_server_setup.sh

# This script will:
# - Install NVIDIA Container Toolkit
# - Pull Triton + TensorRT-LLM container
# - Download Llama 3 model (requires HF_TOKEN)
# - Build TensorRT engine with cache optimization
# - Start Triton server
```

### Manual Setup Alternative

If you prefer manual setup, follow [LAMBDA_LABS_SETUP.md](./LAMBDA_LABS_SETUP.md) steps 1-4.

## Step 3: Setup Local Environment (5 min)

On your **local machine**:

```bash
# Clone repository
git clone https://github.com/your-repo/charter-compiler.git
cd charter-compiler

# Install dependencies
pip install poetry
poetry install

# Or with pip
pip install -r requirements.txt

# Set Lambda IP
export LAMBDA_IP="<your-instance-ip>"
```

## Step 4: Create SSH Tunnel (1 min)

```bash
# Create SSH tunnel to Lambda Triton server
./scripts/lambda_tunnel.sh

# Expected output:
# ✓ SSH connection successful
# ✓ SSH tunnel established (PID: 12345)
# ✓ Triton server is reachable and ready
```

## Step 5: Run Tests (2 min)

```bash
# Run all tests
python scripts/run_lambda_tests.py

# Or run benchmarks
python scripts/benchmark_lambda.py --iterations 20

# Expected output:
# [Test 1] Sequential Chain - Document Processor
# ✓ Average latency: 0.850s
# ✓ Cache hit rate: 72.5%
# ✓ Throughput: 8.2 docs/sec
#
# [Test 2] Delta Matching - Company Analysis
# ✓ Cache hit rate: 87.3%
# ✓ Later iterations cache: 91.2%
#
# All tests completed successfully!
```

## Quick Commands

### Health Check
```bash
# Check Triton server
curl localhost:8000/v2/health/ready

# Check specific model
curl localhost:8000/v2/models/ensemble/ready
```

### Run Specific Tests
```bash
# Sequential chain only
python scripts/run_lambda_tests.py --test sequential

# Delta matching only  
python scripts/run_lambda_tests.py --test delta

# Custom config
python scripts/run_lambda_tests.py --config configs/lambda_labs.yaml
```

### Benchmarks
```bash
# Cache performance
python scripts/benchmark_lambda.py --benchmark cache --iterations 20

# Batch size optimization
python scripts/benchmark_lambda.py --benchmark batch

# Prefix length impact
python scripts/benchmark_lambda.py --benchmark prefix

# All benchmarks
python scripts/benchmark_lambda.py --benchmark all
```

### Monitor Performance
```bash
# View Prometheus metrics
curl localhost:8002/metrics | grep -E "(cache|latency)"

# GPU utilization (on Lambda)
ssh ubuntu@$LAMBDA_IP "nvidia-smi dmon -s u"

# Triton logs (on Lambda)
ssh ubuntu@$LAMBDA_IP "docker logs -f triton-llama3"
```

## Expected Results

After running tests, you should see:

✅ **Sequential Chain Test**
- Cache hit rate: >70%
- Latency reduction: 40-60%
- TTFT: <1s for cached prompts

✅ **Delta Matching Test**
- Cache hit rate: >85%
- Throughput: 2-3x improvement
- Later iterations: >90% cache hit

## Troubleshooting

### "Connection refused"
```bash
# Check tunnel is running
ps aux | grep ssh
# Restart tunnel
./scripts/lambda_tunnel.sh
```

### "Model not ready"
```bash
# Check Triton on Lambda
ssh ubuntu@$LAMBDA_IP "docker ps | grep triton"
ssh ubuntu@$LAMBDA_IP "docker logs triton-llama3 | tail -20"
```

### Low cache hit rates
```bash
# Verify configuration
curl localhost:8000/v2/models/tensorrt_llm/config | jq '.parameters'
# Should show: enable_kv_cache_reuse=true, tokens_per_block=32
```

## Cleanup

```bash
# Stop tunnel (local)
kill $(cat /tmp/lambda_tunnel.pid)

# Stop Triton (on Lambda)
ssh ubuntu@$LAMBDA_IP "docker stop triton-llama3"

# Terminate Lambda instance (via dashboard or CLI)
# WARNING: This will delete all data on the instance
```

## Next Steps

1. **Customize Tests**: Modify `examples/agents/` for your use case
2. **Optimize Config**: Tune `configs/lambda_labs.yaml` parameters
3. **Scale Up**: Test with larger models (Llama 3-70B)
4. **Production**: Deploy with persistent storage

## Cost Optimization Tips

- Use Lambda Labs preemptible instances (50% cheaper)
- Stop instance when not testing
- Backup TensorRT engines to avoid rebuilding
- Use smallest GPU that fits your model (A10 for Llama 3-8B)

## Support

- Full setup guide: [LAMBDA_LABS_SETUP.md](./LAMBDA_LABS_SETUP.md)
- Technical spec: [TECHNICAL_SPEC.md](./TECHNICAL_SPEC.md)
- Implementation tasks: [IMPLEMENTATION_TASKS.md](./IMPLEMENTATION_TASKS.md)
