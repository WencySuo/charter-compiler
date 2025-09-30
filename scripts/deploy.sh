#!/usr/bin/env bash
set -euo pipefail

echo "Starting Triton and Prometheus via docker-compose (requires NVIDIA runtime)"
docker compose up -d



