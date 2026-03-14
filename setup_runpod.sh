#!/bin/bash
# AGISTI RunPod Setup Script
# Run this on the RunPod pod after uploading agisti.tar.gz

set -e

echo "=== AGISTI RunPod Setup ==="

# 1. System packages
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv > /dev/null 2>&1
echo "[1/5] System packages OK"

# 2. Extract code
cd /workspace
if [ -f agisti.tar.gz ]; then
    tar xzf agisti.tar.gz
    echo "[2/5] Code extracted"
else
    echo "[2/5] ERROR: agisti.tar.gz not found in /workspace"
    exit 1
fi

# 3. Create venv & install deps
cd /workspace/agisti
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
pip install transformers safetensors numpy scipy sentence-transformers pydantic datasets -q
pip install -e . -q
echo "[3/5] Python dependencies installed"

# 4. Verify GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo "[4/5] GPU verified"

# 5. Verify data
if [ -f data/probe_bank.jsonl ] && [ -f data/quick_bench.jsonl ]; then
    PROBES=$(wc -l < data/probe_bank.jsonl)
    BENCH=$(wc -l < data/quick_bench.jsonl)
    echo "[5/5] Data: ${PROBES} probes, ${BENCH} bench problems"
else
    echo "[5/5] WARNING: benchmark data not found, running prepare_benchmarks.py..."
    python3 prepare_benchmarks.py
fi

echo ""
echo "=== Setup Complete ==="
echo "To run Phase 1:"
echo "  cd /workspace/agisti"
echo "  source .venv/bin/activate"
echo "  python run_phase1.py --iterations 50 --epoch-size 10"
echo ""
echo "Quick test (3 iterations):"
echo "  python run_phase1.py --iterations 3 --epoch-size 3 --skip-frozen --model Qwen/Qwen2.5-3B"
