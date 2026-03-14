#!/bin/bash
# ═══════════════════════════════════════════════════
# AGISTI Phase 2 — 6x H100 Ultimate Launch Script
# ═══════════════════════════════════════════════════
# 예산: $28 / $18/hr ≈ 1시간 30분
# GPU: 6x NVIDIA H100 80GB (480GB VRAM)
# 디스크: 500GB container + 500GB volume = 1TB
# 목표: 72B self-surgery + 7B cross-pollination + GSM8K/ARC 벤치마크
# ═══════════════════════════════════════════════════

set -e

echo "============================================================"
echo "  AGISTI Phase 2 — 6x H100 ULTIMATE RUN"
echo "  Budget: ~1.5 hours. 실패하면: 주인님 돈이 다됐구나~"
echo "============================================================"

# ─── Step 1: 환경 확인 ───────────────────────────
echo "[1/6] GPU 및 환경 확인..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 --version
echo "PyTorch:"
python3 -c "import torch; print(f'  {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

# ─── Step 2: HF 캐시 설정 (디스크 1TB 활용) ─────
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/hf_cache
mkdir -p /workspace/agisti_output
echo "[2/6] HF_HOME=$HF_HOME"
df -h /workspace | tail -1

# ─── Step 3: 코드 설치 ──────────────────────────
echo "[3/6] 코드 설치..."
cd /workspace/agisti
pip install -e . 2>&1 | tail -3
pip install datasets accelerate bitsandbytes scipy 2>&1 | tail -3
echo "  설치 완료"

# ─── Step 4: 벤치마크 데이터 준비 ────────────────
echo "[4/6] 벤치마크 데이터 준비..."
python3 prepare_benchmarks.py 2>&1 | tail -5
echo "  데이터 준비 완료"

# ─── Step 5: 모델 사전 다운로드 ──────────────────
echo "[5/6] 모델 사전 다운로드 (72B + 7B)..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# 72B 토크나이저만 먼저 (모델은 run_phase2.py가 로딩)
t0 = time.time()
print('  72B 토크나이저 다운로드...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-72B', trust_remote_code=True)
print(f'  72B 토크나이저 완료: {time.time()-t0:.0f}s')

# 7B 전체 (참조 모델)
t0 = time.time()
print('  7B 참조 모델 다운로드...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
print(f'  7B 토크나이저 완료: {time.time()-t0:.0f}s')
print('  (모델 가중치는 run_phase2.py에서 자동 다운로드)')
"
echo "  모델 사전 다운로드 완료"

# ─── Step 6: Phase 2 실행!! ──────────────────────
echo ""
echo "============================================================"
echo "  [6/6] PHASE 2 시작!!"
echo "  Target: Qwen/Qwen2.5-72B (8-bit quantized)"
echo "  Ref:    Qwen/Qwen2.5-7B (Level 3 cross-pollination)"
echo "  GPUs:   6x H100 80GB (480GB VRAM)"
echo "  Plan:   5 iterations + GSM8K/ARC pre/post benchmark"
echo "============================================================"
echo ""

python3 run_phase2.py \
  --model Qwen/Qwen2.5-72B \
  --ref-model Qwen/Qwen2.5-7B \
  --load-in-8bit \
  --iterations 5 \
  --epoch-size 5 \
  --skip-frozen \
  --problems-per-iter 12 \
  --bench-problems 10 \
  --probes-per-domain 5 \
  --lora-rank 16 \
  --virtual-train-steps 20 \
  --surgery-budget 0.01 \
  --output-dir /workspace/agisti_output/phase2_h100x6 \
  2>&1 | tee /workspace/phase2_h100x6.log

echo ""
echo "============================================================"
echo "  Phase 2 완료!"
echo "  로그: /workspace/phase2_h100x6.log"
echo "  결과: /workspace/agisti_output/phase2_h100x6/"
echo "============================================================"
