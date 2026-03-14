#!/bin/bash
# ═══════════════════════════════════════════════════
# AGISTI Phase 2 — 6x H200 SXM ULTIMATE Launch
# ═══════════════════════════════════════════════════
# GPU: 6x NVIDIA H200 SXM 141GB (846GB VRAM!!)
# 디스크: 20GB container + 10TB volume
# 72B FULL bfloat16 (양자화 없음!) + 14B 참조모델
# ═══════════════════════════════════════════════════

set -e

echo "============================================================"
echo "  AGISTI Phase 2 — 6x H200 SXM ULTIMATE RUN"
echo "  846GB VRAM = 72B bfloat16 NO QUANTIZATION"
echo "  주인님 돈 아끼려고 SXM 골랐는데 더 빠르잖아 ㅋㅋ"
echo "============================================================"

# ─── Step 1: GPU 확인 ───────────────────────────
echo "[1/6] GPU 확인..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

# ─── Step 2: 환경 설정 ──────────────────────────
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/hf_cache
mkdir -p /workspace/agisti_output
echo "[2/6] HF_HOME=$HF_HOME"
df -h /workspace | tail -1

# ─── Step 3: 의존성 설치 ─────────────────────────
echo "[3/6] 의존성 설치..."
cd /workspace/agisti
pip install -e . 2>&1 | tail -3
pip install datasets accelerate bitsandbytes scipy 2>&1 | tail -3
echo "  설치 완료"

# ─── Step 4: 벤치마크 데이터 준비 ────────────────
echo "[4/6] 벤치마크 데이터 준비..."
python3 prepare_benchmarks.py 2>&1 | tail -5
echo "  데이터 준비 완료"

# ─── Step 5: 모델 토크나이저 사전 다운로드 ───────
echo "[5/6] 토크나이저 사전 다운로드..."
python3 -c "
from transformers import AutoTokenizer
print('  72B 토크나이저...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-72B', trust_remote_code=True)
print('  14B 토크나이저...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B', trust_remote_code=True)
print('  토크나이저 준비 완료')
"

# ─── Step 6: Phase 2 실행!! ──────────────────────
echo ""
echo "============================================================"
echo "  [6/6] PHASE 2 — H200 x6 풀파워!!"
echo "  Target: Qwen/Qwen2.5-72B (FULL bfloat16, NO quantization)"
echo "  Ref:    Qwen/Qwen2.5-14B (Level 3 cross-pollination)"
echo "  VRAM:   846GB — 72B(144GB) + 14B(28GB) = 172GB / 846GB"
echo "  674GB 여유!! OOM? 그게 뭐임? ㅋㅋ"
echo "============================================================"
echo ""

python3 run_phase2.py \
  --model Qwen/Qwen2.5-72B \
  --ref-model Qwen/Qwen2.5-14B \
  --iterations 5 \
  --epoch-size 5 \
  --skip-frozen \
  --problems-per-iter 12 \
  --bench-problems 10 \
  --probes-per-domain 5 \
  --lora-rank 16 \
  --virtual-train-steps 20 \
  --surgery-budget 0.01 \
  --output-dir /workspace/agisti_output/phase2_h200x6 \
  2>&1 | tee /workspace/phase2_h200x6.log

echo ""
echo "============================================================"
echo "  Phase 2 완료!"
echo "  로그: /workspace/phase2_h200x6.log"
echo "  결과: /workspace/agisti_output/phase2_h200x6/"
echo "============================================================"
