#!/usr/bin/env python3
"""
AGISTI Phase 1 -- GPU Surgery Training
=======================================

RunPod RTX 5090 + Qwen 2.5 7B
Goal: Real AGISTI surgery iterations with activation tracing

Usage:
  python run_phase1.py                          # Default: 50 iterations
  python run_phase1.py --iterations 200         # More iterations
  python run_phase1.py --model Qwen/Qwen2.5-3B  # Smaller model
  python run_phase1.py --skip-frozen            # Skip frozen discovery
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("agisti.phase1")


def print_banner():
    print("""
    ___    ____________ __________
   /   |  / ____/  _/ // /_  __/ /
  / /| | / / __ / / \\_ \\  / / / /
 / ___ |/ /_/ // / ___/ // / / /
/_/  |_|\\____/___//____//_/ /_/

  Phase 1: GPU Surgery Training
  RunPod RTX 5090 + Real Activation Tracing
""")


def parse_args():
    p = argparse.ArgumentParser(description="AGISTI Phase 1 -- GPU Surgery")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B",
                   help="Model name (default: Qwen/Qwen2.5-7B)")
    p.add_argument("--iterations", type=int, default=50,
                   help="Total iterations (default: 50)")
    p.add_argument("--epoch-size", type=int, default=10,
                   help="Iterations per epoch (default: 10)")
    p.add_argument("--output-dir", default="agisti_output",
                   help="Output directory")
    p.add_argument("--skip-frozen", action="store_true",
                   help="Skip frozen zone discovery")
    p.add_argument("--probe-bank", default="data/probe_bank.jsonl")
    p.add_argument("--bench-data", default="data/quick_bench.jsonl")
    p.add_argument("--max-gen-tokens", type=int, default=128,
                   help="Max generation tokens (default: 128)")
    p.add_argument("--problems-per-iter", type=int, default=10,
                   help="Problems per iteration (default: 10)")
    p.add_argument("--bench-problems", type=int, default=20,
                   help="Bench problems per domain (default: 20)")
    p.add_argument("--lora-rank", type=int, default=8,
                   help="LoRA rank for surgery (default: 8)")
    p.add_argument("--virtual-train-steps", type=int, default=20,
                   help="Virtual training steps (default: 20)")
    p.add_argument("--probes-per-domain", type=int, default=10,
                   help="Probes per domain (default: 10)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint directory")
    return p.parse_args()


# ----------------------------------------------------------------
# Step 1: GPU setup + model loading
# ----------------------------------------------------------------
def setup_gpu():
    """Detect and configure GPU."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected! Falling back to CPU.")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    total_vram = 0.0
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_vram += gpu_mem
        logger.info("GPU %d: %s (%.1f GB)", i, gpu_name, gpu_mem)
    if n_gpus > 1:
        logger.info("Total VRAM: %.1f GB across %d GPUs", total_vram, n_gpus)

    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return torch.device("cuda")


def load_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[1/7] Loading model: %s on %s", model_name, device)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GPU: use bfloat16 for RTX 5090
    if device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else str(device),
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  Model loaded: %.1fM params, dtype=%s (%.1fs)",
        n_params / 1e6, dtype, time.time() - t0,
    )

    if device.type == "cuda":
        mem_used = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
        mem_total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9
        logger.info("  GPU memory: %.1f / %.1f GB (across %d GPUs)", mem_used, mem_total, torch.cuda.device_count())

    return model, tokenizer


# ----------------------------------------------------------------
# Step 2: Load data
# ----------------------------------------------------------------
def load_data(probe_path: str, bench_path: str):
    from agisti.probe.active_prober import ProbeBank
    from agisti.benchmark.quick_bench import QuickBenchSuite

    logger.info("[2/7] Loading benchmark data")

    pp = Path(probe_path)
    bp = Path(bench_path)
    if not pp.exists():
        logger.error("Probe bank not found: %s", pp)
        sys.exit(1)
    if not bp.exists():
        logger.error("Bench data not found: %s", bp)
        sys.exit(1)

    probe_bank = ProbeBank.from_jsonl(pp)
    bench_suite = QuickBenchSuite.from_jsonl("phase1", bp)

    logger.info(
        "  Probes: %d across %s",
        probe_bank.total_probes,
        probe_bank.domains,
    )
    logger.info(
        "  Bench:  %d across %s",
        bench_suite.total,
        bench_suite.domains,
    )
    return probe_bank, bench_suite


# ----------------------------------------------------------------
# Step 3: Frozen zone discovery (with GPU eval)
# ----------------------------------------------------------------
def run_frozen_discovery(model, tokenizer, probe_bank, device):
    from agisti.frozen.discovery import FrozenZoneDiscovery
    from agisti.frozen.mask import FrozenMask
    from agisti.types import FreezeLevel

    logger.info("[3/7] Running frozen zone discovery...")
    t0 = time.time()

    discovery = FrozenZoneDiscovery(
        noise_scale=0.1,
        sensitivity_threshold=0.05,
        partial_freeze_threshold=0.02,
    )

    def eval_fn(m):
        correct = 0
        total = 0
        for domain in probe_bank.domains[:3]:
            probes = probe_bank.get_probes(domain, count=5, seed=42)
            for probe in probes:
                total += 1
                try:
                    prompt = f"Question: {probe.question}\n\nAnswer:"
                    inputs = tokenizer(
                        prompt, return_tensors="pt",
                        truncation=True, max_length=512,
                    )
                    input_ids = inputs["input_ids"].to(device)
                    with torch.no_grad():
                        outputs = m.generate(
                            input_ids=input_ids,
                            max_new_tokens=64,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    new_tokens = outputs[0][input_ids.shape[1]:]
                    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    if probe.verify(answer):
                        correct += 1
                except Exception:
                    pass
        return correct / total if total > 0 else 0.0

    report = discovery.discover(model=model, eval_fn=eval_fn)
    logger.info("  Discovery took %.1fs", time.time() - t0)

    try:
        logger.info("  %s", report.summary())
    except Exception:
        pass

    mask = FrozenMask()
    for layer_name in report.frozen_layers:
        mask.freeze_layer(model, layer_name, FreezeLevel.FULL)
    for layer_name in report.partially_frozen_layers:
        mask.freeze_layer(model, layer_name, FreezeLevel.PARTIAL)

    logger.info(
        "  Frozen: %d layers, Trainable: %d layers",
        len(mask.frozen_layers), len(mask.trainable_layers),
    )
    return mask


# ----------------------------------------------------------------
# Step 4: Build components (GPU-optimized)
# ----------------------------------------------------------------
def build_components(args, model, tokenizer, probe_bank, bench_suite, frozen_mask):
    from agisti.probe.active_prober import ActiveProber
    from agisti.generation.generator import ProblemGenerator
    from agisti.evaluation.evaluator import ModelEvaluator
    from agisti.generation.verification import AnswerVerifier
    from agisti.surgery.proposer import SurgeryProposer
    from agisti.surgery.virtual_trainer import VirtualTrainer
    from agisti.surgery.applicator import DeltaApplicator
    from agisti.benchmark.quick_bench import QuickBench
    from agisti.config import QuickBenchConfig

    logger.info("[4/7] Building AGISTI components (GPU mode)")

    verifier = AnswerVerifier()

    prober = ActiveProber(
        probe_bank=probe_bank,
        probes_per_domain=args.probes_per_domain,
        max_gen_tokens=args.max_gen_tokens,
    )

    # Phase 1: Use model itself as teacher for problem generation
    generator = ProblemGenerator(
        teacher_model=model,
        teacher_tokenizer=tokenizer,
        max_gen_tokens=1024,
        temperature=0.8,
    )

    evaluator = ModelEvaluator(
        verifier=verifier,
        max_gen_tokens=args.max_gen_tokens,
        batch_size=4,  # GPU can handle bigger batches
    )

    proposer = SurgeryProposer(
        lora_rank=args.lora_rank,
        budget=0.01,
    )

    frozen_names = set(frozen_mask.frozen_layers) if frozen_mask else set()

    virtual_trainer = VirtualTrainer(
        max_steps=args.virtual_train_steps,
        tokenizer=tokenizer,
    )

    applicator = DeltaApplicator(
        model=model,
        frozen_layer_names=frozen_names,
    )

    qb_config = QuickBenchConfig(sample_per_domain=args.bench_problems)
    quick_bench = QuickBench(
        suite=bench_suite,
        config=qb_config,
        verifier=verifier,
        max_gen_tokens=args.max_gen_tokens,
    )

    logger.info("  All components built successfully")

    return {
        "prober": prober,
        "generator": generator,
        "evaluator": evaluator,
        "verifier": verifier,
        "proposer": proposer,
        "virtual_trainer": virtual_trainer,
        "applicator": applicator,
        "quick_bench": quick_bench,
        "frozen_mask": frozen_mask,
    }


# ----------------------------------------------------------------
# Step 5: Run orchestrator
# ----------------------------------------------------------------
def run_orchestrator(args, model, tokenizer, components, device):
    from agisti.orchestrator.orchestrator import AGISTIOrchestrator
    from agisti.config import (
        PhaseConfig, IterationConfig, CheckpointConfig,
    )
    from agisti.types import PhaseId

    logger.info("[5/7] Starting AGISTI orchestrator (Phase 1)")

    iter_config = IterationConfig(
        problems_per_iteration=args.problems_per_iter,
        virtual_train_steps=args.virtual_train_steps,
        checkpoint_every=args.epoch_size,
        lora_rank=args.lora_rank,
        trace_activations=True,  # Enable activation tracing for real surgery
    )
    max_epochs = (args.iterations + args.epoch_size - 1) // args.epoch_size

    phase_config = PhaseConfig(
        phase=PhaseId.PHASE_1,
        model_name=args.model,
        target_iterations=args.iterations,
        epoch_size=args.epoch_size,
        max_epochs=max_epochs,
        iterations_per_epoch=args.epoch_size,
    )
    ckpt_config = CheckpointConfig(keep_last_n=5)
    output_dir = Path(args.output_dir)

    logger.info("  Phase:      1 (GPU Surgery)")
    logger.info("  Model:      %s", args.model)
    logger.info("  Iterations: %d (%d epochs x %d)", args.iterations, max_epochs, args.epoch_size)
    logger.info("  LoRA rank:  %d", args.lora_rank)
    logger.info("  VTrain:     %d steps", args.virtual_train_steps)
    logger.info("  Output:     %s", output_dir)
    logger.info("  Device:     %s", device)

    orchestrator = AGISTIOrchestrator(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        phase_config=phase_config,
        iteration_config=iter_config,
        checkpoint_config=ckpt_config,
        device=device,
        **components,
    )

    stats = orchestrator.run(
        max_epochs=max_epochs,
        max_iterations=args.iterations,
    )
    return stats


# ----------------------------------------------------------------
# Step 6: Report
# ----------------------------------------------------------------
def print_report(stats, elapsed: float, args):
    logger.info("[6/7] Phase 1 Results")

    print("\n" + "=" * 70)
    print("  AGISTI Phase 1 -- GPU SURGERY RESULTS")
    print("=" * 70)
    print(f"  Model:                {args.model}")
    print(f"  Total iterations:     {stats.total_iterations}")
    print(f"  Accepted:             {stats.total_accepted}")
    print(f"  Rejected:             {stats.total_rejected}")
    acc_rate = stats.total_accepted / max(stats.total_iterations, 1)
    print(f"  Acceptance rate:      {acc_rate:.1%}")
    print(f"  Emergency rollbacks:  {stats.emergency_rollbacks}")
    print(f"  Wall time:            {elapsed:.1f}s ({elapsed/60:.1f}m)")
    if elapsed > 0 and stats.total_iterations > 0:
        print(f"  Iter/hr:              {stats.total_iterations / (elapsed / 3600):.1f}")
        print(f"  Avg iter time:        {elapsed / stats.total_iterations:.1f}s")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory:      {peak_mem:.1f} GB")
    print("=" * 70)

    # Success criteria for Phase 1
    print("\n  Phase 1 Success Criteria:")
    print(f"  {'PASS' if stats.total_iterations >= 10 else 'FAIL'} At least 10 iterations completed")
    print(f"  {'PASS' if acc_rate > 0 else 'FAIL'} At least 1 surgery accepted")

    has_improvement = False
    try:
        if hasattr(stats, 'score_history') and len(stats.score_history) >= 2:
            has_improvement = stats.score_history[-1] > stats.score_history[0]
    except Exception:
        pass
    print(f"  {'PASS' if has_improvement else 'SKIP'} Score improvement observed")

    # Save results
    results = {
        "phase": "phase_1",
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "iterations": stats.total_iterations,
        "accepted": stats.total_accepted,
        "rejected": stats.total_rejected,
        "acceptance_rate": acc_rate,
        "emergency_rollbacks": stats.emergency_rollbacks,
        "wall_time_seconds": elapsed,
        "lora_rank": args.lora_rank,
        "virtual_train_steps": args.virtual_train_steps,
    }
    if torch.cuda.is_available():
        results["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    results_path = Path(args.output_dir) / "phase1_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to: {results_path}")
    print("=" * 70)


# ----------------------------------------------------------------
# Step 7: Cleanup
# ----------------------------------------------------------------
def cleanup():
    """Free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("[7/7] GPU memory freed")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print_banner()
    args = parse_args()
    t_start = time.time()

    device = setup_gpu()

    try:
        model, tokenizer = load_model(args.model, device)
        probe_bank, bench_suite = load_data(args.probe_bank, args.bench_data)

        if args.skip_frozen:
            logger.info("[3/7] Skipping frozen zone discovery (--skip-frozen)")
            from agisti.frozen.mask import FrozenMask
            frozen_mask = FrozenMask()
        else:
            frozen_mask = run_frozen_discovery(model, tokenizer, probe_bank, device)

        components = build_components(args, model, tokenizer, probe_bank, bench_suite, frozen_mask)
        stats = run_orchestrator(args, model, tokenizer, components, device)
        elapsed = time.time() - t_start
        print_report(stats, elapsed, args)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Phase 1 FAILED: %s", e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
