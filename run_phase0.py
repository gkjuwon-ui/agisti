#!/usr/bin/env python3
"""
AGISTI Phase 0 -- Pipeline Validation
======================================

Real benchmarks: GSM8K, ARC-Challenge, MMLU, HellaSwag, TruthfulQA
Model: Qwen 2.5 0.5B (494M params)
Goal: Validate entire AGISTI pipeline runs end-to-end

Usage:
  python run_phase0.py                    # Default: 10 iterations
  python run_phase0.py --iterations 50    # More iterations
  python run_phase0.py --skip-frozen      # Skip frozen zone discovery
  python run_phase0.py --cpu              # Force CPU mode
"""

from __future__ import annotations

import argparse
import json
import logging
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
logger = logging.getLogger("agisti.phase0")


def print_banner():
    print("""
    ___    ____________ __________
   /   |  / ____/  _/ // /_  __/ /
  / /| | / / __ / / \\_ \\  / / / /
 / ___ |/ /_/ // / ___/ // / / /
/_/  |_|\\____/___//____//_/ /_/

  Phase 0: Pipeline Validation
  Real Benchmarks: GSM8K + ARC + MMLU + HellaSwag + TruthfulQA
""")


def parse_args():
    p = argparse.ArgumentParser(description="AGISTI Phase 0")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=5)
    p.add_argument("--output-dir", default="agisti_output")
    p.add_argument("--skip-frozen", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--probe-bank", default="data/probe_bank.jsonl")
    p.add_argument("--bench-data", default="data/quick_bench.jsonl")
    p.add_argument("--max-gen-tokens", type=int, default=64)
    p.add_argument("--problems-per-iter", type=int, default=5)
    p.add_argument("--bench-problems", type=int, default=10)
    return p.parse_args()


# ----------------------------------------------------------------
# Step 1: Load model
# ----------------------------------------------------------------
def load_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[1/6] Loading model: %s on %s", model_name, device)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=str(device),
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  Model loaded: %.1fM params, dtype=%s (%.1fs)",
        n_params / 1e6, dtype, time.time() - t0,
    )
    return model, tokenizer


# ----------------------------------------------------------------
# Step 2: Load data
# ----------------------------------------------------------------
def load_data(probe_path: str, bench_path: str):
    from agisti.probe.active_prober import ProbeBank
    from agisti.benchmark.quick_bench import QuickBenchSuite

    logger.info("[2/6] Loading benchmark data")

    pp = Path(probe_path)
    bp = Path(bench_path)
    if not pp.exists():
        logger.error("Probe bank not found: %s", pp)
        sys.exit(1)
    if not bp.exists():
        logger.error("Bench data not found: %s", bp)
        sys.exit(1)

    probe_bank = ProbeBank.from_jsonl(pp)
    bench_suite = QuickBenchSuite.from_jsonl("phase0", bp)

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
# Step 3: Frozen zone discovery
# ----------------------------------------------------------------
def run_frozen_discovery(model, tokenizer, probe_bank, device):
    from agisti.frozen.discovery import FrozenZoneDiscovery
    from agisti.frozen.mask import FrozenMask
    from agisti.types import FreezeLevel

    logger.info("[3/6] Running frozen zone discovery...")
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
            probes = probe_bank.get_probes(domain, count=3, seed=42)
            for probe in probes:
                total += 1
                try:
                    prompt = f"Question: {probe.question}\n\nAnswer:"
                    inputs = tokenizer(
                        prompt, return_tensors="pt",
                        truncation=True, max_length=256,
                    )
                    input_ids = inputs["input_ids"].to(device)
                    with torch.no_grad():
                        outputs = m.generate(
                            input_ids=input_ids,
                            max_new_tokens=32,
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
# Step 4: Build components
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

    logger.info("[4/6] Building AGISTI components")

    verifier = AnswerVerifier()

    prober = ActiveProber(
        probe_bank=probe_bank,
        probes_per_domain=5,
        max_gen_tokens=args.max_gen_tokens,
    )

    # Template-based generation (0.5B model can't generate valid JSON)
    generator = ProblemGenerator(
        teacher_model=None,
        teacher_tokenizer=None,
        max_gen_tokens=512,
        temperature=0.7,
    )

    evaluator = ModelEvaluator(
        verifier=verifier,
        max_gen_tokens=args.max_gen_tokens,
        batch_size=1,
    )

    proposer = SurgeryProposer(
        lora_rank=4,
        budget=0.01,
    )

    frozen_names = set(frozen_mask.frozen_layers) if frozen_mask else set()

    virtual_trainer = VirtualTrainer(
        max_steps=3,
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

    logger.info("[5/6] Starting AGISTI orchestrator")

    iter_config = IterationConfig(
        problems_per_iteration=args.problems_per_iter,
        virtual_train_steps=3,
        checkpoint_every=args.epoch_size,
    )
    max_epochs = (args.iterations + args.epoch_size - 1) // args.epoch_size

    phase_config = PhaseConfig(
        phase=PhaseId.PHASE_0,
        model_name=args.model,
        target_iterations=args.iterations,
        epoch_size=args.epoch_size,
        max_epochs=max_epochs,
        iterations_per_epoch=args.epoch_size,
    )
    ckpt_config = CheckpointConfig(keep_last_n=3)
    output_dir = Path(args.output_dir)

    logger.info("  Model:      %s", args.model)
    logger.info("  Iterations: %d (%d epochs x %d)", args.iterations, max_epochs, args.epoch_size)
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
    logger.info("[6/6] Phase 0 Results")
    print("\n" + "=" * 60)
    print("  AGISTI Phase 0 -- RESULTS")
    print("=" * 60)
    print(f"  Total iterations:     {stats.total_iterations}")
    print(f"  Accepted:             {stats.total_accepted}")
    print(f"  Rejected:             {stats.total_rejected}")
    acc_rate = stats.total_accepted / max(stats.total_iterations, 1)
    print(f"  Acceptance rate:      {acc_rate:.1%}")
    print(f"  Emergency rollbacks:  {stats.emergency_rollbacks}")
    print(f"  Wall time:            {elapsed:.1f}s")
    if elapsed > 0:
        print(f"  Iter/hr:              {stats.total_iterations / (elapsed / 3600):.1f}")
    print("=" * 60)

    print("\n  Phase 0 Success Criteria:")
    crashed = stats.emergency_rollbacks > stats.total_iterations * 0.5
    ok_run = not crashed and stats.total_iterations > 0
    print(f"  {'PASS' if ok_run else 'FAIL'} Pipeline ran without fatal crash")
    print(f"  {'PASS' if stats.total_iterations > 0 else 'FAIL'} At least 1 iteration completed")
    print(f"  {'PASS' if stats.total_accepted > 0 else 'SKIP'} At least 1 surgery accepted")

    results = {
        "phase": "phase_0",
        "model": args.model,
        "iterations": stats.total_iterations,
        "accepted": stats.total_accepted,
        "rejected": stats.total_rejected,
        "acceptance_rate": acc_rate,
        "emergency_rollbacks": stats.emergency_rollbacks,
        "wall_time_seconds": elapsed,
    }
    results_path = Path(args.output_dir) / "phase0_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to: {results_path}")
    print("=" * 60)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print_banner()
    args = parse_args()
    t_start = time.time()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    try:
        model, tokenizer = load_model(args.model, device)
        probe_bank, bench_suite = load_data(args.probe_bank, args.bench_data)

        if args.skip_frozen:
            logger.info("[3/6] Skipping frozen zone discovery (--skip-frozen)")
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
        logger.error("Phase 0 FAILED: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
