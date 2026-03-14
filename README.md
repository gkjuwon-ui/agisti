<p align="center">
  <h1 align="center">AGISTI</h1>
  <p align="center"><b>Autonomous Generative Intelligence through Self-Taught Iteration</b></p>
  <p align="center"><i>What if an AI could lock itself in a library, study alone for years, and come out smarter — without any teacher?</i></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Goal-Recursive_Self--Improvement-blueviolet" alt="goal">
  <img src="https://img.shields.io/badge/Approach-Teacher--Free_Self--Surgery-blue" alt="approach">
  <img src="https://img.shields.io/badge/Status-PoC_Validated-brightgreen" alt="status">
  <img src="https://img.shields.io/badge/Author-Age_13-red" alt="age">
</p>

---

## The Problem

Current AI training has a ceiling:

- **Supervised learning** — limited by the quality and quantity of human-labeled data
- **RLHF** — limited by the biases and bandwidth of human evaluators
- **Distillation** — a student can never surpass its teacher

Every approach depends on an external signal — a human, a reward model, or a stronger model. **What happens when we remove all of them?**

## The Idea

AGISTI explores a single hypothesis: **a language model can improve itself through recursive self-surgery, with no external teacher.**

The loop:

1. **Probe** — the model measures its own competency across domains
2. **Generate** — it creates problems targeting its weakest areas
3. **Solve & Self-Grade** — it answers its own problems and verifies using structured outputs
4. **Analyze** — it compares internal activations between correct and incorrect answers
5. **Propose Surgery** — it identifies which layers to modify (LoRA-style weight deltas)
6. **Validate** — an independent benchmark (QuickBench) gates every change
7. **Accept or Rollback** — only improvements survive
8. **Repeat** — each iteration starts from the improved model

No teacher. No reward model. No human in the loop. The model is both student and surgeon.

### Why This Could Matter

If a model can reliably improve itself — even by 0.1% per cycle — the implications are significant. Compound a tiny improvement over thousands of iterations and the model diverges from its starting point in ways no human trainer could direct.

This is not AGI. But it may be a **mechanism** by which AGI eventually emerges — an intelligence that bootstraps itself without requiring a smarter intelligence to teach it.

## Architecture

```
+------------------------------------------------------------+
|                 AGISTI Self-Improvement Loop               |
|                                                            |
|  +-----------+    +------------+    +-------------------+  |
|  |  Active   |    |  Problem   |    |  Model Solves     |  |
|  |  Prober   |--->| Generator  |--->|  & Self-Grades    |  |
|  | (5 domains)    | (adaptive) |    | (verifiable ans.) |  |
|  +-----------+    +------------+    +---------+---------+  |
|                                               |            |
|                                    activation differences  |
|                                    (correct vs incorrect)  |
|                                               |            |
|  +-----------+    +------------+    +---------v---------+  |
|  | QuickBench|<---| Delta      |<---|    Surgery        |  |
|  | Validator |    | Applicator |    |    Proposer       |  |
|  |(gatekeeper)    | (LoRA)     |    |  (micro-surgery)  |  |
|  +-----+-----+    +------------+    +-------------------+  |
|        |                                                   |
|   PASS -> accept surgery, next iteration                   |
|   FAIL -> rollback weights, adapt strategy                 |
+------------------------------------------------------------+
```

### 4-Level Ceiling Breaker System

When the basic loop plateaus, AGISTI activates progressively stronger interventions:

| Level | Name | Mechanism |
|-------|------|-----------|
| 1 | **External Signal** | Inject curated problems from GSM8K, ARC, MMLU as additional surgery signal |
| 2 | **RAG Surgery** | Retrieve relevant documents for failed problems → context-aware weight modification |
| 3 | **Cross-Model Pollination** | CKA alignment + Procrustes transform to transfer geometry from a reference model |
| 4 | **Compositional Discovery** | Find problems where individual skills pass but their composition fails |

### Safety Mechanisms

- **Surgery budget (β)**: Caps the magnitude of weight changes per iteration
- **QuickBench gatekeeper**: Every surgery must pass an independent benchmark before acceptance
- **Catastrophe detector**: Triggers emergency rollback if scores drop sharply
- **Frozen zones**: Critical layers (discovered via noise injection) are protected from modification

## Proof of Concept

To validate the framework, we ran the full pipeline on a 72B-parameter model.

The point was not to create a better 72B model — it was to test whether the self-surgery mechanism works at all.

### PoC Setup

| | |
|---|---|
| Model | Qwen 2.5 72B (8-bit quantized, 76 GB across 3 GPUs) |
| Hardware | 3× H100 NVL (288 GB VRAM) |
| Total cost | **$27** |

### PoC Results

| | Result |
|---|---|
| Surgery | **Accepted** — 6 layers modified, delta norm 0.0041 |
| QuickBench | 52.5% post-surgery (passed gatekeeper) |
| Probe improvement | **48.0% → 49.2%** after one iteration |
| Virtual training loss | 4.3198 → 4.3169 (decreased) |

The model generated its own math problems, found 9 it answered incorrectly, proposed weight modifications to 6 layers, and scored higher on the benchmark afterward.

A 1.2% probe improvement in one cycle is small. But it is a **positive signal** — self-improvement occurred without any external teacher. The question is whether this compounds over hundreds or thousands of iterations.


## Project Structure

```
agisti/
├── agisti/                    # Core library
│   ├── benchmark/             # QuickBench + external validators
│   ├── ceiling/               # 4-level ceiling breaker system
│   ├── checkpoint/            # Model state management
│   ├── evaluation/            # Answer verification
│   ├── frozen/                # Frozen zone discovery
│   ├── generation/            # Self-problem generation
│   ├── iteration/             # Single iteration runner
│   ├── orchestrator/          # Multi-iteration orchestration
│   ├── probe/                 # Active probing (competency measurement)
│   ├── surgery/               # Weight modification engine
│   ├── config.py              # Configuration
│   └── types.py               # Core types
├── run_phase0.py              # Probe-only baseline
├── run_phase1.py              # Basic surgery loop
├── run_phase2.py              # Full surgery with ceiling breakers
└── prepare_benchmarks.py      # Benchmark data generation
```

## Quick Start

```bash
pip install -e .
python prepare_benchmarks.py

# Probe baseline (no surgery)
python run_phase0.py --model Qwen/Qwen2.5-7B

# Basic surgery loop
python run_phase1.py --model Qwen/Qwen2.5-7B --iterations 10

# Full surgery with ceiling breakers (requires 250+ GB VRAM)
python run_phase2.py --model Qwen/Qwen2.5-72B --load-in-8bit --iterations 30
```

## Next Steps

- [ ] Long-horizon runs (100+ iterations) to observe compounding improvement curves
- [ ] Activate Level 3 (cross-model pollination) with a reference model
- [ ] Formal evaluation on public benchmarks (GSM8K, MMLU, ARC) pre/post surgery
- [ ] Publish results and methodology as a research paper
- [ ] Scale to 405B+ parameter models

## About the Author

I'm a 13-year-old middle school student from South Korea.

I can't actually code — every line in this repository was written by **Claude via GitHub Copilot** while I directed the architecture, designed the experiments, and ran them. I'm a math person: **KUT Math Competition** Grand Prize ×1, Top Excellence Award ×3.

I built this system, rented 3× H100 NVL GPUs with $30 of pocket money, and ran the experiment in one afternoon.

This project started with a question: *"What if an AI could teach itself without any teacher at all?"*

## Citation

```bibtex
@software{agisti2025,
  title={AGISTI: Autonomous Generative Intelligence through Self-Taught Iteration},
  author={gkjuwon-ui},
  year={2025},
  url={https://github.com/gkjuwon-ui/agisti}
}
```

## License

MIT

---

<p align="center">
  <i>"Can genius emerge without a teacher?"</i><br>
  <i>We're building the mechanism to find out.</i>
</p>
