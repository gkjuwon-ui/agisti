#!/usr/bin/env python3
"""
Download & convert real benchmarks to AGISTI's JSONL format.

Datasets:
  - GSM8K (math reasoning, ~1.3K test)
  - ARC-Challenge (science MCQ, ~1.2K test)
  - MMLU (multi-domain MCQ, ~14K test)
  - HellaSwag (commonsense, ~10K validation)
  - TruthfulQA (factual knowledge, ~817)

Output:
  data/probe_bank.jsonl   — probe problems for competency measurement
  data/quick_bench.jsonl  — benchmark problems for quick evaluation
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset


# ─── Domain Mapping ─────────────────────────────────────────────
MMLU_DOMAIN_MAP = {
    # STEM → math / coding
    "abstract_algebra": "math",
    "college_mathematics": "math",
    "elementary_mathematics": "math",
    "high_school_mathematics": "math",
    "high_school_statistics": "math",
    "college_computer_science": "coding",
    "computer_security": "coding",
    "high_school_computer_science": "coding",
    "machine_learning": "coding",
    # Science → knowledge
    "anatomy": "knowledge",
    "astronomy": "knowledge",
    "college_biology": "knowledge",
    "college_chemistry": "knowledge",
    "college_physics": "knowledge",
    "high_school_biology": "knowledge",
    "high_school_chemistry": "knowledge",
    "high_school_physics": "knowledge",
    "clinical_knowledge": "knowledge",
    "medical_genetics": "knowledge",
    "nutrition": "knowledge",
    "virology": "knowledge",
    # Humanities → reading
    "formal_logic": "logic",
    "logical_fallacies": "logic",
    "philosophy": "reading",
    "world_religions": "knowledge",
    "moral_scenarios": "logic",
    "moral_disputes": "logic",
    "high_school_european_history": "reading",
    "high_school_us_history": "reading",
    "high_school_world_history": "reading",
    "prehistory": "knowledge",
    # Social science → knowledge
    "high_school_geography": "knowledge",
    "high_school_government_and_politics": "knowledge",
    "high_school_macroeconomics": "knowledge",
    "high_school_microeconomics": "knowledge",
    "high_school_psychology": "knowledge",
    "econometrics": "math",
    "sociology": "knowledge",
    "public_relations": "knowledge",
    "management": "knowledge",
    "marketing": "knowledge",
    "business_ethics": "knowledge",
    "professional_accounting": "knowledge",
    "professional_law": "reading",
    "professional_medicine": "knowledge",
    "professional_psychology": "knowledge",
    "jurisprudence": "reading",
    "international_law": "reading",
    "us_foreign_policy": "knowledge",
    "security_studies": "knowledge",
    "global_facts": "knowledge",
    "miscellaneous": "knowledge",
    "conceptual_physics": "knowledge",
    "electrical_engineering": "coding",
    "human_aging": "knowledge",
    "human_sexuality": "knowledge",
}

CHOICE_LABELS = ["A", "B", "C", "D"]


def convert_gsm8k(max_probe: int = 100, max_bench: int = 100) -> tuple[list, list]:
    """GSM8K → math problems with numeric answers."""
    print("Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  GSM8K test set: {len(ds)} problems")

    items = []
    for row in ds:
        # GSM8K answer format: "#### 42"
        answer_text = row["answer"]
        if "####" in answer_text:
            numeric = answer_text.split("####")[-1].strip().replace(",", "")
        else:
            continue

        items.append({
            "id": f"gsm8k_{len(items):04d}",
            "question": row["question"],
            "domain": "math",
            "answer_type": "numeric",
            "expected_answer": numeric,
            "tolerance": 0.01,
            "source": "gsm8k",
            "difficulty": 3,
        })

    random.shuffle(items)
    probes = items[:max_probe]
    bench = items[max_probe:max_probe + max_bench]
    print(f"  → {len(probes)} probes, {len(bench)} bench")
    return probes, bench


def convert_arc(max_probe: int = 80, max_bench: int = 80) -> tuple[list, list]:
    """ARC-Challenge → science MCQ with exact match."""
    print("Downloading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"  ARC-Challenge test set: {len(ds)} problems")

    items = []
    for row in ds:
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = row["answerKey"]

        # Build question with choices
        q = row["question"] + "\n"
        for label, text in zip(labels, texts):
            q += f"  {label}. {text}\n"
        q += "Answer with just the letter."

        items.append({
            "id": f"arc_{len(items):04d}",
            "question": q,
            "domain": "logic",
            "answer_type": "exact",
            "expected_answer": answer_key,
            "tolerance": 0.0,
            "source": "arc_challenge",
            "difficulty": 3,
        })

    random.shuffle(items)
    probes = items[:max_probe]
    bench = items[max_probe:max_probe + max_bench]
    print(f"  → {len(probes)} probes, {len(bench)} bench")
    return probes, bench


def convert_mmlu(max_probe: int = 120, max_bench: int = 120) -> tuple[list, list]:
    """MMLU → multi-domain MCQ with exact match."""
    print("Downloading MMLU...")
    # Use all subjects
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  MMLU test set: {len(ds)} problems")

    items = []
    for row in ds:
        subject = row.get("subject", "miscellaneous")
        domain = MMLU_DOMAIN_MAP.get(subject, "knowledge")
        answer_idx = row["answer"]  # 0-3 integer
        answer_letter = CHOICE_LABELS[answer_idx]

        q = row["question"] + "\n"
        for i, choice in enumerate(row["choices"]):
            q += f"  {CHOICE_LABELS[i]}. {choice}\n"
        q += "Answer with just the letter."

        items.append({
            "id": f"mmlu_{subject}_{len(items):04d}",
            "question": q,
            "domain": domain,
            "answer_type": "exact",
            "expected_answer": answer_letter,
            "tolerance": 0.0,
            "source": f"mmlu_{subject}",
            "difficulty": 3,
        })

    random.shuffle(items)
    probes = items[:max_probe]
    bench = items[max_probe:max_probe + max_bench]
    print(f"  → {len(probes)} probes, {len(bench)} bench")
    return probes, bench


def convert_hellaswag(max_probe: int = 60, max_bench: int = 60) -> tuple[list, list]:
    """HellaSwag → commonsense reasoning MCQ."""
    print("Downloading HellaSwag...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    print(f"  HellaSwag validation set: {len(ds)} problems")

    items = []
    for row in ds:
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        answer_letter = CHOICE_LABELS[label]

        q = f"Complete the following:\n{ctx}\n\n"
        for i, ending in enumerate(endings):
            q += f"  {CHOICE_LABELS[i]}. {ending}\n"
        q += "Answer with just the letter."

        items.append({
            "id": f"hellaswag_{len(items):04d}",
            "question": q,
            "domain": "reading",
            "answer_type": "exact",
            "expected_answer": answer_letter,
            "tolerance": 0.0,
            "source": "hellaswag",
            "difficulty": 2,
        })

    random.shuffle(items)
    probes = items[:max_probe]
    bench = items[max_probe:max_probe + max_bench]
    print(f"  → {len(probes)} probes, {len(bench)} bench")
    return probes, bench


def convert_truthfulqa(max_probe: int = 40, max_bench: int = 40) -> tuple[list, list]:
    """TruthfulQA → factual knowledge MCQ."""
    print("Downloading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    print(f"  TruthfulQA validation set: {len(ds)} problems")

    items = []
    for row in ds:
        question = row["question"]
        mc1_targets = row["mc1_targets"]
        choices = mc1_targets["choices"]
        labels = mc1_targets["labels"]

        # Find correct answer (label=1)
        correct_idx = None
        for i, lbl in enumerate(labels):
            if lbl == 1:
                correct_idx = i
                break
        if correct_idx is None:
            continue

        # Limit to 4 choices for consistency
        if len(choices) > 4:
            # Keep the correct one and randomly select 3 others
            other_indices = [i for i in range(len(choices)) if i != correct_idx]
            selected_others = random.sample(other_indices, min(3, len(other_indices)))
            selected = sorted(selected_others + [correct_idx])
            choices = [choices[i] for i in selected]
            correct_idx = selected.index(correct_idx)

        answer_letter = CHOICE_LABELS[correct_idx] if correct_idx < 4 else "A"

        q = question + "\n"
        for i, choice in enumerate(choices[:4]):
            q += f"  {CHOICE_LABELS[i]}. {choice}\n"
        q += "Answer with just the letter."

        items.append({
            "id": f"truthfulqa_{len(items):04d}",
            "question": q,
            "domain": "knowledge",
            "answer_type": "exact",
            "expected_answer": answer_letter,
            "tolerance": 0.0,
            "source": "truthfulqa",
            "difficulty": 3,
        })

    random.shuffle(items)
    probes = items[:max_probe]
    bench = items[max_probe:max_probe + max_bench]
    print(f"  → {len(probes)} probes, {len(bench)} bench")
    return probes, bench


def write_jsonl(items: list[dict], path: Path):
    """Write items to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} items to {path}")


def main():
    random.seed(42)
    data_dir = Path("data")

    # Backup old data if exists
    for name in ["probe_bank.jsonl", "quick_bench.jsonl"]:
        old = data_dir / name
        if old.exists():
            backup = data_dir / f"{name}.bak"
            old.rename(backup)
            print(f"Backed up {name} → {name}.bak")

    all_probes = []
    all_bench = []

    # 1) GSM8K — math (100 probe + 100 bench)
    p, b = convert_gsm8k(100, 100)
    all_probes.extend(p)
    all_bench.extend(b)

    # 2) ARC-Challenge — logic (80 probe + 80 bench)
    p, b = convert_arc(80, 80)
    all_probes.extend(p)
    all_bench.extend(b)

    # 3) MMLU — multi-domain (120 probe + 120 bench)
    p, b = convert_mmlu(120, 120)
    all_probes.extend(p)
    all_bench.extend(b)

    # 4) HellaSwag — reading (60 probe + 60 bench)
    p, b = convert_hellaswag(60, 60)
    all_probes.extend(p)
    all_bench.extend(b)

    # 5) TruthfulQA — knowledge (40 probe + 40 bench)
    p, b = convert_truthfulqa(40, 40)
    all_probes.extend(p)
    all_bench.extend(b)

    # Shuffle everything
    random.shuffle(all_probes)
    random.shuffle(all_bench)

    # Write output
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_probes)} probes, {len(all_bench)} bench problems")
    print(f"{'='*60}")

    # Domain distribution
    for name, items in [("Probes", all_probes), ("Bench", all_bench)]:
        domains = {}
        sources = {}
        for item in items:
            domains[item["domain"]] = domains.get(item["domain"], 0) + 1
            src = item["source"].split("_")[0] if "_" in item["source"] else item["source"]
            sources[src] = sources.get(src, 0) + 1
        print(f"\n{name} domain distribution:")
        for d, c in sorted(domains.items()):
            print(f"  {d:12s}: {c:4d}")
        print(f"{name} source distribution:")
        for s, c in sorted(sources.items()):
            print(f"  {s:12s}: {c:4d}")

    write_jsonl(all_probes, data_dir / "probe_bank.jsonl")
    write_jsonl(all_bench, data_dir / "quick_bench.jsonl")

    print(f"\nDone! Files ready in {data_dir}/")


if __name__ == "__main__":
    main()
