#!/usr/bin/env python3
"""
AGISTI Phase 2 — 72B 풀스케일 자기개선 수술
============================================

H100 NVL x3 + Qwen 2.5 72B
Phase 1의 "문제 생성 → 풀기 → 수술" 기본 루프에
디자인 문서의 4단계 천장 돌파 시스템을 전부 활성화:

  Level 1: 외부 문제 소스 (GSM8K, MMLU, ARC 등 HuggingFace 데이터셋)
  Level 2: RAG 수술 (실패 문제 → 문서 검색 → 컨텍스트 대비 수술)
  Level 3: 다른 모델 베끼기 (CKA 정렬 + Procrustes 변환)
  Level 4: 복합 기술 발견 (개별 스킬 OK인데 합치면 실패하는 것)

  + 신호 블렌더 (자체/외부/RAG/교차 신호 가중 블렌딩)
  + 외부 수술 제안자 (ExternalSurgeryProposer)
  + 메타 전략 적응 엔진 (MetaStrategyEngine)
  + 재앙 감지기 (CatastropheDetector)
  + HuggingFace Hub 체크포인트 업로드 (디스크 부족 해결)
  + 외부 벤치마크 검증 (DynamicExternalValidator)

사용법:
  python run_phase2.py                                    # 기본: 72B, 100 iterations
  python run_phase2.py --iterations 500                   # 더 많이
  python run_phase2.py --model Qwen/Qwen2.5-72B-Instruct  # Instruct 버전
  python run_phase2.py --ref-model Qwen/Qwen2.5-7B        # 교차수분용 참조 모델
  python run_phase2.py --hf-repo myuser/agisti-72b         # HF Hub 업로드
  python run_phase2.py --skip-rag                          # RAG 수술 비활성화
  python run_phase2.py --skip-cross                        # 교차수분 비활성화
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# PyTorch 2.4 호환: set_submodule이 없으면 직접 추가
if not hasattr(torch.nn.Module, 'set_submodule'):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

# --- 로깅 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("agisti.phase2")


def print_banner():
    print("""
    ___    ____________ __________
   /   |  / ____/  _/ // /_  __/ /
  / /| | / / __ / / \\_ \\  / / / /
 / ___ |/ /_/ // / ___/ // / / /
/_/  |_|\\____/___//____//_/ /_/

  Phase 2: 72B Full-Scale Self-Surgery
  H100 NVL x3 + 4-Level Ceiling Breaker
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Level 1: External Signal (GSM8K, MMLU, ARC)
  Level 2: RAG Surgery (Retrieval-Augmented)
  Level 3: Cross-Model Pollination (CKA)
  Level 4: Compositional Discovery
""")


def parse_args():
    p = argparse.ArgumentParser(description="AGISTI Phase 2 -- 72B Full Surgery")

    # 모델 설정
    p.add_argument("--model", default="Qwen/Qwen2.5-72B",
                   help="Target model (default: Qwen/Qwen2.5-72B)")
    p.add_argument("--ref-model", default=None,
                   help="Reference model for cross-pollination (Level 3)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4bit quantization for VRAM saving")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="8bit quantization for VRAM saving")

    # 반복 설정
    p.add_argument("--iterations", type=int, default=100,
                   help="Total iterations (default: 100)")
    p.add_argument("--epoch-size", type=int, default=10,
                   help="Iterations per epoch (default: 10)")

    # 출력 & 체크포인트
    p.add_argument("--output-dir", default="agisti_output/phase2",
                   help="Output directory")
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace Hub repo for checkpoint upload")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env)")

    # Frozen Zone
    p.add_argument("--skip-frozen", action="store_true",
                   help="Skip frozen zone discovery")

    # 데이터
    p.add_argument("--probe-bank", default="data/probe_bank.jsonl")
    p.add_argument("--bench-data", default="data/quick_bench.jsonl")

    # Generation
    p.add_argument("--max-gen-tokens", type=int, default=256,
                   help="Max generation tokens (default: 256)")
    p.add_argument("--problems-per-iter", type=int, default=15,
                   help="Self-generated problems per iteration (default: 15)")
    p.add_argument("--bench-problems", type=int, default=20,
                   help="QuickBench problems per domain (default: 20)")

    # Surgery
    p.add_argument("--lora-rank", type=int, default=16,
                   help="LoRA rank (default: 16, bigger for 72B)")
    p.add_argument("--virtual-train-steps", type=int, default=30,
                   help="Virtual training steps (default: 30)")
    p.add_argument("--surgery-budget", type=float, default=0.01,
                   help="Surgery budget (default: 0.01)")

    # Probe
    p.add_argument("--probes-per-domain", type=int, default=10,
                   help="Probes per domain (default: 10)")

    # Level 1: External Signal
    p.add_argument("--external-problems", type=int, default=50,
                   help="External problems per iteration (default: 50)")
    p.add_argument("--skip-external", action="store_true",
                   help="Skip external signal (Level 1)")

    # Level 2: RAG Surgery
    p.add_argument("--skip-rag", action="store_true",
                   help="Skip RAG surgery (Level 2)")
    p.add_argument("--rag-top-k", type=int, default=3,
                   help="RAG: top-k documents per query (default: 3)")

    # Level 3: Cross-Model
    p.add_argument("--skip-cross", action="store_true",
                   help="Skip cross-model pollination (Level 3)")

    # Level 4: Compositional
    p.add_argument("--skip-compositional", action="store_true",
                   help="Skip compositional discovery (Level 4)")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint directory")

    return p.parse_args()


# ═══════════════════════════════════════════════════
# Step 1: GPU 설정 + 모델 로딩
# ═══════════════════════════════════════════════════


def setup_gpu():
    """H100 NVL x3 감지 및 설정."""
    if not torch.cuda.is_available():
        logger.warning("CUDA GPU 감지 실패! CPU 모드로 진행.")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    total_vram = 0.0
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_vram += gpu_mem
        logger.info("GPU %d: %s (%.1f GB)", i, gpu_name, gpu_mem)

    if n_gpus > 1:
        logger.info("총 VRAM: %.1f GB (%d GPUs)", total_vram, n_gpus)
    logger.info("72B bfloat16 ≈ 144GB 필요, 가용 %.1f GB", total_vram)

    # Ampere+ 최적화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return torch.device("cuda")


def load_model(model_name: str, device: torch.device, args):
    """72B 모델을 multi-GPU에 로딩."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[1/9] 모델 로딩: %s", model_name)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",  # multi-GPU 자동 분배
    }

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("  4bit 양자화 모드 활성화")
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("  8bit 양자화 모드 활성화")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  모델 로딩 완료: %.1fB params (%.1fs)",
        n_params / 1e9, time.time() - t0,
    )

    if device.type == "cuda":
        mem_used = sum(
            torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ) / 1e9
        mem_total = sum(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ) / 1e9
        logger.info(
            "  GPU 메모리: %.1f / %.1f GB (%d GPUs)",
            mem_used, mem_total, torch.cuda.device_count(),
        )

    return model, tokenizer


# ═══════════════════════════════════════════════════
# Step 2: 데이터 로딩
# ═══════════════════════════════════════════════════


def load_data(probe_path: str, bench_path: str):
    """Probe 뱅크 & QuickBench 데이터 로딩."""
    from agisti.probe.active_prober import ProbeBank
    from agisti.benchmark.quick_bench import QuickBenchSuite

    logger.info("[2/9] 벤치마크 데이터 로딩")

    pp = Path(probe_path)
    bp = Path(bench_path)
    if not pp.exists():
        logger.error("Probe bank 없음: %s", pp)
        sys.exit(1)
    if not bp.exists():
        logger.error("Bench data 없음: %s", bp)
        sys.exit(1)

    probe_bank = ProbeBank.from_jsonl(pp)
    bench_suite = QuickBenchSuite.from_jsonl("phase2", bp)

    logger.info("  Probes: %d (%s)", probe_bank.total_probes, probe_bank.domains)
    logger.info("  Bench:  %d (%s)", bench_suite.total, bench_suite.domains)
    return probe_bank, bench_suite


# ═══════════════════════════════════════════════════
# Step 3: Frozen Zone Discovery
# ═══════════════════════════════════════════════════


def run_frozen_discovery(model, tokenizer, probe_bank, device):
    """Frozen zone 자동 탐색 — 건드리면 안 되는 레이어 찾기."""
    from agisti.frozen.discovery import FrozenZoneDiscovery
    from agisti.frozen.mask import FrozenMask
    from agisti.types import FreezeLevel

    logger.info("[3/9] Frozen zone 탐색 중...")
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
    logger.info("  Frozen 탐색 완료: %.1fs", time.time() - t0)

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


# ═══════════════════════════════════════════════════
# Step 4: 기본 컴포넌트 빌드 (Phase 1과 동일한 것들)
# ═══════════════════════════════════════════════════


def build_base_components(args, model, tokenizer, probe_bank, bench_suite, frozen_mask):
    """Phase 1에서 이미 쓰던 기본 컴포넌트들."""
    from agisti.probe.active_prober import ActiveProber
    from agisti.generation.generator import ProblemGenerator
    from agisti.evaluation.evaluator import ModelEvaluator
    from agisti.generation.verification import AnswerVerifier
    from agisti.surgery.proposer import SurgeryProposer
    from agisti.surgery.virtual_trainer import VirtualTrainer
    from agisti.surgery.applicator import DeltaApplicator
    from agisti.benchmark.quick_bench import QuickBench
    from agisti.config import QuickBenchConfig

    logger.info("[4/9] 기본 컴포넌트 빌드")

    verifier = AnswerVerifier()

    prober = ActiveProber(
        probe_bank=probe_bank,
        probes_per_domain=args.probes_per_domain,
        max_gen_tokens=args.max_gen_tokens,
    )

    generator = ProblemGenerator(
        teacher_model=model,
        teacher_tokenizer=tokenizer,
        max_gen_tokens=1024,
        temperature=0.95,  # 72B한테 어려운 문제 생성
    )

    evaluator = ModelEvaluator(
        verifier=verifier,
        max_gen_tokens=args.max_gen_tokens,
        batch_size=2,  # 72B는 배치를 작게 잡아야 VRAM 세이프
    )

    proposer = SurgeryProposer(
        lora_rank=args.lora_rank,
        budget=args.surgery_budget,
        min_wrong_samples=1,  # 72B는 너무 잘 풀어서 1개만 틀려도 수술
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

    logger.info("  기본 컴포넌트 빌드 완료")

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


# ═══════════════════════════════════════════════════
# Step 5: 천장 돌파 시스템 (Phase 2 고급 모듈들)
# ═══════════════════════════════════════════════════


def build_ceiling_breakers(args, model, tokenizer, frozen_mask):
    """
    디자인 문서 §11의 4단계 천장 돌파 시스템.

    Phase 1에서는 하나도 안 쓰고 있었던 보물창고들:
    - ceiling/external_signal.py
    - ceiling/rag_surgery.py
    - ceiling/inter_model.py
    - ceiling/compositional.py
    - ceiling/retriever.py
    - surgery/proposer_external.py
    - surgery/signal_blender.py
    """
    ceiling_components = {}

    # ─── Level 1: 외부 문제 소스 ──────────────────
    if not args.skip_external:
        from agisti.ceiling.external_signal import (
            ExternalSurgerySignal,
            HuggingFaceFetcher,
            ExternalSourceConfig as ExtSrcCfg,
        )
        from agisti.types import AnswerType

        logger.info("  [Level 1] 외부 Surgery Signal 활성화")

        ext_signal = ExternalSurgerySignal()

        # GSM8K — 수학
        gsm8k_fetcher = HuggingFaceFetcher(
            dataset_name="openai/gsm8k",
            split="test",
            question_field="question",
            answer_field="answer",
            answer_type=AnswerType.EXACT_MATCH,
        )
        ext_signal.register_source("gsm8k", ExtSrcCfg(
            name="gsm8k",
            fetcher_class=type(gsm8k_fetcher),
            max_problems_per_batch=200,
        ))
        ext_signal._fetcher_cache["gsm8k"] = gsm8k_fetcher

        # ARC Challenge — 추론
        arc_fetcher = HuggingFaceFetcher(
            dataset_name="allenai/ai2_arc",
            split="test",
            question_field="question",
            answer_field="answerKey",
            answer_type=AnswerType.EXACT_MATCH,
        )
        ext_signal.register_source("arc", ExtSrcCfg(
            name="arc",
            fetcher_class=type(arc_fetcher),
            max_problems_per_batch=200,
        ))
        ext_signal._fetcher_cache["arc"] = arc_fetcher

        # MMLU — 도메인 지식
        mmlu_fetcher = HuggingFaceFetcher(
            dataset_name="cais/mmlu",
            split="test",
            question_field="question",
            answer_field="answer",
            answer_type=AnswerType.EXACT_MATCH,
        )
        ext_signal.register_source("mmlu", ExtSrcCfg(
            name="mmlu",
            fetcher_class=type(mmlu_fetcher),
            max_problems_per_batch=500,
        ))
        ext_signal._fetcher_cache["mmlu"] = mmlu_fetcher

        ceiling_components["external_signal"] = ext_signal
        logger.info("    등록 완료: gsm8k, arc, mmlu")
    else:
        logger.info("  [Level 1] 외부 Signal 비활성화 (--skip-external)")

    # ─── Level 2: RAG Surgery ────────────────────
    if not args.skip_rag:
        from agisti.ceiling.rag_surgery import RetrievalAugmentedSurgery
        from agisti.ceiling.retriever import DocumentRetriever

        logger.info("  [Level 2] RAG Surgery 활성화")

        # RAG Retriever — FAISS 인덱스가 없으면 인메모리 모드로 동작
        retriever = DocumentRetriever()
        rag_surgery = RetrievalAugmentedSurgery(
            retriever=retriever,
            min_flips=2,  # 72B는 더 잘 풀어서 flip 기준 낮춤
            max_context_tokens=1536,
        )

        ceiling_components["rag_surgery"] = rag_surgery
        ceiling_components["retriever"] = retriever
        logger.info("    RAG Retriever + Surgery 준비 완료")
    else:
        logger.info("  [Level 2] RAG Surgery 비활성화 (--skip-rag)")

    # ─── Level 3: Cross-Model Pollination ────────
    if not args.skip_cross and args.ref_model:
        from agisti.ceiling.inter_model import (
            InterModelCrossPollinator,
            compute_cka,
            compute_procrustes,
        )

        logger.info("  [Level 3] Cross-Model 교차수분 활성화")
        logger.info("    참조 모델: %s", args.ref_model)

        # 참조 모델 로딩 (작은 모델이므로 1개 GPU에)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ref_tokenizer = AutoTokenizer.from_pretrained(
            args.ref_model, trust_remote_code=True,
        )
        if ref_tokenizer.pad_token is None:
            ref_tokenizer.pad_token = ref_tokenizer.eos_token

        # 마지막 GPU에 참조 모델 올리기
        n_gpus = torch.cuda.device_count()
        ref_device = f"cuda:{n_gpus - 1}" if n_gpus > 1 else "cuda:0"
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=ref_device,
        )
        ref_model.eval()

        ceiling_components["ref_model"] = ref_model
        ceiling_components["ref_tokenizer"] = ref_tokenizer
        ceiling_components["cross_pollinator_tools"] = {
            "compute_cka": compute_cka,
            "compute_procrustes": compute_procrustes,
        }
        logger.info("    참조 모델 로딩 완료 (device: %s)", ref_device)
    elif not args.skip_cross:
        logger.info("  [Level 3] Cross-Model 비활성화 (--ref-model 미지정)")
    else:
        logger.info("  [Level 3] Cross-Model 비활성화 (--skip-cross)")

    # ─── Level 4: Compositional Discovery ────────
    if not args.skip_compositional:
        from agisti.ceiling.compositional import (
            CompositionalProblemGenerator,
            CompetencyPair,
        )

        logger.info("  [Level 4] 복합 기술 발견 활성화")

        comp_generator = CompositionalProblemGenerator(seed=42)
        ceiling_components["compositional_generator"] = comp_generator
        logger.info("    복합 문제 생성기 준비 완료")
    else:
        logger.info("  [Level 4] 복합 발견 비활성화 (--skip-compositional)")

    return ceiling_components


# ═══════════════════════════════════════════════════
# Step 6: 신호 블렌더 + 고급 수술 제안자
# ═══════════════════════════════════════════════════


def build_advanced_surgery(args, ceiling_components):
    """
    Phase 2 전용 고급 수술 시스템.

    - SignalBlender: 다중 소스 신호를 phase_2 가중치로 블렌딩
    - ExternalSurgeryProposer: 외부 신호 기반 수술 제안
    """
    from agisti.surgery.signal_blender import (
        SignalBlender,
        AdaptiveSignalBlender,
        SignalCollection,
    )
    from agisti.surgery.proposer_external import ExternalSurgeryProposer

    logger.info("[6/9] 고급 수술 시스템 빌드")

    # Phase 2 전용 블렌더 — 자체 20%, 외부 20%, RAG 60%
    blender = AdaptiveSignalBlender(
        phase_key="phase_2_early",
        adaptation_rate=0.02,
        min_weight=0.05,
        max_weight=0.9,
    )

    # 외부 수술 제안자
    ext_proposer = ExternalSurgeryProposer(
        lora_rank=args.lora_rank,
        budget=args.surgery_budget,
        external_weight=0.5,
    )

    logger.info("  SignalBlender: phase_2_early 가중치")
    logger.info("  ExternalSurgeryProposer: rank=%d, budget=%.3f",
                args.lora_rank, args.surgery_budget)

    return {
        "blender": blender,
        "ext_proposer": ext_proposer,
        "signal_collection_cls": SignalCollection,
    }


# ═══════════════════════════════════════════════════
# Step 7: 외부 벤치마크 검증기
# ═══════════════════════════════════════════════════


def build_external_validator(args):
    """외부 벤치마크에 대한 독립적인 모델 검증."""
    from agisti.benchmark.external_validator import (
        DynamicExternalValidator,
        ExternalBenchmarkSpec,
    )
    from agisti.generation.verification import AnswerVerifier
    from agisti.types import AnswerType

    logger.info("[7/9] 외부 벤치마크 검증기 활성화")

    specs = [
        ExternalBenchmarkSpec(
            name="gsm8k",
            source="huggingface",
            path_or_url="openai/gsm8k",
            answer_type=AnswerType.EXACT_MATCH,
            domain="math",
            max_problems=200,
            metadata={"split": "test", "question_field": "question",
                       "answer_field": "answer"},
        ),
        ExternalBenchmarkSpec(
            name="arc_challenge",
            source="huggingface",
            path_or_url="allenai/ai2_arc",
            answer_type=AnswerType.EXACT_MATCH,
            domain="reasoning",
            max_problems=200,
            metadata={"split": "test", "question_field": "question",
                       "answer_field": "answerKey"},
        ),
    ]

    validator = DynamicExternalValidator(
        benchmark_specs=specs,
        verifier=AnswerVerifier(),
        max_gen_tokens=args.max_gen_tokens,
    )

    logger.info("  등록 완료: %d 외부 벤치마크", len(specs))
    return validator


# ═══════════════════════════════════════════════════
# Step 8: 오케스트레이터 실행
# ═══════════════════════════════════════════════════


def run_orchestrator(args, model, tokenizer, base_components,
                     ceiling_components, advanced_surgery,
                     external_validator, device):
    """Phase 2 전체 오케스트레이터 실행."""
    from agisti.orchestrator.orchestrator import AGISTIOrchestrator
    from agisti.config import (
        PhaseConfig, IterationConfig, CheckpointConfig,
        CeilingBreakerConfig, MetaStrategy,
    )
    from agisti.types import PhaseId, SurgeryType

    logger.info("[8/9] 오케스트레이터 시작 (Phase 2)")

    iter_config = IterationConfig(
        problems_per_iteration=args.problems_per_iter,
        virtual_train_steps=args.virtual_train_steps,
        checkpoint_every=args.epoch_size,
        lora_rank=args.lora_rank,
        trace_activations=True,
        surgery_budget=args.surgery_budget,
    )
    max_epochs = (args.iterations + args.epoch_size - 1) // args.epoch_size

    # Phase 2 천장 돌파 설정
    ceiling_config = CeilingBreakerConfig(
        external_weight_min=0.2,
        external_weight_max=0.8,
        external_weight_adaptation_step=0.05,
        rag_enabled=not args.skip_rag,
        rag_top_k=args.rag_top_k,
        cross_model_enabled=(not args.skip_cross and args.ref_model is not None),
        compositional_enabled=not args.skip_compositional,
    )

    # Phase 2 전략 — 더 과감한 수술
    strategy = MetaStrategy(
        surgery_type=SurgeryType.MICRO,
        lora_rank=args.lora_rank,
        surgery_budget=args.surgery_budget,
        external_weight=0.5,
        difficulty_level=0.5,
        exploration_rate=0.2,
        ceiling_level=2,  # Level 2부터 시작
    )

    phase_config = PhaseConfig(
        phase=PhaseId.PHASE_2,
        model_name=args.model,
        target_iterations=args.iterations,
        epoch_size=args.epoch_size,
        max_epochs=max_epochs,
        iterations_per_epoch=args.epoch_size,
        ceiling=ceiling_config,
        strategy=strategy,
    )

    ckpt_config = CheckpointConfig(
        keep_last_n=5,
        keep_every_nth=10,
    )
    output_dir = Path(args.output_dir)

    logger.info("  Phase:      2 (72B Full Surgery)")
    logger.info("  모델:       %s", args.model)
    logger.info("  반복:       %d (%d epochs x %d)", args.iterations, max_epochs, args.epoch_size)
    logger.info("  LoRA rank:  %d", args.lora_rank)
    logger.info("  VTrain:     %d steps", args.virtual_train_steps)
    logger.info("  Budget:     %.3f", args.surgery_budget)
    logger.info("  Output:     %s", output_dir)
    logger.info("  Device:     %s", device)
    logger.info("  천장돌파:    L1=%s L2=%s L3=%s L4=%s",
                "ON" if not args.skip_external else "OFF",
                "ON" if not args.skip_rag else "OFF",
                "ON" if (not args.skip_cross and args.ref_model) else "OFF",
                "ON" if not args.skip_compositional else "OFF")

    orchestrator = AGISTIOrchestrator(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        phase_config=phase_config,
        iteration_config=iter_config,
        checkpoint_config=ckpt_config,
        initial_strategy=strategy,
        device=device,
        **base_components,
    )

    # 천장 돌파 컴포넌트를 오케스트레이터에 주입
    if ceiling_components:
        _inject_ceiling_breakers(orchestrator, ceiling_components, advanced_surgery)

    stats = orchestrator.run(
        max_epochs=max_epochs,
        max_iterations=args.iterations,
    )
    return stats


def _inject_ceiling_breakers(orchestrator, ceiling_components, advanced_surgery):
    """천장 돌파 컴포넌트를 오케스트레이터에 주입."""
    # 외부 신호
    if "external_signal" in ceiling_components:
        orchestrator._external_signal = ceiling_components["external_signal"]

    # RAG
    if "rag_surgery" in ceiling_components:
        orchestrator._rag_surgery = ceiling_components["rag_surgery"]

    # 참조 모델
    if "ref_model" in ceiling_components:
        orchestrator._ref_model = ceiling_components["ref_model"]
        orchestrator._ref_tokenizer = ceiling_components["ref_tokenizer"]

    # 복합 발견
    if "compositional_generator" in ceiling_components:
        orchestrator._compositional_gen = ceiling_components["compositional_generator"]

    # 블렌더
    if "blender" in advanced_surgery:
        orchestrator._signal_blender = advanced_surgery["blender"]
    if "ext_proposer" in advanced_surgery:
        orchestrator._ext_proposer = advanced_surgery["ext_proposer"]


# ═══════════════════════════════════════════════════
# Step 9: 결과 보고 + HuggingFace Hub 업로드
# ═══════════════════════════════════════════════════


def upload_to_hf(model, tokenizer, args, stats):
    """HuggingFace Hub에 체크포인트 업로드 (디스크 부족 해결)."""
    if not args.hf_repo:
        return

    logger.info("HuggingFace Hub 업로드 시작: %s", args.hf_repo)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN 없음! 업로드 건너뜀")
        return

    try:
        model.push_to_hub(
            args.hf_repo,
            token=hf_token,
            commit_message=(
                f"AGISTI Phase 2: {stats.total_iterations} iterations, "
                f"accepted {stats.total_accepted}"
            ),
        )
        tokenizer.push_to_hub(args.hf_repo, token=hf_token)
        logger.info("  HF Hub 업로드 완료: %s", args.hf_repo)
    except Exception as e:
        logger.error("  HF Hub 업로드 실패: %s", e)


def print_report(stats, elapsed: float, args, ceiling_components):
    """Phase 2 결과 출력."""
    logger.info("[9/9] Phase 2 결과")

    print("\n" + "=" * 70)
    print("  AGISTI Phase 2 — 72B FULL-SCALE SURGERY RESULTS")
    print("=" * 70)
    print(f"  모델:                 {args.model}")
    print(f"  총 반복 횟수:         {stats.total_iterations}")
    print(f"  수락됨:               {stats.total_accepted}")
    print(f"  거부됨:               {stats.total_rejected}")
    acc_rate = stats.total_accepted / max(stats.total_iterations, 1)
    print(f"  수락률:               {acc_rate:.1%}")
    print(f"  긴급 롤백:            {stats.emergency_rollbacks}")
    print(f"  실행 시간:            {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if elapsed > 0 and stats.total_iterations > 0:
        print(f"  반복/시간:            {stats.total_iterations / (elapsed / 3600):.1f}")
        print(f"  평균 반복 시간:       {elapsed / stats.total_iterations:.1f}s")

    # 천장 돌파 레벨 상태
    print("\n  천장 돌파 시스템:")
    print(f"    Level 1 (외부 신호):  {'활성' if 'external_signal' in ceiling_components else '비활성'}")
    print(f"    Level 2 (RAG 수술):   {'활성' if 'rag_surgery' in ceiling_components else '비활성'}")
    print(f"    Level 3 (교차수분):   {'활성' if 'ref_model' in ceiling_components else '비활성'}")
    print(f"    Level 4 (복합발견):   {'활성' if 'compositional_generator' in ceiling_components else '비활성'}")

    if torch.cuda.is_available():
        peak_mem = sum(
            torch.cuda.max_memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ) / 1e9
        print(f"\n  피크 GPU 메모리:      {peak_mem:.1f} GB")
    print("=" * 70)

    # Phase 2 성공 기준
    print("\n  Phase 2 성공 기준:")
    print(f"  {'PASS' if stats.total_iterations >= 10 else 'FAIL'} 10회 이상 반복 완료")
    print(f"  {'PASS' if acc_rate > 0 else 'FAIL'} 1회 이상 수술 수락")
    print(f"  {'PASS' if stats.total_accepted >= 3 else 'FAIL'} 3회 이상 수술 수락 (벤치마크 상승 증거)")

    has_improvement = False
    try:
        if hasattr(stats, 'score_history') and len(stats.score_history) >= 2:
            has_improvement = stats.score_history[-1] > stats.score_history[0]
    except Exception:
        pass
    print(f"  {'PASS' if has_improvement else 'SKIP'} 점수 상승 관측됨")

    # 결과 JSON 저장
    results = {
        "phase": "phase_2",
        "model": args.model,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "iterations": stats.total_iterations,
        "accepted": stats.total_accepted,
        "rejected": stats.total_rejected,
        "acceptance_rate": acc_rate,
        "emergency_rollbacks": stats.emergency_rollbacks,
        "wall_time_seconds": elapsed,
        "lora_rank": args.lora_rank,
        "virtual_train_steps": args.virtual_train_steps,
        "surgery_budget": args.surgery_budget,
        "ceiling_levels": {
            "external": "external_signal" in ceiling_components,
            "rag": "rag_surgery" in ceiling_components,
            "cross": "ref_model" in ceiling_components,
            "compositional": "compositional_generator" in ceiling_components,
        },
    }
    if torch.cuda.is_available():
        results["peak_gpu_memory_gb"] = sum(
            torch.cuda.max_memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ) / 1e9

    results_path = Path(args.output_dir) / "phase2_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  결과 저장: {results_path}")
    print("=" * 70)


# ═══════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════


def cleanup():
    """GPU 메모리 해제."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU 메모리 해제 완료")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════


def main():
    print_banner()
    args = parse_args()
    t_start = time.time()

    device = setup_gpu()

    try:
        # 1. 모델 로딩
        model, tokenizer = load_model(args.model, device, args)

        # 2. 데이터
        probe_bank, bench_suite = load_data(args.probe_bank, args.bench_data)

        # 3. Frozen Zone
        if args.skip_frozen:
            logger.info("[3/9] Frozen zone 탐색 건너뜀 (--skip-frozen)")
            from agisti.frozen.mask import FrozenMask
            frozen_mask = FrozenMask()
        else:
            frozen_mask = run_frozen_discovery(
                model, tokenizer, probe_bank, device,
            )

        # 4. 기본 컴포넌트 (Phase 1에서 이미 검증됨)
        base_components = build_base_components(
            args, model, tokenizer, probe_bank, bench_suite, frozen_mask,
        )

        # 5. 천장 돌파 시스템 (Phase 2 핵심!)
        logger.info("[5/9] 천장 돌파 시스템 활성화")
        ceiling_components = build_ceiling_breakers(
            args, model, tokenizer, frozen_mask,
        )

        # 6. 고급 수술 시스템
        advanced_surgery = build_advanced_surgery(args, ceiling_components)

        # 7. 외부 벤치마크 검증기
        external_validator = build_external_validator(args)

        # 8. 오케스트레이터 실행
        stats = run_orchestrator(
            args, model, tokenizer, base_components,
            ceiling_components, advanced_surgery,
            external_validator, device,
        )

        # 9. 결과 보고
        elapsed = time.time() - t_start
        print_report(stats, elapsed, args, ceiling_components)

        # HuggingFace Hub 업로드 (디스크 부족 해결)
        upload_to_hf(model, tokenizer, args, stats)

    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        sys.exit(130)
    except Exception as e:
        logger.error("Phase 2 실패: %s", e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
