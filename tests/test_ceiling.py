"""
Tests for agisti.ceiling — external_signal, retriever, rag_surgery,
inter_model, compositional.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from agisti.types import (
    SelfSignal,
    ExternalSignal,
    RAGSignal,
    CrossSignal,
    BlendedSignal,
    Document,
    LoRADelta,
    LoRALayerDelta,
)
from agisti.ceiling.external_signal import ExternalSurgerySignal
from agisti.ceiling.retriever import DocumentRetriever
from agisti.ceiling.rag_surgery import RetrievalAugmentedSurgery
from agisti.ceiling.inter_model import InterModelSurgery
from agisti.ceiling.compositional import CompositionalDiscovery


# ─── Level 1: ExternalSurgerySignal Tests ────────

class TestExternalSurgerySignal:
    """Tests for Level 1 ceiling breaker — external model signals."""

    def test_creation(self):
        signal_gen = ExternalSurgerySignal()
        assert signal_gen is not None

    def test_generate_signal(self):
        signal_gen = ExternalSurgerySignal()
        # Mock model weights
        target_weights = {
            "layer.0.weight": torch.randn(32, 32),
            "layer.1.weight": torch.randn(32, 32),
        }
        external_weights = {
            "layer.0.weight": torch.randn(32, 32),
            "layer.1.weight": torch.randn(32, 32),
        }
        signal = signal_gen.generate(
            target_weights=target_weights,
            external_weights=external_weights,
        )
        assert isinstance(signal, ExternalSignal)
        assert signal.layer_signals is not None

    def test_signal_contains_all_layers(self):
        signal_gen = ExternalSurgerySignal()
        target = {"layer.0.weight": torch.randn(16, 16)}
        external = {"layer.0.weight": torch.randn(16, 16)}
        signal = signal_gen.generate(target, external)
        assert "layer.0.weight" in signal.layer_signals

    def test_signal_magnitude(self):
        signal_gen = ExternalSurgerySignal()
        # Identical weights → minimal signal
        w = torch.randn(32, 32)
        target = {"layer.0.weight": w.clone()}
        external = {"layer.0.weight": w.clone()}
        signal = signal_gen.generate(target, external)
        for name, mag in signal.layer_signals.items():
            assert abs(mag) < 0.1  # Should be near zero

    def test_missing_layers_handled(self):
        signal_gen = ExternalSurgerySignal()
        target = {"layer.0.weight": torch.randn(16, 16)}
        external = {
            "layer.0.weight": torch.randn(16, 16),
            "extra_layer.weight": torch.randn(16, 16),
        }
        signal = signal_gen.generate(target, external)
        # Should only include matching layers
        assert isinstance(signal, ExternalSignal)


# ─── Level 2: DocumentRetriever Tests ────────────

class TestDocumentRetriever:
    """Tests for FAISS-based document retrieval."""

    def _make_documents(self, n: int = 20) -> list[Document]:
        return [
            Document(
                content=f"Document about topic {i}: " + "word " * 50,
                metadata={"topic": f"topic_{i}", "index": i},
            )
            for i in range(n)
        ]

    def test_creation(self):
        retriever = DocumentRetriever()
        assert retriever is not None

    def test_index_documents(self):
        retriever = DocumentRetriever()
        docs = self._make_documents(10)
        retriever.index(docs)
        assert retriever.num_documents == 10

    def test_search(self):
        retriever = DocumentRetriever()
        docs = self._make_documents(20)
        retriever.index(docs)
        results = retriever.search("topic 5", k=5)
        assert len(results) == 5
        assert all(isinstance(d, Document) for d in results)

    def test_search_empty_index(self):
        retriever = DocumentRetriever()
        results = retriever.search("anything", k=5)
        assert len(results) == 0

    def test_search_k_larger_than_corpus(self):
        retriever = DocumentRetriever()
        docs = self._make_documents(3)
        retriever.index(docs)
        results = retriever.search("topic", k=10)
        assert len(results) <= 3

    def test_add_document(self):
        retriever = DocumentRetriever()
        retriever.index(self._make_documents(5))
        retriever.add(Document(
            content="New document about ML",
            metadata={"topic": "ml"},
        ))
        assert retriever.num_documents == 6

    def test_clear_index(self):
        retriever = DocumentRetriever()
        retriever.index(self._make_documents(10))
        retriever.clear()
        assert retriever.num_documents == 0


class TestRetrievalAugmentedSurgery:
    """Tests for Level 2 RAG-informed surgery."""

    def test_creation(self):
        rag = RetrievalAugmentedSurgery()
        assert rag is not None

    def test_generate_signal(self):
        rag = RetrievalAugmentedSurgery()
        docs = [
            Document(
                content="Research paper about improving attention layers",
                metadata={"source": "arxiv"},
            )
            for _ in range(5)
        ]
        rag.load_documents(docs)
        signal = rag.generate_signal(
            query="How to improve attention mechanism?",
            model_weights={"attn.weight": torch.randn(16, 16)},
        )
        assert isinstance(signal, RAGSignal)

    def test_empty_knowledge_base(self):
        rag = RetrievalAugmentedSurgery()
        signal = rag.generate_signal(
            query="test",
            model_weights={"w": torch.randn(8, 8)},
        )
        assert isinstance(signal, RAGSignal)
        # Should still produce valid signal even without docs


# ─── Level 3: InterModelSurgery Tests ────────────

class TestInterModelSurgery:
    """Tests for cross-model CKA + Procrustes alignment."""

    def test_creation(self):
        ims = InterModelSurgery()
        assert ims is not None

    def test_compute_layer_mapping(self):
        ims = InterModelSurgery()
        # Mock activations from two models
        source_acts = {
            f"layer.{i}": torch.randn(32, 64)
            for i in range(4)
        }
        target_acts = {
            f"layer.{i}": torch.randn(32, 64)
            for i in range(4)
        }
        mapping = ims.compute_layer_mapping(source_acts, target_acts)
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_generate_cross_signal(self):
        ims = InterModelSurgery()
        source_weights = {
            "layer.0.weight": torch.randn(32, 32),
        }
        target_weights = {
            "layer.0.weight": torch.randn(32, 32),
        }
        signal = ims.generate_signal(
            source_weights=source_weights,
            target_weights=target_weights,
            layer_mapping={"layer.0.weight": "layer.0.weight"},
        )
        assert isinstance(signal, CrossSignal)

    def test_cka_similarity_matrix(self):
        ims = InterModelSurgery()
        source_acts = {
            f"layer.{i}": torch.randn(32, 64)
            for i in range(3)
        }
        target_acts = {
            f"layer.{i}": torch.randn(32, 64)
            for i in range(3)
        }
        cka_matrix = ims.compute_cka_matrix(source_acts, target_acts)
        assert cka_matrix.shape == (3, 3)
        # CKA values should be in [0, 1]
        assert cka_matrix.min() >= -0.1  # Allow small numerical errors
        assert cka_matrix.max() <= 1.1

    def test_procrustes_alignment(self):
        ims = InterModelSurgery()
        # Create source and apply known rotation
        source = torch.randn(100, 32)
        rotation = torch.linalg.qr(torch.randn(32, 32))[0]
        target = source @ rotation

        aligned, transform = ims.align_representations(source, target)
        # After alignment, should match target closely
        error = torch.norm(aligned - target) / torch.norm(target)
        assert error < 0.1

    def test_transfer_delta(self):
        ims = InterModelSurgery()
        # Create a simple delta to transfer
        source_delta = LoRADelta(layers={
            "layer.0": LoRALayerDelta(
                A=torch.randn(32, 4),
                B=torch.randn(4, 32),
                rank=4,
                alpha=1.0,
            ),
        })
        mapping = {"layer.0": "layer.0"}
        transferred = ims.transfer_delta(
            source_delta=source_delta,
            layer_mapping=mapping,
        )
        assert isinstance(transferred, LoRADelta)


# ─── Level 4: CompositionalDiscovery Tests ───────

class TestCompositionalDiscovery:
    """Tests for compositional/emergent capability discovery."""

    def test_creation(self):
        cd = CompositionalDiscovery()
        assert cd is not None

    def test_decompose_capability(self):
        cd = CompositionalDiscovery()
        # A complex capability that might decompose into sub-skills
        components = cd.decompose(
            capability="multi-step reasoning",
            model_scores={
                "basic_arithmetic": 0.9,
                "word_problems": 0.7,
                "multi_step": 0.4,
                "symbolic_reasoning": 0.6,
            },
        )
        assert isinstance(components, list)
        assert len(components) > 0

    def test_identify_prerequisites(self):
        cd = CompositionalDiscovery()
        prereqs = cd.identify_prerequisites(
            target_capability="multi_step_reasoning",
            competency_scores={
                "basic_arithmetic": 0.9,
                "single_step": 0.8,
                "chain_of_thought": 0.3,
            },
        )
        assert isinstance(prereqs, list)

    def test_suggest_training_order(self):
        cd = CompositionalDiscovery()
        order = cd.suggest_training_order(
            capabilities=["arithmetic", "algebra", "calculus"],
            current_scores={
                "arithmetic": 0.9,
                "algebra": 0.5,
                "calculus": 0.2,
            },
        )
        assert isinstance(order, list)
        assert len(order) > 0

    def test_emergence_detection(self):
        cd = CompositionalDiscovery()
        # Track scores over iterations
        history = {
            "arithmetic": [0.3, 0.5, 0.7, 0.8, 0.85],
            "algebra": [0.1, 0.1, 0.2, 0.3, 0.7],  # Sudden jump
        }
        emergent = cd.detect_emergence(history)
        assert isinstance(emergent, list)

    def test_capability_graph(self):
        cd = CompositionalDiscovery()
        graph = cd.build_capability_graph(
            capabilities=["add", "multiply", "factorize"],
            dependencies={
                "multiply": ["add"],
                "factorize": ["multiply"],
            },
        )
        assert isinstance(graph, dict)
        assert "factorize" in graph

    def test_compositionality_score(self):
        cd = CompositionalDiscovery()
        score = cd.compositionality_score(
            component_scores={"a": 0.8, "b": 0.7, "c": 0.9},
            composite_score=0.5,
        )
        assert isinstance(score, float)
        # Composite < components → poor compositionality
        assert score < 1.0

    def test_gap_analysis(self):
        cd = CompositionalDiscovery()
        gaps = cd.gap_analysis(
            target_scores={"math": 0.9, "logic": 0.8, "coding": 0.7},
            current_scores={"math": 0.7, "logic": 0.4, "coding": 0.6},
        )
        assert isinstance(gaps, dict)
        assert "logic" in gaps  # Biggest gap
        assert gaps["logic"] > gaps["math"]
