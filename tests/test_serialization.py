"""
Tests for agisti.utils.serialization — JSON/JSONL I/O, encoder, merge.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from agisti.utils.serialization import (
    AgistiEncoder,
    to_json,
    from_json,
    save_json,
    load_json,
    save_jsonl,
    append_jsonl,
    load_jsonl,
    stream_jsonl,
    count_jsonl,
    dataclass_to_dict,
    deep_merge,
    flatten_dict,
)


class TestAgistiEncoder:
    """Tests for custom JSON encoder."""

    def test_enum_serialization(self):
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        out = json.dumps(Color.RED, cls=AgistiEncoder)
        assert out == '"red"'

    def test_path_serialization(self):
        p = Path("/some/path")
        out = json.dumps(p, cls=AgistiEncoder)
        assert "/some/path" in out

    def test_set_serialization(self):
        s = {3, 1, 2}
        out = json.dumps(s, cls=AgistiEncoder)
        parsed = json.loads(out)
        assert parsed == [1, 2, 3]  # sorted

    def test_bytes_serialization(self):
        b = b"\x00\xff\x10"
        out = json.dumps(b, cls=AgistiEncoder)
        parsed = json.loads(out)
        assert parsed == "00ff10"

    def test_dataclass_serialization(self):
        @dataclass
        class Point:
            x: float
            y: float

        p = Point(x=1.0, y=2.0)
        out = json.dumps(p, cls=AgistiEncoder)
        parsed = json.loads(out)
        assert parsed == {"x": 1.0, "y": 2.0}

    def test_nested_dataclass(self):
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            name: str
            inner: Inner

        obj = Outer(name="test", inner=Inner(value=42))
        out = json.dumps(obj, cls=AgistiEncoder)
        parsed = json.loads(out)
        assert parsed["inner"]["value"] == 42

    def test_tensor_serialization(self):
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        out = json.dumps(t, cls=AgistiEncoder)
        parsed = json.loads(out)
        assert parsed == [1.0, 2.0, 3.0]

    def test_large_tensor_truncation(self):
        import torch
        t = torch.randn(200, 200)
        out = json.dumps(t, cls=AgistiEncoder)
        parsed = json.loads(out)
        assert parsed["_truncated"] is True
        assert parsed["numel"] == 40000


class TestToFromJson:
    """Tests for to_json / from_json."""

    def test_roundtrip(self):
        data = {"key": "value", "num": 42, "nested": {"a": [1, 2]}}
        text = to_json(data)
        loaded = from_json(text)
        assert loaded == data

    def test_compact(self):
        text = to_json({"x": 1}, indent=None)
        assert "\n" not in text


class TestFileIO:
    """Tests for JSON file I/O."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "list": [1, 2, 3]}
            save_json(data, path)
            loaded = load_json(path)
            assert loaded == data

    def test_save_creates_parents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "test.json"
            save_json({"x": 1}, path)
            assert path.exists()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json(Path("/nonexistent/file.json"))

    def test_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"korean": "한국어", "emoji": "🎉"}
            save_json(data, path)
            loaded = load_json(path)
            assert loaded["korean"] == "한국어"


class TestJSONL:
    """Tests for JSONL I/O."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            records = [
                {"id": 1, "val": "a"},
                {"id": 2, "val": "b"},
                {"id": 3, "val": "c"},
            ]
            save_jsonl(records, path)
            loaded = load_jsonl(path)
            assert loaded == records

    def test_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "append.jsonl"
            append_jsonl({"x": 1}, path)
            append_jsonl({"x": 2}, path)
            loaded = load_jsonl(path)
            assert len(loaded) == 2
            assert loaded[0]["x"] == 1
            assert loaded[1]["x"] == 2

    def test_stream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stream.jsonl"
            records = [{"i": i} for i in range(10)]
            save_jsonl(records, path)

            streamed = list(stream_jsonl(path))
            assert len(streamed) == 10
            assert streamed[5]["i"] == 5

    def test_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "count.jsonl"
            save_jsonl([{"x": i} for i in range(7)], path)
            assert count_jsonl(path) == 7

    def test_count_nonexistent(self):
        assert count_jsonl(Path("/does/not/exist.jsonl")) == 0

    def test_handles_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "blanks.jsonl"
            with open(path, "w") as f:
                f.write('{"a": 1}\n\n{"b": 2}\n')
            loaded = load_jsonl(path)
            assert len(loaded) == 2


class TestDataclassToDict:
    """Tests for dataclass serialization."""

    def test_simple(self):
        @dataclass
        class Point:
            x: float
            y: float

        d = dataclass_to_dict(Point(1.0, 2.0))
        assert d == {"x": 1.0, "y": 2.0}

    def test_exclude(self):
        @dataclass
        class Config:
            name: str
            secret: str
            value: int

        d = dataclass_to_dict(
            Config("test", "hidden", 42),
            exclude={"secret"},
        )
        assert "secret" not in d
        assert d["name"] == "test"

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError, match="Expected dataclass"):
            dataclass_to_dict({"not": "a dataclass"})


class TestDeepMerge:
    """Tests for deep_merge."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = deep_merge(base, override)
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 99
        assert result["a"]["z"] == 100
        assert result["b"] == 3

    def test_no_mutation(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert "b" not in base  # base unchanged


class TestFlattenDict:
    """Tests for flatten_dict."""

    def test_flat_input(self):
        d = {"a": 1, "b": 2}
        flat = flatten_dict(d)
        assert flat == {"a": 1, "b": 2}

    def test_nested(self):
        d = {"a": {"b": {"c": 1}}, "d": 2}
        flat = flatten_dict(d)
        assert flat == {"a.b.c": 1, "d": 2}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        flat = flatten_dict(d, sep="/")
        assert flat == {"a/b": 1}

    def test_prefix(self):
        d = {"a": 1}
        flat = flatten_dict(d, prefix="root")
        assert flat == {"root.a": 1}
