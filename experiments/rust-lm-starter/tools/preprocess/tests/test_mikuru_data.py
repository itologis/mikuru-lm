#!/usr/bin/env python3
"""Unit tests for the Mikuru JSONL audit helpers (stdlib unittest)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mikuru_data import (  # noqa: E402
    JsonlError,
    conversation_chars,
    find_duplicate_ids,
    fnv1a_64,
    iter_jsonl,
    role_counts,
    summarise,
    validate_record,
)


def record(**overrides):
    base = {
        "id": "rec-1",
        "messages": [
            {"role": "user", "content": "Hello, Mikuru!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "language": "en",
        "category": "general_conversation",
        "source_dataset": "Mikuru Test",
        "source_record_id": "1",
        "source_url": "N/A",
        "license": "Project Internal",
        "synthetic": True,
        "provenance_note": "Unit-test record.",
    }
    base.update(overrides)
    return base


def fields(problems):
    return [field for field, _message in problems]


class ValidationTests(unittest.TestCase):
    def test_valid_record_has_no_problems(self):
        self.assertEqual(validate_record(record()), [])

    def test_blank_license_is_rejected(self):
        problems = validate_record(record(license="   "))
        self.assertIn("license", fields(problems))

    def test_missing_required_field_is_rejected(self):
        value = record()
        del value["provenance_note"]
        self.assertIn("provenance_note", fields(validate_record(value)))

    def test_empty_messages_are_rejected(self):
        self.assertIn("messages", fields(validate_record(record(messages=[]))))

    def test_blank_content_is_rejected(self):
        value = record(messages=[{"role": "user", "content": " "}])
        self.assertIn("messages", fields(validate_record(value)))

    def test_unknown_split_is_rejected(self):
        self.assertIn("split", fields(validate_record(record(split="test"))))

    def test_demo_split_is_accepted(self):
        self.assertEqual(validate_record(record(split="demo")), [])

    def test_synthetic_must_be_boolean(self):
        self.assertIn("synthetic", fields(validate_record(record(synthetic=1))))

    def test_blank_reference_is_rejected(self):
        self.assertIn("references", fields(validate_record(record(references=[" "]))))

    def test_duplicate_ids_are_reported_with_first_location(self):
        located = [
            ("a.jsonl", 1, {"id": "x"}),
            ("b.jsonl", 2, {"id": "x"}),
        ]
        duplicates = find_duplicate_ids(located)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0][0:3], ("b.jsonl", 2, "x"))
        self.assertEqual(duplicates[0][3], "a.jsonl:1")


class JsonlTests(unittest.TestCase):
    def test_blank_line_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(record()) + "\n\n", encoding="utf-8")
            with self.assertRaises(JsonlError):
                list(iter_jsonl(path))

    def test_non_object_record_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")
            with self.assertRaises(JsonlError):
                list(iter_jsonl(path))

    def test_valid_stream_yields_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                json.dumps(record()) + "\n" + json.dumps(record(id="rec-2")) + "\n",
                encoding="utf-8",
            )
            rows = list(iter_jsonl(path))
            self.assertEqual([line for line, _ in rows], [1, 2])


class HelperTests(unittest.TestCase):
    def test_fnv1a_64_reference_vectors(self):
        self.assertEqual(fnv1a_64(""), 0xCBF29CE484222325)
        self.assertEqual(fnv1a_64("a"), 0xAF63DC4C8601EC8C)

    def test_conversation_metrics(self):
        value = record()
        self.assertEqual(conversation_chars(value), len("Hello, Mikuru!") + len("Hi there!"))
        self.assertEqual(role_counts(value), {"user": 1, "assistant": 1})

    def test_summarise_empty(self):
        self.assertEqual(summarise([]), {"min": None, "mean": None, "max": None})


if __name__ == "__main__":
    unittest.main()
