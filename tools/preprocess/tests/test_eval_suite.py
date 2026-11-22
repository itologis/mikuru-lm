#!/usr/bin/env python3
"""Unit tests for the evaluation harness parsing and classification."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_suite import classify_stdout, extract_continuation  # noqa: E402


class ContinuationTests(unittest.TestCase):
    def test_extracts_text_after_marker(self):
        stdout = "Generated continuation:\nHello there!\n"
        self.assertEqual(extract_continuation(stdout), "Hello there!")

    def test_missing_marker_returns_none(self):
        self.assertIsNone(extract_continuation("no marker here"))

    def test_empty_generation_is_a_failure(self):
        result = classify_stdout("Generated continuation:\n\n")
        self.assertFalse(result["ok"], "an empty generation must not be reported as ok")
        self.assertIn("empty generation", result["error"])
        self.assertEqual(result["text"], "")

    def test_whitespace_only_generation_is_a_failure(self):
        result = classify_stdout("Generated continuation:\n   \n")
        self.assertFalse(result["ok"], "whitespace-only output is still an empty generation")

    def test_missing_marker_is_a_failure(self):
        result = classify_stdout("something unexpected")
        self.assertFalse(result["ok"])
        self.assertIsNone(result["text"])

    def test_nonempty_generation_passes(self):
        result = classify_stdout("Generated continuation:\nI'm a model.\n")
        self.assertTrue(result["ok"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["text"], "I'm a model.")


if __name__ == "__main__":
    unittest.main()
