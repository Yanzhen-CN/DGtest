from __future__ import annotations

import unittest

from eval_benchmarks import build_code_program
from score_eval import normalize_gsm8k_number, score_gsm8k, score_math500


class Gsm8kScoringTests(unittest.TestCase):
    def test_requires_explicit_final_answer_line(self) -> None:
        row = {
            "sample_id": "GSM8K/test",
            "gold_answer": "42",
            "generated_text": "The intermediate result is 42, but no final line was emitted.",
        }
        outcome = score_gsm8k(row)
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.detail, "missing_or_invalid_final_answer")

    def test_exact_final_answer_normalizes_commas(self) -> None:
        row = {
            "sample_id": "GSM8K/test",
            "gold_answer": "1,234",
            "generated_text": "Reasoning.\nFinal answer: $1,234$.",
        }
        outcome = score_gsm8k(row)
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.extracted_answer, "1234")

    def test_rejects_non_numeric_final_payload(self) -> None:
        self.assertEqual(normalize_gsm8k_number(r"\boxed{42}"), "")


class Math500ScoringTests(unittest.TestCase):
    def test_symbolically_equivalent_boxed_answer(self) -> None:
        row = {
            "sample_id": "MATH500/test",
            "gold_answer": r"\frac{1}{2}",
            "generated_text": r"Work. Therefore, $\boxed{0.5}$.",
        }
        try:
            outcome = score_math500(row, precision=6)
        except SystemExit as exc:
            self.skipTest(str(exc))
        self.assertTrue(outcome.passed)


class MbppExecutionTests(unittest.TestCase):
    def test_sanitized_imports_precede_generated_code(self) -> None:
        row = {
            "benchmark": "mbpp",
            "generated_text": "def area(x):\n    return math.floor(x)",
            "test_imports": ["import math"],
            "test_list": ["assert area(1.2) == 1"],
        }
        program = build_code_program("mbpp", row)
        self.assertTrue(program.startswith("import math\n"))
        self.assertNotIn("challenge", program)


if __name__ == "__main__":
    unittest.main()
