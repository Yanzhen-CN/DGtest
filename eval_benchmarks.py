from __future__ import annotations

import re
from typing import Any


BENCHMARK_SPECS = {
    "humaneval": ("openai/openai_humaneval", None, "test"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
}


def load_benchmark(name: str, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("[ERROR] install benchmark dependencies: pip install -r requirements.txt") from exc

    if name not in BENCHMARK_SPECS:
        raise SystemExit(f"[ERROR] unsupported benchmark: {name}")
    repo, subset, split = BENCHMARK_SPECS[name]
    dataset = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [normalize_item(name, index, dict(row)) for index, row in enumerate(dataset)]


def normalize_item(name: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    if name == "humaneval":
        task_id = str(row.get("task_id", f"HumanEval/{index}"))
        source_prompt = str(row["prompt"])
        prompt = (
            "Complete the following Python function correctly. Return Python code only, without Markdown fences. "
            "Preserve the function name and signature.\n\n" + source_prompt
        )
        return {
            "sample_id": task_id,
            "prompt": prompt,
            "source_prompt": source_prompt,
            "entry_point": row["entry_point"],
            "test": row["test"],
            "reference": row.get("canonical_solution", ""),
        }
    if name == "mbpp":
        task_id = str(row.get("task_id", index))
        prompt = (
            "Write a correct Python solution for the task below. Return Python code only, without Markdown fences.\n\n"
            f"Task: {row['prompt'] if 'prompt' in row else row['text']}\n\n"
            "The solution must satisfy these tests:\n" + "\n".join(row.get("test_list", []))
        )
        return {
            "sample_id": f"MBPP/{task_id}",
            "prompt": prompt,
            "test_list": row.get("test_list", []),
            "test_setup_code": row.get("test_setup_code", ""),
            "challenge_test_list": row.get("challenge_test_list", []),
            "reference": row.get("code", ""),
        }
    if name == "gsm8k":
        answer = str(row["answer"])
        gold = answer.rsplit("####", 1)[-1].strip()
        return {
            "sample_id": f"GSM8K/{index}",
            "prompt": (
                "Solve this math problem step by step. End with a single line in the form `Final answer: <answer>`.\n\n"
                + str(row["question"])
            ),
            "reference": answer,
            "gold_answer": gold,
        }
    if name == "math500":
        gold = row.get("answer") or extract_boxed(str(row.get("solution", "")))
        return {
            "sample_id": str(row.get("unique_id", f"MATH500/{index}")),
            "prompt": (
                "Solve this mathematics problem with clear reasoning. Put the final answer in `\\boxed{}`.\n\n"
                + str(row["problem"])
            ),
            "reference": row.get("solution", ""),
            "gold_answer": str(gold),
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
        }
    raise AssertionError(name)


def extract_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text.strip()
    pos = start + len(marker)
    depth = 1
    for index in range(pos, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[pos:index].strip()
    return text[pos:].strip()


def extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced[-1] if fenced else text).strip()


def extract_numeric_answer(text: str) -> str:
    patterns = [
        r"(?i)final\s+answer\s*[:=]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip().rstrip(".$")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else ""


def build_code_program(benchmark: str, row: dict[str, Any]) -> str:
    prediction = extract_code(str(row.get("generated_text", "")))
    if benchmark == "humaneval":
        source_prompt = str(row.get("source_prompt", ""))
        code = prediction if re.search(r"\bdef\s+", prediction) else source_prompt + prediction
        return f"{code}\n\n{row['test']}\ncheck({row['entry_point']})\n"
    setup = str(row.get("test_setup_code", ""))
    tests = list(row.get("test_list", [])) + list(row.get("challenge_test_list", []))
    return "\n".join([setup, prediction, *tests]) + "\n"
