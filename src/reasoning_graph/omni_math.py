from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from .dataset import ensure_directory, write_jsonl
from .schemas import GraphPayload, SamplePayload
from .sentence_parser import extract_final_answer, split_sentences


OMNI_MATH_DATASET = "KbsdJames/Omni-MATH"
OMNI_MATH_SPLIT = "test"
OMNI_MATH_MIRROR_URL = (
    "https://hf-mirror.com/datasets/KbsdJames/Omni-MATH/resolve/main/test.jsonl"
)
GSM8K_DATASET_CANDIDATES = ("openai/gsm8k", "gsm8k")
GSM8K_CONFIG = "main"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_THINKING = "disabled"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "high"
JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class DependencyValidationError(ValueError):
    """Raised when a model dependency response cannot be converted to graph edges."""


def load_omni_math_rows(raw_path: Path) -> list[dict[str, Any]]:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        dataset = load_dataset(OMNI_MATH_DATASET, split=OMNI_MATH_SPLIT)
        rows = [dict(row) for row in dataset]
    except Exception:
        rows = _download_omni_math_jsonl()

    ensure_directory(raw_path.parent)
    write_jsonl(raw_path, rows)
    return rows


def _download_omni_math_jsonl() -> list[dict[str, Any]]:
    request = urllib.request.Request(OMNI_MATH_MIRROR_URL)
    rows: list[dict[str, Any]] = []
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsm8k_rows(raw_path: Path, split: str) -> list[dict[str, Any]]:
    if raw_path.exists():
        return read_jsonl(raw_path)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    last_error: Exception | None = None
    for dataset_name in GSM8K_DATASET_CANDIDATES:
        try:
            dataset = load_dataset(dataset_name, GSM8K_CONFIG, split=split)
            rows = [
                {
                    "gsm8k_id": f"gsm8k_{split}_{index:05d}",
                    "question": row["question"],
                    "answer": row["answer"],
                }
                for index, row in enumerate(dataset)
            ]
            write_jsonl(raw_path, rows)
            return rows
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"failed to load GSM8K {split} split: {last_error}") from last_error


def build_omni_math_payload(row: dict[str, Any], index: int) -> dict[str, Any]:
    sample_id = f"omni_math_test_{index:05d}"
    problem = _required_str(row, "problem")
    solution = _required_str(row, "solution")
    answer = _required_str(row, "answer")

    question_nodes = split_sentences(problem)
    reasoning_nodes = split_sentences(solution) + [f"#### {answer}"]
    graph = GraphPayload(nodes=question_nodes + reasoning_nodes, edges=[])
    return SamplePayload(
        id=sample_id,
        gsm8k_id=sample_id,
        task_type="math",
        question=problem,
        gold_answer=answer,
        gold_reasoning_graph=graph,
    ).to_dict()


def build_gsm8k_payload(row: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    sample_id = _optional_str(row, "gsm8k_id") or f"gsm8k_{split}_{index:05d}"
    question = _required_str(row, "question")
    answer_text = _required_str(row, "answer")

    question_nodes = split_sentences(question)
    answer_nodes = split_sentences(answer_text)
    gold_answer = extract_final_answer(answer_nodes)
    graph = GraphPayload(nodes=question_nodes + answer_nodes, edges=[])
    return SamplePayload(
        id=sample_id,
        gsm8k_id=sample_id,
        task_type="math",
        question=question,
        gold_answer=gold_answer,
        gold_reasoning_graph=graph,
    ).to_dict()


def _required_str(row: dict[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Omni-MATH row is missing non-empty string field: {field_name}")
    return value.strip()


def _optional_str(row: dict[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_dependency_response(content: str) -> dict[str, Any]:
    fenced = JSON_FENCE_RE.match(content)
    if fenced:
        content = fenced.group(1)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DependencyValidationError(f"DeepSeek response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("dependencies"), list):
        raise DependencyValidationError("response must be an object with a dependencies list")
    return parsed


def dependencies_to_edges(
    response_payload: dict[str, Any],
    *,
    first_reasoning_index: int,
    node_count: int,
    mode: str,
) -> list[list[int]]:
    if mode not in {"std", "test"}:
        raise ValueError("mode must be 'std' or 'test'")

    target_to_predecessors: dict[int, set[int]] = {
        target: set() for target in range(first_reasoning_index, node_count)
    }
    seen_targets: set[int] = set()

    for item in response_payload["dependencies"]:
        if not isinstance(item, dict):
            raise DependencyValidationError("each dependency item must be an object")
        target = _coerce_index(item.get("target"), "dependency target")
        predecessors = item.get("predecessors")
        if target < first_reasoning_index or target >= node_count:
            raise DependencyValidationError(f"target {target} is not a reasoning node")
        if not isinstance(predecessors, list):
            raise DependencyValidationError("predecessors must be a list")

        seen_targets.add(target)
        for predecessor in predecessors:
            predecessor = _coerce_index(predecessor, "predecessor index")
            if predecessor < 0 or predecessor >= node_count:
                raise DependencyValidationError(
                    f"predecessor {predecessor} is out of range for target {target}"
                )
            if predecessor == target:
                raise DependencyValidationError(
                    f"self dependency is not allowed for target {target}"
                )
            target_to_predecessors[target].add(predecessor)

    missing_targets = set(target_to_predecessors) - seen_targets
    if mode == "std" and missing_targets:
        missing = ", ".join(str(target) for target in sorted(missing_targets))
        raise DependencyValidationError(f"std mode requires dependencies for: {missing}")

    if mode == "std":
        empty_targets = [
            target
            for target, predecessors in target_to_predecessors.items()
            if not predecessors
        ]
        if empty_targets:
            empty = ", ".join(str(target) for target in empty_targets)
            raise DependencyValidationError(f"std mode requires non-empty predecessors for: {empty}")

    edges: list[list[int]] = []
    for target, predecessors in sorted(target_to_predecessors.items()):
        for predecessor in sorted(predecessors):
            edges.append([predecessor, target])
    _validate_acyclic(edges, node_count)
    return edges


def _validate_acyclic(edges: list[list[int]], node_count: int) -> None:
    successors: list[list[int]] = [[] for _ in range(node_count)]
    for predecessor, target in edges:
        successors[predecessor].append(target)

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int, path: list[int]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle_start = path.index(node)
            cycle = path[cycle_start:]
            cycle_nodes = cycle[:-1] if len(cycle) > 1 and cycle[0] == cycle[-1] else cycle
            cycle_text = " -> ".join(str(index) for index in cycle_nodes + [cycle_nodes[0]])
            cycle_edges = ", ".join(
                f"{source}->{target}"
                for source, target in zip(
                    cycle_nodes,
                    cycle_nodes[1:] + [cycle_nodes[0]],
                )
            )
            raise DependencyValidationError(
                "dependency graph contains a cycle: "
                f"{cycle_text}. Remove or redirect at least one of these edges: {cycle_edges}"
            )

        visiting.add(node)
        for successor in successors[node]:
            visit(successor, path + [successor])
        visiting.remove(node)
        visited.add(node)

    for node in range(node_count):
        visit(node, [node])


def _coerce_index(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise DependencyValidationError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise DependencyValidationError(f"{label} must be an integer")


def build_dependency_prompt(
    nodes: list[str],
    first_reasoning_index: int,
    mode: str,
    *,
    previous_response: str | None = None,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    numbered_nodes = "\n".join(f"{index}: {node}" for index, node in enumerate(nodes))
    target_indexes = list(range(first_reasoning_index, len(nodes)))
    empty_rule = (
        "Every target must have at least one predecessor."
        if mode == "std"
        else "A target may have an empty predecessors list if no prior node is directly needed."
    )
    system_prompt = (
        "You label sentence-level reasoning graph edges for math solutions. "
        "Return strict JSON only. Do not include markdown or commentary."
    )
    user_prompt = (
        "Given numbered nodes from a math problem and its reference reasoning, "
        "for each reasoning-stage target node, choose the prior, direct, minimal "
        "predecessor nodes needed to derive that target.\n\n"
        "Rules:\n"
        "- Only create dependencies for the listed target nodes.\n"
        "- The predecessor index is mostly smaller than the target index, except when the proof is written out of order.\n"
        # "- A predecessor may be before or after the target in the text when the proof is written out of order.\n"
        "- Never use the target itself as a predecessor.\n"
        "- The complete directed graph must be acyclic; never create circular dependencies.\n"
        # "- Include exactly one dependency object for every target index, in ascending order.\n"
        "- Multiple predecessors are allowed and welcome when they are all logically necessary.\n"
        "- Prefer direct and minimal dependencies; omit redundant ancestors, but do not force a single predecessor.\n"
        "- If a target states a claim before its proof, it may depend on later proof nodes as long as the graph remains acyclic.\n"
        "- If a claim depends on later proof nodes, those proof nodes must not also depend on the claim; use definitions, assumptions, case setup, or earlier algebra as their predecessors.\n"
        "- When fixing a cycle, remove or redirect the weakest edge in the cycle rather than dropping all useful dependencies.\n"
        f"- {empty_rule}\n"
        "- Return exactly this JSON shape: "
        '{"dependencies":[{"target":4,"predecessors":[0,2]}]}.\n\n'
        f"Target node indexes: {target_indexes}\n\n"
        f"Nodes:\n{numbered_nodes}"
    )
    if previous_response is not None and validation_error is not None:
        user_prompt += (
            "\n\nYour previous answer failed validation and must be corrected.\n"
            f"Validation error: {validation_error}\n"
            f"Previous answer:\n{previous_response}\n\n"
            "Return a corrected strict JSON object only."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class DeepSeekClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.thinking = (
            thinking
            or os.getenv("DEEPSEEK_THINKING", DEFAULT_DEEPSEEK_THINKING)
        ).lower()
        self.reasoning_effort = (
            reasoning_effort
            or os.getenv("DEEPSEEK_REASONING_EFFORT", DEFAULT_DEEPSEEK_REASONING_EFFORT)
        ).lower()
        if not self.api_key or self.api_key == "your_api_key_here":
            raise RuntimeError("DEEPSEEK_API_KEY or API_KEY is not configured.")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("DEEPSEEK_THINKING must be 'enabled' or 'disabled'")
        if self.reasoning_effort not in {"high", "max"}:
            raise ValueError("DEEPSEEK_REASONING_EFFORT must be 'high' or 'max'")

    def _request_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.thinking},
        }
        if self.thinking == "enabled":
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = 0
        return payload

    def chat_json(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(self._request_payload(messages), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected DeepSeek response: {payload}") from exc


def annotate_payload_edges(
    payload: dict[str, Any],
    *,
    mode: str,
    client: DeepSeekClient,
    max_retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    nodes = payload["gold_reasoning_graph"]["nodes"]
    question_nodes = split_sentences(payload["question"])
    first_reasoning_index = len(question_nodes)
    last_error: Exception | None = None
    previous_response: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            messages = build_dependency_prompt(
                nodes,
                first_reasoning_index,
                mode,
                previous_response=previous_response,
                validation_error=str(last_error) if last_error else None,
            )
            response_content = client.chat_json(messages)
            previous_response = response_content
            response = parse_dependency_response(response_content)
            edges = dependencies_to_edges(
                response,
                first_reasoning_index=first_reasoning_index,
                node_count=len(nodes),
                mode=mode,
            )
            payload["gold_reasoning_graph"]["edges"] = edges
            print(f"success at attempt {attempt}")
            return payload
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt} unqualified, continuing...")
            if attempt < max_retries and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"failed to annotate {payload['id']}: {last_error}") from last_error


def iter_selected_rows(
    rows: list[dict[str, Any]],
    *,
    start: int,
    limit: int | None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    stop = len(rows) if limit is None else min(len(rows), start + limit)
    for index in range(start, stop):
        yield index, rows[index]


def read_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def error_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_errors.jsonl")


def load_rows_for_args(args: argparse.Namespace, raw_path: Path) -> list[dict[str, Any]]:
    if args.dataset == "omni_math":
        if args.split != OMNI_MATH_SPLIT:
            raise ValueError("Omni-MATH only supports the test split")
        return load_omni_math_rows(raw_path)
    return load_gsm8k_rows(raw_path, args.split)


def build_payload_for_args(
    args: argparse.Namespace,
    row: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    if args.dataset == "omni_math":
        return build_omni_math_payload(row, index)
    return build_gsm8k_payload(row, index, args.split)


def raw_path_for_args(root: Path, args: argparse.Namespace) -> Path:
    return root / "data" / "raw" / args.dataset / f"{args.split}.jsonl"


def output_path_for_args(root: Path, args: argparse.Namespace) -> Path:
    return (
        root
        / "data"
        / "processed"
        / args.dataset
        / f"{args.split}_graphs_{args.mode}.jsonl"
    )


def run_omni_math_build(args: argparse.Namespace) -> Path:
    root = args.root
    raw_path = raw_path_for_args(root, args)
    output_path = output_path_for_args(root, args)
    rows = load_rows_for_args(args, raw_path)
    existing_ids = {
        row.get("id") for row in read_existing_jsonl(output_path)
    } if args.resume else set()
    client = DeepSeekClient(
        model=args.model,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )

    if not args.resume and output_path.exists():
        output_path.unlink()

    for index, row in iter_selected_rows(rows, start=args.start, limit=args.limit):
        payload = build_payload_for_args(args, row, index)
        if payload["id"] in existing_ids:
            continue
        try:
            annotated = annotate_payload_edges(
                payload,
                mode=args.mode,
                client=client,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep,
            )
        except RuntimeError as exc:
            if args.on_error == "stop":
                raise
            append_jsonl(
                error_output_path(output_path),
                {"id": payload["id"], "index": index, "error": str(exc)},
            )
            print(f"skipped {payload['id']}: {exc}")
            continue
        append_jsonl(output_path, annotated)
        print(f"wrote {annotated['id']}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DeepSeek reasoning graphs.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset",
        choices=["omni_math", "gsm8k"],
        default="omni_math",
        help="Dataset to process. Defaults to omni_math.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Dataset split. Omni-MATH only has test; GSM8K supports train/test.",
    )
    parser.add_argument("--mode", choices=["std", "test"], default="std")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "DeepSeek model name. Defaults to DEEPSEEK_MODEL or "
            f"{DEFAULT_DEEPSEEK_MODEL}."
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help=(
            "DeepSeek thinking mode. Use 'enabled' for reasoning mode. "
            "Defaults to DEEPSEEK_THINKING or disabled."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["high", "max"],
        default=None,
        help=(
            "Reasoning effort when thinking is enabled. Defaults to "
            "DEEPSEEK_REASONING_EFFORT or high."
        ),
    )
    parser.add_argument(
        "--on-error",
        choices=["stop", "skip"],
        default="stop",
        help="Stop at the first failed sample, or log the failure and continue.",
    )
    return parser
