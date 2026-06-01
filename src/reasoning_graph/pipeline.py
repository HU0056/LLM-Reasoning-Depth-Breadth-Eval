from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .dataset import ensure_directory, load_gsm8k_dataset, write_jsonl
from .graph_builder import build_reasoning_edges
from .schemas import GraphPayload, SamplePayload
from .sentence_parser import extract_final_answer, split_sentences
from .similarity import calculate_similarity


def download_gsm8k(config: PipelineConfig) -> list[Path]:
    dataset = load_gsm8k_dataset()
    ensure_directory(config.raw_dir)

    written_files: list[Path] = []
    for split_name, split_dataset in dataset.items():
        rows = [
            {
                "gsm8k_id": f"gsm8k_{split_name}_{index:05d}",
                "question": sample["question"],
                "answer": sample["answer"],
            }
            for index, sample in enumerate(split_dataset)
        ]
        output_path = config.raw_dir / f"{split_name}.jsonl"
        write_jsonl(output_path, rows)
        written_files.append(output_path)

    return written_files


def build_gsm8k_graphs(config: PipelineConfig) -> list[Path]:
    dataset = load_gsm8k_dataset()
    ensure_directory(config.processed_dir)

    written_files: list[Path] = []
    for split_name, split_dataset in dataset.items():
        payloads = []
        for index, sample in enumerate(split_dataset):
            sample_id = f"gsm8k_{split_name}_{index:05d}"
            question_nodes = split_sentences(sample["question"])
            answer_nodes = split_sentences(sample["answer"])
            gold_answer = extract_final_answer(answer_nodes)
            edges = build_reasoning_edges(
                question_nodes=question_nodes,
                answer_nodes=answer_nodes,
                similarity_fn=calculate_similarity,
                bound=config.bound,
            )
            graph = GraphPayload(nodes=question_nodes + answer_nodes, edges=edges)
            payloads.append(
                SamplePayload(
                    id=sample_id,
                    gsm8k_id=sample_id,
                    task_type="math",
                    question=sample["question"],
                    gold_answer=gold_answer,
                    gold_reasoning_graph=graph,
                ).to_dict()
            )

        output_path = config.processed_dir / f"{split_name}_graphs.jsonl"
        write_jsonl(output_path, payloads)
        written_files.append(output_path)

    return written_files
