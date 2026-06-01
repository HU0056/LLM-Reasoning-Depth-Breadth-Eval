from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class GraphPayload:
    nodes: list[str]
    edges: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SamplePayload:
    id: str
    gsm8k_id: str
    task_type: str
    question: str
    gold_answer: str
    gold_reasoning_graph: GraphPayload

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_reasoning_graph"] = self.gold_reasoning_graph.to_dict()
        return payload
