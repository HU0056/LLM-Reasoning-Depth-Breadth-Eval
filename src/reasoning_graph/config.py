from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_BOUND = 1 - 1 / 2.718281828459045


@dataclass(slots=True)
class PipelineConfig:
    root_dir: Path
    bound: float = DEFAULT_BOUND

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "gsm8k"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed" / "gsm8k"
