from pathlib import Path

from reasoning_graph import PipelineConfig, build_gsm8k_graphs, download_gsm8k


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = PipelineConfig(root_dir=root)
    download_gsm8k(config)
    build_gsm8k_graphs(config)


if __name__ == "__main__":
    main()
