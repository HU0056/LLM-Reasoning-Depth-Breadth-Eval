from __future__ import annotations

import _bootstrap  # noqa: F401

from reasoning_graph.omni_math import build_arg_parser, run_omni_math_build

def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = run_omni_math_build(args)
    print(output_path)


if __name__ == "__main__":
    main()

