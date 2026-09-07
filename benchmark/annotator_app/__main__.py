"""Entry point: python3 -m benchmark.annotator_app [--port 5001]"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.annotator_app.server import serve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--samples", type=Path, default=Path("benchmark/data/samples"))
    p.add_argument("--annotated", type=Path, default=Path("benchmark/data/annotated"))
    args = p.parse_args()
    serve(
        host=args.host,
        port=args.port,
        samples_dir=args.samples,
        annotated_dir=args.annotated,
    )


if __name__ == "__main__":
    main()
