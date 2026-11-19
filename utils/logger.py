"""JSONL training metrics, also emitted to the console."""

import json
from pathlib import Path


def log_metrics(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(metrics, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)
