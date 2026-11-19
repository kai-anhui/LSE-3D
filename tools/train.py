"""Training entry point: python -m tools.train or python tools/train.py."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
from engine.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lse3d_exp01.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/lse3d_exp01")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--allow-unverified", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
