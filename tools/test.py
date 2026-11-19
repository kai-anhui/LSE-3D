"""Test using a frozen threshold, or calibrate on the validation split only."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
from engine.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    args.command = "calibrate" if args.calibrate else "evaluate"
    args.split = args.split or ("val" if args.calibrate else "test")
    evaluate(args)


if __name__ == "__main__":
    main()
