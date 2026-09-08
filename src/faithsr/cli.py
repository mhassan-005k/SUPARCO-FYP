from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .inference import process_tiff


def infer(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, state = load_checkpoint(args.checkpoint, device)
    audit = process_tiff(args.input, args.output, model, device, args.tile_size, args.overlap, f"faithsr3x-epoch-{state.get('epoch', 'unknown')}")
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="faithsr"); commands = parser.add_subparsers(required=True)
    command = commands.add_parser("infer", help="Super-resolve an 8-bit TIFF by exactly 3x")
    command.add_argument("--input", required=True); command.add_argument("--output", required=True); command.add_argument("--checkpoint", required=True)
    command.add_argument("--device"); command.add_argument("--tile-size", type=int, default=512); command.add_argument("--overlap", type=int, default=32); command.set_defaults(handler=infer)
    args = parser.parse_args(); args.handler(args)


if __name__ == "__main__": main()
