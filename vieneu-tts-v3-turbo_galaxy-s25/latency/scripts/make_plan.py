#!/usr/bin/env python3
"""Write a six-round order using all permutations of the three finalists."""
import argparse
from itertools import permutations
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("full")
parser.add_argument("step")
parser.add_argument("pico")
parser.add_argument("output", type=Path)
args = parser.parse_args()
rows = [f"{arm}\t{index}\t{3 if position == 0 and index > 0 else 0}"
        for index, order in enumerate(permutations((args.full, args.step, args.pico)))
        for position, arm in enumerate(order)]
rows.extend((f"{args.full}\t-3", f"{args.step}\t-3"))
args.output.write_text("\n".join(rows) + "\n")
