"""Upgrade a JEM checkpoint written before WintonForcing split its precipitation.

JEM f759ae4 / e6b2d3c replaced WintonForcing's single `snowfall` field (flat
index 6) with `evaporation`, `liquid_precipitation` and `frozen_precipitation`,
so an older carry.msgpack holds two leaves fewer than the model expects and no
longer loads. The coupler overwrites the ice forcing at the start of every
step (the exchanger runs before the ice), so the stored forcing values never
reach the integration: the old `snowfall` leaf is replaced by three zero
leaves of its shape and dtype, the later forcing leaves are renumbered, and the
WintonForcing node in the stored tree structure gains the two extra leaves.

The file is rewritten in place, atomically, after the original is copied to
<file>.pre-precip-split. A checkpoint already in the new layout is left
untouched, so running this before every resubmission is safe.

Example (from the case directory):
    python tools/upgrade_winton_checkpoint.py output_T31/winton_ctl/checkpoint/carry.msgpack
"""
import argparse
import os
import shutil
from pathlib import Path

import flax.serialization
import numpy as np

FORCING = "['components']['seaice']['forcing']"
OLD_FIELDS, NEW_FIELDS, SNOWFALL = 12, 14, 6
OLD_NODE = "CustomNode(WintonForcing[None], [" + ", ".join(["*"] * OLD_FIELDS) + "])"
NEW_NODE = "CustomNode(WintonForcing[None], [" + ", ".join(["*"] * NEW_FIELDS) + "])"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("carry_file", type=Path, help="The coupled carry.msgpack of a run's checkpoint.")
args = parser.parse_args()

payload = flax.serialization.msgpack_restore(args.carry_file.read_bytes())
leaves, paths, structure = list(payload["leaves"]), list(payload["leaf_paths"]), payload["tree_structure"]


def forcing_path(k):
    return f"{FORCING}[<flat index {k}>]"


if NEW_NODE in structure:
    print(f"{args.carry_file}: already in the new layout, left as it is")
    raise SystemExit(0)
if structure.count(OLD_NODE) != 1:
    raise SystemExit(f"{args.carry_file}: no single {OLD_FIELDS}-field WintonForcing node in the stored structure")
first = paths.index(forcing_path(0))
if paths[first:first + OLD_FIELDS] != [forcing_path(k) for k in range(OLD_FIELDS)] or \
        forcing_path(OLD_FIELDS) in paths:
    raise SystemExit(f"{args.carry_file}: the sea-ice forcing leaves are not the expected {OLD_FIELDS}")

snowfall = np.asarray(leaves[first + SNOWFALL])
split = [np.zeros_like(snowfall) for _ in range(NEW_FIELDS - OLD_FIELDS + 1)]
leaves = leaves[:first + SNOWFALL] + split + leaves[first + SNOWFALL + 1:]
paths = paths[:first] + [forcing_path(k) for k in range(NEW_FIELDS)] + paths[first + OLD_FIELDS:]
upgraded = {"leaves": leaves, "leaf_paths": paths, "tree_structure": structure.replace(OLD_NODE, NEW_NODE)}

backup = args.carry_file.with_name(args.carry_file.name + ".pre-precip-split")
shutil.copy2(args.carry_file, backup)
temporary = args.carry_file.with_name(args.carry_file.name + ".tmp")
with open(temporary, "wb") as f:
    f.write(flax.serialization.msgpack_serialize(upgraded))
    f.flush()
    os.fsync(f.fileno())
os.replace(temporary, args.carry_file)
print(f"{args.carry_file}: snowfall leaf split into evaporation, liquid and frozen precipitation "
      f"({len(leaves)} leaves); original kept as {backup.name}")
