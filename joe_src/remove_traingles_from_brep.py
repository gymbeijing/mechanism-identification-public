"""
Load an OCC native BREP model and save it without the triangles.
How much space so we save
"""

import argparse
from pathlib import Path
import json
from collections import defaultdict
import copy
import shutil
import numpy as np

from occwl.compound import Compound

def make_directory(file, input, output):
    sub_dirs = []
    parent = file.parent
    while parent != input:
        sub_dirs.append(parent.stem)
        parent = parent.parent
    sub_dirs.reverse()
    output_dir = output
    for s in sub_dirs:
        output_dir = output_dir / s
        output_dir.mkdir(exist_ok=True)
    return output_dir

def process_file(file, input, output):
    size_before = file.stat().st_size
    output_folder = make_directory(file, input, output)
    shp = Compound.load_from_occ_native(file)
    output_file = output_folder / (file.stem + ".brep")
    shp.save_to_occ_native(output_file)
    size_after = output_file.stat().st_size
    percent_saving = (size_before-size_after)/size_before
    print(f"File {file.stem}  Before {size_before}  After {size_after}   Saving {percent_saving}%")
    return percent_saving

def remove_traingles(input, output):
    savings = []
    for file in input.glob("**/*.brep"):
        percent_saving = process_file(file, input, output)
        savings.append(percent_saving)
    savings = np.array(savings)
    print(f"Mean saving without triangles {savings.mean()}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Path to folders containing BREP data")
    p.add_argument("--output", type=str, required=True, help="Folder to save processed files")
    
    args = p.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    input = Path(args.input)
    output = Path(args.output)
    output.mkdir(exist_ok=True)
    remove_traingles(input, output)