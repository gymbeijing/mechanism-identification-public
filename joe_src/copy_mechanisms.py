"""
Copy step files with mechanisms into a folder
"""
import argparse
from pathlib import Path
import json
from collections import defaultdict
import copy
import shutil

def find_num_matching_strings(step_file, body_names):
    body_names_copy = copy.deepcopy(body_names)
    num_matches = 0
    with open(step_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            if "MANIFOLD_SOLID_BREP" in line:
                found = []
                for body_name in body_names_copy:
                    if body_name in line:
                        num_matches += 1
                        found.append(body_name)
                        break
                for f in found:
                    body_names_copy.remove(f)
    return num_matches

def parse_guid(filestem):
    split_bits = filestem.split("_") 
    assert len(split_bits) > 3, "Should split to at least 3"
    return split_bits[1]

def cache_data(guid_to_files, cache_path):
    cache = {}
    for guid in guid_to_files:
        cache[guid] = []
        for file in guid_to_files[guid]:
            cache[guid].append(str(file))
    with open(cache_path, "w") as fp:
        json.dump(cache, fp, indent=4, ensure_ascii=False, sort_keys=False)

def doc_guid_to_step_files(step_folder):
    cache_path = step_folder.parent / "guid_to_files.json"
    if cache_path.exists():
        with open(cache_path, "r") as fp:
            return json.load(fp)
        
    guid_to_files = defaultdict(list)
    for file in step_folder.glob("*/*.step"):
        guid = parse_guid(file.stem)
        guid_to_files[guid].append(file)
    cache_data(guid_to_files, cache_path)
    return guid_to_files


def copy_mechanisms(mechanisms_json, step_folder, output_folder):
    with open(mechanisms_json, "r") as fp:
        mechanisms_data = json.load(fp)

    guid_to_step = doc_guid_to_step_files(step_folder)

    mech_types = defaultdict(list)
    for doc in mechanisms_data:
        for match in doc["matches"]:
            mech_type = match["query"]
            mech_types[mech_type].append(doc)

    for mech_type in mech_types:
        folder_name = mech_type.lower().replace(" ", "_")
        type_output_folder = output_folder / folder_name
        type_output_folder.mkdir(exist_ok=True)

        for doc in mech_types[mech_type]:
            doc_guid = doc["document_guid"]
            matches = doc["matches"]
            body_names = []
            for m in matches:
                if m["query"] == mech_type:
                    body_names.append(m["body_name"])
            step_files = guid_to_step[doc_guid]
            for step_file in step_files:
                num_matches = find_num_matching_strings(step_file, body_names)
                if num_matches < len(body_names):
                    print(f"File {Path(step_file).stem} contains only {num_matches} matches for the {len(body_names)} expected body names")
                    continue
                best_step_for_doc = Path(step_file)
                dest = type_output_folder / (best_step_for_doc.stem + best_step_for_doc.suffix)
                shutil.copy(str(step_file), str(dest))
                break




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mechanisms_json", type=str, required=True, help="json file with mechanisms")
    p.add_argument("--step_folder", type=str, required=True, help="Folder containing STEP files")
    p.add_argument("--output_folder", type=str, required=True, help="Folder containing STEP files")
    
    args = p.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    mechanisms_json = Path(args.mechanisms_json)
    step_folder = Path(args.step_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)
    copy_mechanisms(mechanisms_json, step_folder, output_folder)



