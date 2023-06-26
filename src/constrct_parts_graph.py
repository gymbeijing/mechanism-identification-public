import networkx as nx
import json
import logging
import argparse
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def read_assembly_id(file_path):
    """
    :param file_path: relative file path of the txt file that stores all the assembly ids
    :return: a string list of assembly ids
    """
    with open(file_path, 'r') as f:
        assembly_id_list = f.read().splitlines()
    return assembly_id_list[3:]   # remove the first three "_"


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly_id_path', type=str, help='Path to the file that stores all the processed assembly ids')
    parser.add_argument('--assembly_folder', type=str, help='Path to the folder that stores all the assembly.json files')
    args = parser.parse_args()

    assembly_id_path = args.assembly_id_path   # should be "../raw_data/processed.txt"
    assembly_id_list = read_assembly_id(assembly_id_path)
    logging.info(f'Found {len(assembly_id_list)} assembly ids in {args.assembly_id_path}')

    assembly_folder = args.assembly_folder   # should be "../raw_data/assembly"
	assembly_json_list = glob.glob
