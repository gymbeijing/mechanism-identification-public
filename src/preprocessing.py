import networkx as nx
import json

def read_assembly_ids(file_path):
    """
    :param file_path: relative file path of the txt file that stores all the assembly ids
    :return: a string list of assembly ids
    """
    with open(file_path, 'r') as f:
        assembly_id_list = f.read().splitlines()
    return assembly_id_list[3:]   # remove the first three "_"

def obtain_and_save_embeddings(assembly_id_list, assembly_json_folder):
    for assembly_id in assembly_id_list:
        json_file_path = assembly_json_folder + "/" + assembly_id + "_assembly.json"
        with open(json_file_path, 'r') as f:
            assembly = json.load(f)
            # print(assembly)

    return 0;

def construct_part_graphs(assembly_id_list):
    return 0;


def generate_and_save_orderings(part_graphs, part_emnbeddings):
    return;


if __name__ == '__main__':
    assembly_id_filepath = "../raw_data/processed.txt"
    assembly_id_list = read_assembly_ids(assembly_id_filepath)
    print(f'Found {len(assembly_id_list)} assembly ids')
    assembly_json_folder = "../raw_data/assembly_jsons"
    part_embeddings = obtain_and_save_embeddings(assembly_id_list, assembly_json_folder)
    part_graphs = construct_part_graphs(assembly_id_list)
    generate_and_save_orderings(part_graphs, part_embeddings)
