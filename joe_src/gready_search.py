"""
Perfrom a gready search to match decoded embeddings
to the best matching assemblies
"""
import argparse
from pathlib import Path
import networkx as nx
import json
from tqdm import tqdm 
import random
from collections import defaultdict
import numpy as np
import torch

from src.build_graphs import make_all_graphs, load_json, split_meta_string
from src.utils import *



def find_random_baseline_iou(graph, predicted_seq_length):
    assert predicted_seq_length > 1, "Predicted sequence of length 1 is invalid"
    assert graph.number_of_nodes() >= predicted_seq_length, \
        "Number of nodes in graph should not be smaller than predicted_seq_length"
    
    nodes = list(graph.nodes)
    random_central_node = random.choice(nodes)
    nodes.remove(random_central_node)
    random_neighbors = set(random.sample(nodes, k=predicted_seq_length-1))
    ground_truth_neighbors = set([n for n in graph.neighbors(random_central_node)])

    return iou_from_two_sets(ground_truth_neighbors, random_neighbors)


def find_random_neighbors_baseline_iou(graph, matching_body_ids):
    predicted_seq_length = len(matching_body_ids)
    assert predicted_seq_length > 1, "Predicted sequence of length 1 is invalid"
    assert graph.number_of_nodes() >= predicted_seq_length, \
        "Number of nodes in graph should not be smaller than predicted_seq_length"
    
    central_node_from_search = matching_body_ids[0]
    nodes = list(graph.nodes)
    nodes.remove(central_node_from_search)

    random_neighbors = set(random.sample(nodes, k=predicted_seq_length-1))
    ground_truth_neighbors = set([n for n in graph.neighbors(central_node_from_search)])
    return iou_from_two_sets(ground_truth_neighbors, random_neighbors)


def find_top_k_iou_data(
        k_value, 
        assembly_ids, 
        dists, 
        matching_body_indices, 
        graphs,
        index_to_body
    ):
    top_k_assembly_info = []
    ious_from_search = []
    iou_random = []
    iou_random_neighbors = []
    
    for i in range(k_value):
        assembly_id = assembly_ids[i]
        dist = dists[i].item()
        matching_body_indices_for_assembly = matching_body_indices[i]
        matching_body_ids_for_assembly = find_matching_body_ids(index_to_body, matching_body_indices_for_assembly)
        graph = graphs[assembly_id]
        iou_from_search = find_iou(matching_body_ids_for_assembly, graph)
        iou_random = find_random_baseline_iou(graph, len(matching_body_ids_for_assembly))
        iou_random_neighbors = find_random_neighbors_baseline_iou(graph, matching_body_ids_for_assembly)
        
        top_k_assembly_info.append(
            {
                "assembly_id": assembly_id,
                "assembly_num_bodies": graph.number_of_nodes(),
                "dist": dist,
                "matching_body_ids": matching_body_ids_for_assembly,
                "iou_from_search": iou_from_search,
                "iou_random": iou_random,
                "iou_random_neighbors": iou_random_neighbors
            }
        )
    iou_data = {
        "k": k_value,
        "assembly_info": top_k_assembly_info,
        "top_k_iou_from_search": np.array(iou_from_search).max(),
        "top_k_iou_random": np.array(iou_random).max(),
        "top_k_iou_random_neighbors": np.array(iou_random_neighbors).max()
    }
    return iou_data



def gready_dist_to_assembly(assembly_nodes, dists_mat):
    # dists_mat.shape = (num_parts_to_search, seq_length)
    # assembly_nodes.shape = (num_bodies_in_assembly)
    num_parts_to_search = dists_mat.shape[0]
    seq_length = dists_mat.shape[1]
    assert assembly_nodes.shape[0] >= seq_length, "Cant match seq to smaller assembly"

    dists_for_assembly = dists_mat[assembly_nodes]
    # dists_for_assembly.shape = (num_bodies_in_assembly, seq_length)

    # Now we will set dists_for_assembly cols equal to inf
    # each time we choose a node
    bodies_matching_sequence = torch.ones((seq_length)).long() * num_parts_to_search
    all_part_dists = []
    for i in range(seq_length):
        # Find the best match from an embedding in the 
        # sequence to a body in the assembly
        index = torch.argmin(dists_for_assembly)
        body_in_assembly, index_in_sequence = unravel_index(index, dists_for_assembly.shape)
        assert body_in_assembly < dists_for_assembly.shape[0], "Check a body index"
        assert index_in_sequence < dists_for_assembly.shape[1], "Check a sequence index"

        all_part_dists.append(dists_for_assembly[body_in_assembly, index_in_sequence].clone())

        # Don't pick the same body or sequence index twice
        dists_for_assembly[body_in_assembly, :] =  torch.inf
        dists_for_assembly[:, index_in_sequence] =  torch.inf
        body_index = assembly_nodes[body_in_assembly]
        bodies_matching_sequence[index_in_sequence] = body_index

    assert bodies_matching_sequence.max() < num_parts_to_search, "Bad index"

    all_part_dists = torch.stack(all_part_dists)
    mean_dist = all_part_dists.mean()
    max_dist = all_part_dists.max()
    min_dist = all_part_dists.min()

    return mean_dist, bodies_matching_sequence

def find_dists_to_assemblies(seq, part_embeddings, assembly_node_lists):
    # Truncate the sequence at the pad
    seq_len = find_sequence_length(seq)
    seq = seq[:seq_len]

    # Build the big dists matrix
    dists_mat = []
    for i in range(seq_len):
        part_emb = torch.unsqueeze(seq[i], dim=0)
        dists_to_emb = torch.norm(part_embeddings-part_emb, dim=1)
        dists_mat.append(dists_to_emb)
    dists_mat = torch.stack(dists_mat, dim=1)

    # dists_mat.shape = (num_part_embeddings, seq_len)
    assert dists_mat.shape == (part_embeddings.shape[0], seq_len), \
        "Check dist mat shape"
    
    dists_to_assemblies = []
    assembly_ids = []
    matching_body_indices = []
    for assembly_id in assembly_node_lists:
        assembly_nodes = assembly_node_lists[assembly_id]
        if assembly_nodes.shape[0] < seq_len:
            continue
        dist_to_assembly, bodies_matching_sequence = gready_dist_to_assembly(assembly_nodes, dists_mat)
        assembly_ids.append(assembly_id)
        dists_to_assemblies.append(dist_to_assembly)
        matching_body_indices.append(bodies_matching_sequence)

    dists_to_assemblies = torch.stack(dists_to_assemblies)
    indices_to_sort = torch.argsort(dists_to_assemblies)
    assembly_ids_sorted = []
    matching_body_indices_sorted = []
    dists_to_assemblies_sorted = dists_to_assemblies[indices_to_sort]
    for index in indices_to_sort:
        assembly_ids_sorted.append(assembly_ids[index])
        matching_body_indices_sorted.append(matching_body_indices[index])

    return assembly_ids_sorted, dists_to_assemblies_sorted, matching_body_indices_sorted

def gready_search(
        assembly_dataset, 
        metadata_pathname, 
        part_embeddings_file, 
        sequence_embeddings_file, 
        output
    ):
    # Load the data
    metadata = load_json(metadata_pathname)
    graphs = make_all_graphs(assembly_dataset, metadata_pathname)
    part_embeddings = load_embeddings(part_embeddings_file)
    sequence_embeddings = load_embeddings(sequence_embeddings_file)

    assert part_embeddings.shape[0] == len(metadata), "Should have metadata for each body"

    # Look up table for going from bodies to the indices of the embeddings
    body_to_index, index_to_body = make_body_to_index(metadata)
    assembly_node_lists = make_assembly_node_lists(graphs, body_to_index)

    num_sequence_embeddings = sequence_embeddings.shape[0]
    random_seq_indices = random.sample(range(num_sequence_embeddings), k=1000)

    k_values = [1, 5, 10]
    search_results = []
    for seq_index in tqdm(range(num_sequence_embeddings)):
        seq = sequence_embeddings[seq_index]
        assembly_ids, dists, matching_body_indices = find_dists_to_assemblies(seq, part_embeddings, assembly_node_lists)

        sequence_data = {
            "sequence_index": seq_index,
            "top_k_ious": []
        }

        for k_value in k_values:
            iou_data = find_top_k_iou_data(k_value, assembly_ids, dists, matching_body_indices, graphs, index_to_body)
            sequence_data["top_k_ious"].append(iou_data)
        
        search_results.append(sequence_data)
    
    with open(output, "w") as fp:
        json.dump(search_results, fp, indent=4, ensure_ascii=False, sort_keys=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assembly_dataset", type=str, required=True, help="json files containing assemblies")
    p.add_argument("--metadata", type=str, required=True, help="Metadata telling us what assemblies to include")
    p.add_argument("--part_embeddings", type=str, required=True, help="Embeddings of each part")
    p.add_argument("--sequence_embeddings", type=str, required=True, help="Embeddings of generated sequence")
    p.add_argument("--output", type=str, required=True, help="Output json file")
    
    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assembly_dataset = Path(args.assembly_dataset)
    metadata = Path(args.metadata)
    output = Path(args.output)
    part_embeddings = Path(args.part_embeddings)
    sequence_embeddings = Path(args.sequence_embeddings)
    
    gready_search(assembly_dataset, metadata, part_embeddings, sequence_embeddings, output)