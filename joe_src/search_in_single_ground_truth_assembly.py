"""
A script to evaluate IoU in the single ground truth assembly
defined by the central node.
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

from src.build_graphs import make_all_graphs
from src.utils import *

def find_network_iou(
        central_body_id, 
        matching_neighboring_bodies, 
        graph
    ):
    ground_truth_neighbors = set([n for n in graph.neighbors(central_body_id)])
    predicted_neighbors = set(matching_neighboring_bodies)

    return iou_from_two_sets(ground_truth_neighbors, predicted_neighbors)


def find_random_iou(central_body_id, num_predicted_neighbors, graph):
    assert num_predicted_neighbors >= 1, "Must predict 1 neighbor"
    assert graph.number_of_nodes() > num_predicted_neighbors, \
        "Assembly must have more bodies than num_predicted_neighbors"
    
    nodes = list(graph.nodes)
    nodes.remove(central_body_id)

    # This loop is a kind of monte carlo simulation
    # to find the expected IoU by chance alone
    ious = []
    for i in range(100):
        random_neighbors = set(random.sample(nodes, k=num_predicted_neighbors))
        ground_truth_neighbors = set([n for n in graph.neighbors(central_body_id)])
        iou = iou_from_two_sets(ground_truth_neighbors, random_neighbors)
        ious.append(iou)
    return np.array(ious).mean()

def find_best_matching_bodies(
        neighbor_seq_embeddings, 
        part_embeddings, 
        assembly_node_indices_without_central_node, 
        index_to_body
    ):
    assert assembly_node_indices_without_central_node.shape[0] >= neighbor_seq_embeddings.shape[0], \
        "Must have as many nodes in the assembly as in the predicted sequence"
    part_embeddings_for_assembly = part_embeddings[assembly_node_indices_without_central_node]

    seq_len = neighbor_seq_embeddings.shape[0]

    # Build the big dists matrix
    dists_mat = []
    for i in range(seq_len):
        part_emb = torch.unsqueeze(neighbor_seq_embeddings[i], dim=0)
        dists_to_emb = torch.norm(part_embeddings_for_assembly-part_emb, dim=1)
        dists_mat.append(dists_to_emb)
    dists_mat = torch.stack(dists_mat, dim=1)
    assert dists_mat.shape == (assembly_node_indices_without_central_node.shape[0], seq_len), "Check dists shape"


    # Now we will set dists_for_assembly cols equal to inf
    # each time we choose a node
    invalid_id = assembly_node_indices_without_central_node.max() + 1
    bodies_matching_sequence = torch.ones((seq_len)).long() * invalid_id
    assert bodies_matching_sequence.shape[0] == seq_len, "Check seq len"
    all_part_dists = []
    for i in range(seq_len):
        # Find the best match from an embedding in the 
        # sequence to a body in the assembly
        index = torch.argmin(dists_mat)
        body_in_assembly, index_in_sequence = unravel_index(index, dists_mat.shape)
        assert body_in_assembly < dists_mat.shape[0], "Check a body index"
        assert index_in_sequence < dists_mat.shape[1], "Check a sequence index"
        assert index_in_sequence < seq_len, "Index off sequence"

        all_part_dists.append(dists_mat[body_in_assembly, index_in_sequence].clone())

        # Don't pick the same body or sequence index twice
        dists_mat[body_in_assembly, :] =  torch.inf
        dists_mat[:, index_in_sequence] =  torch.inf
        body_index = assembly_node_indices_without_central_node[body_in_assembly]
        bodies_matching_sequence[index_in_sequence] = body_index

    assert bodies_matching_sequence.max() < invalid_id, "Bad index"

    all_part_dists = torch.stack(all_part_dists)
    mean_dist = all_part_dists.mean()
    max_dist = all_part_dists.max()
    min_dist = all_part_dists.min()

    matching_neighboring_bodies = []
    for index in bodies_matching_sequence:
        matching_neighboring_bodies.append(index_to_body[index.item()])

    return matching_neighboring_bodies, mean_dist

def search_in_ground_truth_assembly(
        assembly_dataset, 
        metadata_pathname, 
        part_embeddings_file, 
        sequence_embeddings_file, 
        sequence_central_nodes_file, 
        output
    ):
    # Load the data
    metadata = load_json(metadata_pathname)
    graphs = make_all_graphs(assembly_dataset, metadata_pathname)
    part_embeddings = load_embeddings(part_embeddings_file)
    sequence_embeddings = load_embeddings(sequence_embeddings_file)
    sequence_central_nodes = load_json(sequence_central_nodes_file)  

    assert part_embeddings.shape[0] == len(metadata), "Should have metadata for each body"
    assert len(sequence_central_nodes) == sequence_embeddings.shape[0], "Should have data for each embedding"

    # Look up table for going from bodies to the indices of the embeddings
    body_to_index, index_to_body = make_body_to_index(metadata)
    assembly_node_lists = make_assembly_node_lists(graphs, body_to_index)
    results = []
    for index, seq_central_node_info in enumerate(tqdm(sequence_central_nodes)):
        seq_embedding = sequence_embeddings[index]
        seq_len = find_sequence_length(seq_embedding)
        seq_embedding = seq_embedding[:seq_len]

        assembly_id = seq_central_node_info["aid"]
        graph = graphs[assembly_id]

        if graph.number_of_nodes() < seq_len:
            # In this case we can't match the assembly to the
            # predicted parts
            continue


        check_assembly_id, central_body_id = split_meta_string(seq_central_node_info["c_part_md"])
        assert check_assembly_id == assembly_id, "Should have same assembly id"
        central_node_index = body_to_index[central_body_id]
        assembly_node_list = assembly_node_lists[assembly_id]
        assembly_nodes_without_central_node = assembly_node_list[assembly_node_list != central_node_index]
        assert assembly_nodes_without_central_node.shape[0] == assembly_node_list.shape[0] - 1, "Should remove just 1"
        matching_neighboring_bodies, mean_dist = find_best_matching_bodies(seq_embedding[1:], part_embeddings, assembly_nodes_without_central_node, index_to_body)

        iou_network = find_network_iou(central_body_id, matching_neighboring_bodies, graph)
        iou_random = find_random_iou(central_body_id, len(matching_neighboring_bodies), graph)
        sequence_data = {
            "sequence_index": index,
            "central_body_id": central_body_id,
            "matching_neighboring_bodies": matching_neighboring_bodies,
            "assembly_id": assembly_id,
            "assembly_num_bodies": graph.number_of_nodes(),
            "assembly_num_contacts": graph.number_of_edges(),
            "dist": mean_dist.item(),
            "iou_network": iou_network,
            "random_baseline_iou": iou_random
        }
        results.append(sequence_data)
        
    with open(output, "w") as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False, sort_keys=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assembly_dataset", type=str, required=True, help="json files containing assemblies")
    p.add_argument("--metadata", type=str, required=True, help="Metadata telling us what assemblies to include")
    p.add_argument("--part_embeddings", type=str, required=True, help="Embeddings of each part")
    p.add_argument("--sequence_embeddings", type=str, required=True, help="Embeddings of generated sequence")
    p.add_argument("--sequence_central_nodes", type=str, required=True, help="Assembly and central nodes for each embedding")
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
    sequence_central_nodes = Path(args.sequence_central_nodes)
    
    search_in_ground_truth_assembly(assembly_dataset, metadata, part_embeddings, sequence_embeddings, sequence_central_nodes, output)