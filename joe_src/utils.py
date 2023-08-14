"""
Useful utility functions
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

def split_meta_string(s):
    # Example: 0000_100029_94515530_f130d4f0-054c-11ec-a094-0a2b8709b52f_0000
    #          Chunk   Assembly id               Body id                  Image
    split_str = s.split("_")
    assert len(split_str) == 5, "Should split into 5 chunks"
    assembly_id = f"{split_str[1]}_{split_str[2]}"
    body_id = split_str[3]
    return assembly_id, body_id

def assemblies_and_bodies(metadata):
    assemblies = defaultdict(set)
    for m in metadata:
        assembly_id, body_id = split_meta_string(m)
        assemblies[assembly_id].add(body_id)
    return assemblies

def load_json(filename):
    with open(filename, "r") as fp:
        return json.load(fp)
    
def load_embeddings(file):
    return torch.load(file)

def load_k_embeddings(folder, k=1):
    files = list(folder.glob('*.pt'))
    sampled_files = random.choices(files, k=k)
    embedding_list = []
    for file in sampled_files:
        embedding_list.append(load_embeddings(file))

    return embedding_list

def iou_from_two_sets(a, b):
    intersection = a.intersection(b)
    union = b.union(b)
    return len(intersection)/len(union)


def find_iou(matching_body_ids, graph):
    # The first index is the central node
    central_node = matching_body_ids[0]

    ground_truth_neighbors = set([n for n in graph.neighbors(central_node)])
    predicted_neighbors = set([n for n in matching_body_ids[1:]])
    assert len(predicted_neighbors) == len(matching_body_ids)-1, "Check slice"

    return iou_from_two_sets(ground_truth_neighbors, predicted_neighbors)



def make_body_to_index(metadata):
    body_to_index = {}
    index_to_body = {}
    for index, m in enumerate(metadata):
        assembly_id, body_id = split_meta_string(m)
        body_to_index[body_id] = index
        index_to_body[index] = body_id
    return body_to_index, index_to_body

def make_assembly_node_lists(graphs, body_to_index):
    """
    For each graph, find the indices of the embeddings 
    as a torch LongTensor
    """
    assembly_node_lists = {}
    for assembly_id in graphs:
        graph = graphs[assembly_id]
        node_indices = []
        for body_id in graph.nodes:
            body_index = body_to_index[body_id]
            node_indices.append(body_index)
        node_indices = torch.Tensor(node_indices).long()
        assembly_node_lists[assembly_id] = node_indices
    return assembly_node_lists

def find_sequence_length(seq):
    seq_norms = torch.norm(seq, dim=1)
    eps = 0.5
    if seq_norms.min() >= eps:
        return seq_norms.shape[0]
    pad_index = torch.argmax((seq_norms < eps).long())
    return pad_index

def find_matching_body_ids(index_to_body, matching_body_indices):
    matching_body_ids = []
    for index in matching_body_indices:
        matching_body_ids.append(index_to_body[index.item()])
    return matching_body_ids

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))