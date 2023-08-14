"""
Compute an estimate of the expected IoU for
matching a sequence of length m with and assembly of 
length n.
"""
import argparse
from pathlib import Path
import networkx as nx
import json
from tqdm import tqdm 
import random
from collections import defaultdict
import numpy as np

from src.build_graphs import make_all_graphs

def iou_from_guess(graph, seq_length, k):
    ious = []
    for i in range(k):
        nodes = list(graph.nodes)
        random_neighbors = set(random.sample(nodes, k=seq_length))
        random_central = random.sample(random_neighbors, k=1)[0]
        random_neighbors.remove(random_central)

        found_neighbors = set([ n for n in graph.neighbors(random_central)])

        intersection = random_neighbors.intersection(found_neighbors)
        union = random_neighbors.union(found_neighbors)
        iou = len(intersection) / len(union)
        ious.append(iou)
    return np.array(ious).max()




def random_baseline(assembly_dataset, metadata_pathname, output):
    graphs = make_all_graphs(assembly_dataset, metadata_pathname)
    k_values = [1, 5, 10]
    ious = defaultdict(dict)
    mean_num_edges = defaultdict(list)
    ious_by_assembly = {}
    count = 0
    for graph_id in tqdm(graphs):
        count +=1
        # For fast debug
        # if count > 100:
        #     break
        graph = graphs[graph_id]
        num_nodes = graph.number_of_nodes()
        mean_num_edges[num_nodes].append(graph.number_of_edges())
        if num_nodes >= 2:
            data_for_assembly = []
            for seq_length in range(2, min(num_nodes, 9)+1):
                
                # This is the loop for the top_k
                for k in k_values:
                    num_random_guesses = 100
                    ious_random = []
                    # This loop is for 100 random guesses as a kind of
                    # "monte carlo simulation" 
                    for i in range(num_random_guesses):
                        iou = iou_from_guess(graph, seq_length, k)
                        if num_nodes == 2:
                            assert iou >= 1.0, "Wrong"
                        ious_random.append(iou)
                    iou = np.array(ious_random).mean()
                    data_for_assembly.append(
                        {
                            "num_nodes": num_nodes,
                            "seq_length": seq_length,
                            "k": k,
                            "iou": iou
                        }
                    )

                    if not seq_length in ious[num_nodes]:
                        ious[num_nodes][seq_length] = {}
                    if not k in ious[num_nodes][seq_length]:
                        ious[num_nodes][seq_length][k] = []
                    ious[num_nodes][seq_length][k].append(iou)
            ious_by_assembly[graph_id] = data_for_assembly

    ious_output = []
    for num_nodes in sorted(ious.keys()):
        for seq_length in sorted(ious[num_nodes].keys()):
            for k in k_values:
                iou = np.array(ious[num_nodes][seq_length][k]).mean()
                print(f"Assembly num node: {num_nodes},  Sequence length {seq_length}, Top k {k},  Baseline IoU {iou}")
                ious_output.append(
                    {
                        "num_nodes": num_nodes,
                        "seq_length": seq_length,
                        "k": k,
                        "iou": iou,
                        "num_assemblies": len(ious[num_nodes][seq_length][k])
                    }
                )
                
    
    mean_num_edges_output = []
    for n in sorted(mean_num_edges.keys()):
        mean_num_edges_output.append(
            {
                "num_bodies_in_assembly": n,
                "mean_num_edges": np.array(mean_num_edges[n]).mean()
            }
        )
        

    data = {
        "ious": ious_output,
        "ious_by_assembly": ious_by_assembly,
        "mean_number_of_edges": mean_num_edges_output
    }
    with open(output, "w") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assembly_dataset", type=str, required=True, help="json files containing assemblies")
    p.add_argument("--metadata", type=str, required=True, help="Metadata telling us what assemblies to include")
    p.add_argument("--output", type=str, required=True, help="Output json file")
    
    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assembly_dataset = Path(args.assembly_dataset)
    metadata = Path(args.metadata)
    output = Path(args.output)
    
    random_baseline(assembly_dataset, metadata, output)